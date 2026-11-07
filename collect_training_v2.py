"""Filter the traceable external database before any corrected-task training.

Exclude every official train/test structure by neutralized fragment-parent
connectivity, including alternate stereochemistry. This is deliberately stricter
than the organizer's prohibition on external concentrations for test compounds.
No external record is used to overwrite a test prediction.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SOURCES = ['external.csv', 'primary_literature.csv', 'literature_candidates.csv', 'camphor2022.csv']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def parent_key(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError('Invalid structure')
    molecule = rdMolStandardize.FragmentParent(molecule)
    molecule = rdMolStandardize.Uncharger().uncharge(molecule)
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def build(output):
    training = pd.read_csv(DATA / 'train_v2.csv', usecols=['SMILES'])
    test = pd.read_csv(DATA / 'test.csv', usecols=['SMILES'])
    excluded = set(training.SMILES.map(parent_key)) | set(test.SMILES.map(parent_key))
    proposed, dropped = [], []
    numerical = pd.read_csv(DATA / SOURCES[0])
    for index, row in numerical[numerical.usable_for_training.eq(True)].iterrows():
        assert all(row[c + '_relation'] == '=' for c in ['IC50', 'CC50', 'SI'])
        assert row.IC50 > 0 and row.CC50 > 0 and pd.notna(row.doi)
        proposed.append({'SMILES': row.smiles, 'Culture': row.cell, 'strain': row.strain,
                         'IC50': row.IC50, 'CC50': row.CC50, 'EC50': np.nan, 'SI': row.SI,
                         'model_target': row.CC50 / row.IC50, 'doi': row.doi,
                         'outcome': 'Measured', 'source_url': row.source_url,
                         'source_file': SOURCES[0], 'source_row': int(index),
                         'target_basis': 'ratio_of_explicitly_paired_source_concentrations'})
    for name in SOURCES[1:]:
        primary = pd.read_csv(DATA / name)
        for index, row in primary[primary.retrieval_status.eq('eligible')].iterrows():
            inactive = row.outcome == 'Not active'
            value = 0. if inactive else row.SI_from_rounded_means
            if not np.isfinite(value) or value < 0:
                continue
            proposed.append({'SMILES': row.smiles, 'Culture': row.cell, 'strain': row.strain,
                             'IC50': row.IC50, 'CC50': row.CC50, 'EC50': row.get('EC50', np.nan),
                             'SI': row.SI if not inactive else np.nan,
                             'model_target': value, 'doi': row.doi, 'outcome': row.outcome,
                             'source_url': row.source_url, 'source_file': name,
                             'source_row': int(index),
                             'target_basis': 'explicit_inactive_task_convention_zero_not_measured_SI' if inactive else 'source_paired_ratio_of_rounded_means'})
    for row in proposed:
        row['parent_connectivity'] = parent_key(row['SMILES'])
        row['scaffold'] = MurckoScaffold.MurckoScaffoldSmiles(smiles=row['SMILES']) or 'ACYCLIC'
    # Conflicting eligible external values cannot be silently selected. Preserve
    # them in the source database but omit the entire ambiguous context here.
    frame = pd.DataFrame(proposed)
    keep, conflicts = [], 0
    for _, group in frame.groupby(['parent_connectivity', 'Culture', 'strain'], dropna=False):
        if group.parent_connectivity.iloc[0] in excluded:
            dropped.extend(group.index.tolist())
            continue
        values = group.model_target.to_numpy(float)
        if values.max() - values.min() > max(1., .05 * abs(values.mean())):
            conflicts += len(group)
            continue
        # Keep one observation per parent/context rather than weighting repeated
        # citations more heavily. Pick a deterministic source order, never a label.
        keep.append(group.sort_values(['source_file', 'source_row']).index[0])
    result = frame.loc[keep].sort_values(['source_file', 'source_row']).reset_index(drop=True)
    assert not set(result.parent_connectivity) & excluded
    assert np.isfinite(result.model_target.to_numpy()).all()
    assert result.doi.notna().all() and result.source_url.notna().all()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    report = {'task_version': 'official_train_v2_zero_inactivity_2026-11-07',
              'rows_proposed': len(proposed), 'rows_excluded_competition_parent': len(dropped),
              'rows_excluded_conflicting_external_context': conflicts,
              'rows_retained': len(result), 'parent_structures_retained': result.parent_connectivity.nunique(),
              'measured_rows': int(result.outcome.eq('Measured').sum()),
              'explicit_inactive_rows': int(result.outcome.eq('Not active').sum()),
              'test_parent_overlap': 0, 'train_parent_overlap': 0,
              'policy': 'Exclude all official train/test parent connectivities; no lookup or test-compound concentration feature is allowed.',
              'used_by_current_model': False,
              'source_sha256': {name: sha(DATA / name) for name in ['train_v2.csv', 'test.csv', *SOURCES]},
              'collector_sha256': sha(__file__), 'output_sha256': sha(output),
              'source_urls_and_snapshot_hashes': ['data/sources.json', 'data/evidence_sources.json',
                                                'data/primary_structure_evidence.json'],
              'limitations': ['Outside-competition external records have different assay contexts.',
                              'Model_target zero is a task convention for an explicit inactive annotation, not a measured SI.',
                              'Additional fold-level scaffold exclusions are required if this pool is used for scaffold validation.']}
    output.with_suffix('.validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return result, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA / 'external_training_v2.csv')
    build(parser.parse_args().output)
