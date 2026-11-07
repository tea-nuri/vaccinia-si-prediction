"""Rebuild an audited high-SI extension; competition parents are excluded first.

The output is a research pool, not an automatic replacement of the best model.
Concentrations are kept as measured endpoints, not used as model features.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from rdkit import Chem

import corrected_pipeline as cp
from collect_training_v2 import parent_key

MANIFEST = cp.DATA / 'high_si_sources.json'


def normalized_json_hash(raw):
    payload = json.loads(raw)
    # ChEMBL returns the same activity set in variable order. Preserve every
    # field and metadata value, while treating that one collection as unordered.
    if isinstance(payload, dict) and 'activities' in payload:
        payload['activities'] = sorted(payload['activities'], key=lambda r:r['activity_id'])
    # API nodes may omit optional null fields (observed for modality). Only
    # absent/null equivalence is allowed; all non-null information is checked.
    def without_null_fields(value):
        if isinstance(value, dict):
            return {k:without_null_fields(v) for k,v in value.items() if v is not None}
        if isinstance(value, list):
            return [without_null_fields(v) for v in value]
        return value
    payload = without_null_fields(payload)
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(canonical).hexdigest()


def sources(offline):
    manifest = json.loads(MANIFEST.read_text())
    result = {}
    with zipfile.ZipFile(cp.DATA / 'high_si_sources.zip') as archive:
        for entry in manifest['sources']:
            pinned = archive.read(entry['key'])
            if hashlib.sha256(pinned).hexdigest() != entry['sha256']:
                raise ValueError(f"Snapshot corrupted: {entry['key']}")
            result[entry['key']] = pinned
    if not offline:
        def verify(entry):
            for attempt in range(3):
                req = urllib.request.Request(entry['url'], headers={'User-Agent': 'Reproducible selectivity data audit'})
                try:
                    with urllib.request.urlopen(req, timeout=30) as response:
                        raw = response.read()
                    break
                except (urllib.error.URLError, TimeoutError) as error:
                    if attempt == 2:
                        raise RuntimeError(f"Cannot refresh {entry['key']}; use --offline for verified snapshots") from error
                    time.sleep(attempt + 1)
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                if (not entry.get('normalized_json_sha256')
                    or normalized_json_hash(raw) != entry['normalized_json_sha256']):
                    raise ValueError(f"Source changed: review {entry['key']} before admission")
            return entry['key']
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(verify, manifest['sources']))
    return manifest, result


def numeric_mean(text):
    if any(x in text for x in '<>≤≥'):
        raise ValueError('Censored concentration is not an exact mean')
    match = re.match(r'^\s*(\d+(?:\.\d+)?)', text)
    if not match:
        raise ValueError(f'Unexpected measurement: {text}')
    return float(match[1])


def build(offline=False, output=None):
    manifest, raw = sources(offline)
    train, test = cp.read_inputs()
    excluded = set(pd.concat([train.SMILES, test.SMILES]).map(parent_key))
    base_path = cp.DATA / 'external_training_candidate_v3.csv'
    if cp.sha(base_path) != manifest['base_sha256']:
        raise ValueError('Base pool changed')
    base = pd.read_csv(base_path)
    base_parents = set(base.SMILES.map(parent_key))
    new, rejected = [], []

    def admit(structure, identity, values):
        parent = parent_key(structure)
        # Do not obtain the target until the compound passes the exclusion gate.
        if parent in excluded or parent in base_parents:
            rejected.append({'record': identity, 'reason': 'competition_parent' if parent in excluded else 'existing_external_parent'})
            return
        row = values()
        row.update(SMILES=structure, parent_connectivity=parent,
                   scaffold=cp.molecule_features(structure)[1], outcome='Measured')
        if not np.isfinite(row['SI']) or row['SI'] < 0:
            raise ValueError('Invalid target')
        row['model_target'] = row['SI']
        new.append(row)

    article = ET.fromstring(raw['article.xml'])
    table = next(t for t in article.findall('.//table-wrap') if t.get('id') == 'tbl0010')
    table_rows = []
    previous_name = None
    previous_cc = None
    for tr in table.findall('.//tbody/tr'):
        cells = [' '.join(''.join(c.itertext()).split()) for c in tr]
        name = cells[0] or previous_name
        cc = cells[2] or previous_cc
        previous_name, previous_cc = name, cc
        table_rows.append((name, cells[1], cc, cells[3]))
    for name, ic, cc, si in table_rows:
        key = 'structures/' + name.replace(' ', '_') + '.json'
        if key not in raw:
            rejected.append({'record': name, 'reason': 'reference_compound_not_targeted'})
            continue
        prop = json.loads(raw[key])['PropertyTable']['Properties'][0]
        structure = prop['SMILES']
        mol = Chem.MolFromSmiles(structure)
        if mol is None or Chem.MolToInchiKey(mol) != prop['InChIKey']:
            raise ValueError('PubChem structure/InChIKey inconsistency')
        strain = 'Western Reserve' if '(WR)' in ic else 'Copenhagen'

        def paper_values(name=name, ic=ic, cc=cc, si=si, strain=strain, key=key, prop=prop):
            if 'nM' not in ic:
                raise ValueError('Expected nM potency; do not silently convert other units')
            potency, toxicity = numeric_mean(ic) / 1000, numeric_mean(cc)
            value = numeric_mean(si)
            return dict(compound_name=name, Culture='HeLa', strain=strain,
                        IC50=potency, CC50=toxicity, EC50=np.nan, SI=value,
                        IC50_relation='=', CC50_relation='=', SI_relation='=',
                        concentration_units='uM', potency_endpoint='IC50', toxicity_endpoint='CC50',
                        IC50_original=ic, CC50_original=cc, SI_original=si,
                        ratio_of_reported_means=toxicity / potency,
                        doi='10.1016/j.antiviral.2010.10.001',
                        source_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC9533862/',
                        source_table='Table 1', source_row=name + ' / ' + strain,
                        target_basis='direct_reported_exact_SI',
                        structure_url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{prop['CID']}",
                        source_sha256=hashlib.sha256(raw['article.xml']).hexdigest(),
                        structure_sha256=hashlib.sha256(raw[key]).hexdigest(),
                        context_note='Published in vitro plaque reduction endpoint; authors warn that IC50/SI can vary with endpoint. No claim of clinical efficacy.')

        admit(structure, name + ' / ' + strain, paper_values)

    doc = json.loads(raw['chembl/document.json'])
    if doc['doi'] != '10.1021/jm050473m':
        raise ValueError('Unexpected ChEMBL document')
    for record in json.loads(raw['chembl/assay.json'])['activities']:
        if record['activity_id'] not in [1690860, 1690861]:
            continue

        def chembl_values(record=record):
            if (record['standard_relation'] != '=' or record['standard_units'] is not None
                or record['data_validity_comment'] is not None or record['potential_duplicate'] != 0
                or record['document_chembl_id'] != doc['document_chembl_id']
                or record['assay_chembl_id'] != 'CHEMBL860528'
                or record['assay_description'] != 'Selectivity index, CC50 in HFF cell/ EC50 against vaccinia virus'):
                raise ValueError('Unreviewed selectivity record')
            return dict(compound_name=record['molecule_chembl_id'], Culture='HFF', strain=np.nan,
                        IC50=np.nan, CC50=np.nan, EC50=np.nan,
                        SI=float(record['standard_value']), SI_relation='=',
                        potency_endpoint='EC50', toxicity_endpoint='CC50',
                        doi=doc['doi'], source_row=record['activity_id'],
                        source_url=f"https://www.ebi.ac.uk/chembl/api/data/activity/{record['activity_id']}.json",
                        target_basis='direct_reported_exact_SI',
                        source_sha256=hashlib.sha256(raw['chembl/assay.json']).hexdigest(),
                        assay_id=record['assay_chembl_id'], assay_description=record['assay_description'],
                        context_note='Strain absent from curated record: preserved as missing. No reconstruction of concentration pairs from separate assays.')

        admit(record['canonical_smiles'], str(record['activity_id']), chembl_values)

    added = pd.DataFrame(new)
    if added.duplicated(['parent_connectivity', 'Culture', 'strain']).any():
        raise ValueError('Duplicate new parent/context')
    if set(added.parent_connectivity) & excluded:
        raise AssertionError('Competition overlap')
    combined = pd.concat([base, added], ignore_index=True)
    output = output or cp.DATA / 'external_training_high_si_v4.csv'
    combined.to_csv(output, index=False)
    added.to_csv(output.with_name(output.stem + '.added.csv'), index=False)
    report = dict(task_version=cp.TASK_VERSION, status='audited_research_pool',
                  base_rows=len(base), added_rows=len(added), output_rows=len(combined),
                  added_unique_molecules=added.parent_connectivity.nunique(),
                  added_high_SI_rows=int((added.SI > 1000).sum()),
                  output_high_SI_rows=int((combined.model_target > 1000).sum()),
                  competition_parent_overlap=0, rejected=rejected,
                  output_sha256=cp.sha(output), base_sha256=cp.sha(base_path),
                  script_sha256=cp.sha(__file__), sources_manifest_sha256=cp.sha(MANIFEST),
                  sources_archive_sha256=cp.sha(cp.DATA / 'high_si_sources.zip'),
                  limitations=['Small source-biased extension; not evidence of Kaggle gain.',
                               'Reported SI retained rather than replacing it with a rounded concentration ratio.',
                               'CC50/EC50 pairs and strain absent from ChEMBL records remain missing.',
                               'All validation scaffolds must be removed from the external pool in scaffold CV.'])
    cp.save_json(output.with_suffix('.validation.json'), report)
    print(json.dumps(report, indent=2))
    return combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    build(args.offline, args.output)
