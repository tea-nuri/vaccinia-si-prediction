"""Reproduce the 60-row external candidate pool from verified public sources.

The selected model does not automatically consume this pool. Every official
train/test parent is excluded, and validation must additionally exclude held-out
scaffolds. Published inequality bounds are never substituted as exact targets.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from collect_training_v2 import parent_key
from corrected_pipeline import ALLOWED, DATA, ROOT, TASK_VERSION, molecule_features, read_inputs, save_json, sha

MANIFEST = DATA / 'external_candidate_v3.sources.json'
BASE_SHA = '257ddc9f4ec8e600c8a16b7f7662bb77a88b68d3e5764a630edb5a8321657fc9'


def cached_sources(manifest, cache, offline):
    def fetch(source):
        key = Path(source['cache_key'])
        if key.is_absolute() or '..' in key.parts:
            raise ValueError('Unsafe source cache path')
        target = cache / key
        if target.exists():
            raw = target.read_bytes()
        else:
            if offline:
                raise FileNotFoundError(f'Offline source absent: {key}')
            request = urllib.request.Request(source['url'], headers={'User-Agent': 'Reproducible ML source audit'})
            with urllib.request.urlopen(request, timeout=40) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != source['sha256']:
                raise ValueError(f'Upstream source changed; audit before admission: {key}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.tmp')
            temporary.write_bytes(raw)
            temporary.replace(target)
        if hashlib.sha256(raw).hexdigest() != source['sha256']:
            raise ValueError(f'Cache hash mismatch: {key}')
        return str(key), raw
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(pool.map(fetch, manifest['sources']))


def printed_point_si(text):
    """SI is the final parenthesized number, independently of CC50 notation."""
    match = re.search(r'\(([^()]+)\)\s*$', text)
    if match and re.fullmatch(r'\d+(?:[.,]\d+)?', match.group(1)):
        return float(match.group(1).replace(',', '.'))
    return None


def unambiguous_mean(text):
    if any(c in text for c in ['<', '>', '≤', '≥', '/']):
        return None
    match = re.fullmatch(r'\s*(\d+(?:[.,]\d+)?)\s*(?:±[^()]*)?\s*', text)
    return float(match.group(1).replace(',', '.')) if match else None


def collect_chEMBL(manifest, sources, excluded, base_parents):
    proposed = []
    rejected = {'competition_parent': 0, 'existing_external_parent': 0}
    for key, raw in sources.items():
        if not key.startswith('chembl/') or not Path(key).name.startswith('activity_'):
            continue
        document_key = str(Path(key).parent / 'document.json')
        document = json.loads(sources[document_key])
        for row in json.loads(raw)['activities']:
            context = manifest['assays'].get(row['assay_chembl_id'])
            if context is None or row.get('standard_relation') != '=':
                continue
            parent = parent_key(row['canonical_smiles'])
            if parent in excluded:
                rejected['competition_parent'] += 1
                continue
            if parent in base_parents:
                rejected['existing_external_parent'] += 1
                continue
            if document['doi'].casefold() != context['doi'].casefold():
                raise ValueError('Assay DOI does not match the curated document')
            if row.get('data_validity_comment') is not None or row.get('potential_duplicate') != 0:
                raise ValueError('Unreviewed activity quality flag')
            if row.get('standard_units') is not None or row.get('units') is not None:
                raise ValueError('Selectivity must be dimensionless')
            if row['standard_type'] not in ['Ratio CC50/EC50', 'Selectivity Index']:
                raise ValueError('Not the curated cellular selectivity endpoint')
            description = row['assay_description']
            if 'CC50' not in description or 'Vaccinia' not in description:
                raise ValueError('Endpoint definition changed')
            value = float(row['standard_value'])
            if not np.isfinite(value) or value < 0:
                raise ValueError('Invalid selectivity')
            proposed.append({'SMILES': row['canonical_smiles'], 'Culture': context['culture'],
                             'strain': context['strain'], 'IC50': np.nan, 'CC50': np.nan,
                             'EC50': np.nan, 'SI': value, 'model_target': value,
                             'doi': context['doi'], 'outcome': 'Measured',
                             'source_url': f"https://www.ebi.ac.uk/chembl/api/data/activity/{row['activity_id']}.json",
                             'source_file': key, 'source_row': row['activity_id'],
                             'target_basis': 'direct_reported_exact_selectivity_not_recomputed',
                             'parent_connectivity': parent,
                             'scaffold': molecule_features(row['canonical_smiles'])[1],
                             'activity_id': row['activity_id'], 'assay_id': row['assay_chembl_id'],
                             'assay_description': description,
                             'source_sha256': hashlib.sha256(raw).hexdigest()})
    frame = pd.DataFrame(proposed)
    keep = []
    for _, group in frame.groupby(['parent_connectivity', 'Culture', 'strain'], dropna=False):
        if group.model_target.nunique() != 1:
            raise ValueError('Conflicting external parent/context labels')
        keep.append(group.sort_values('activity_id').index[0])
    return frame.loc[keep].sort_values('activity_id').reset_index(drop=True), rejected


def collect_paper(manifest, sources, excluded):
    paper = manifest['paper']
    article = ET.fromstring(sources['paper/article.xml'])
    table = next(t for t in article.findall('.//table-wrap') if t.get('id') == paper['table_id'])
    rows = {}
    for tr in table.findall('.//tr'):
        cells = [' '.join(''.join(c.itertext()).split()) for c in tr if c.tag in ['td', 'th']]
        if cells:
            rows[cells[0]] = cells
    admitted, rejected = [], []
    for label, name in paper['names'].items():
        resolver_raw = sources[f'paper/structures/{label}.json']
        resolver = json.loads(resolver_raw)
        if resolver['status'] != 'SUCCESS' or resolver.get('warnings'):
            raise ValueError('Name resolution needs review')
        smiles = resolver['smiles']
        parent = parent_key(smiles)
        if parent in excluded:
            rejected.append({'compound': label, 'reason': 'competition_parent'})
            continue
        # Exclusion precedes reading any target values for this structure.
        row = rows[label]
        si = printed_point_si(row[-1])
        if si is None:
            rejected.append({'compound': label, 'reason': 'censored_or_absent_SI',
                             'CC50_text': row[-2], 'EC50_text': row[-1]})
            continue
        ec = unambiguous_mean(re.sub(r'\s*\([^()]+\)\s*$', '', row[-1]))
        cc = unambiguous_mean(row[-2])
        if ec is None or ec <= 0:
            raise ValueError('Unexpected EC50 annotation')
        admitted.append({'SMILES': smiles, 'Culture': paper['culture'], 'strain': paper['strain'],
                         'IC50': np.nan, 'CC50': np.nan if cc is None else cc, 'EC50': ec,
                         'SI': si, 'model_target': si, 'doi': paper['doi'], 'outcome': 'Measured',
                         'source_url': paper['url'], 'source_file': 'PMC12846358_Table1', 'source_row': label,
                         'target_basis': 'direct_reported_exact_selectivity_not_recomputed',
                         'parent_connectivity': parent, 'scaffold': molecule_features(smiles)[1],
                         'compound_name': name['name'], 'structure_pdf_page': name['pdf_page'],
                         'structure_resolver_sha256': hashlib.sha256(resolver_raw).hexdigest(),
                         'structure_supplement_sha256': paper['supplement_pdf_sha256'],
                         'CC50_text': row[-2], 'EC50_text': row[-1], 'EC50_relation': '=',
                         'CC50_relation': '=' if cc is not None else 'ambiguous_duration_specific_values',
                         'concentration_units': 'uM', 'source_table': 'Table 1',
                         'context_note': 'Use printed SI; no ratio is reconstructed from CC50 cells containing multiple durations.'})
    return pd.DataFrame(admitted), rejected


def build(cache, output, offline=False):
    manifest = json.loads(MANIFEST.read_text())
    if manifest['task_version'] != TASK_VERSION:
        raise ValueError('Source manifest belongs to another task version')
    if sha(DATA / 'external_training_v2.csv') != BASE_SHA:
        raise ValueError('Base pool changed; audit before regeneration')
    train, test = read_inputs()
    excluded = set(pd.concat([train.SMILES, test.SMILES]).map(parent_key))
    base = pd.read_csv(DATA / 'external_training_v2.csv')
    sources = cached_sources(manifest, cache, offline)
    extra, chembl_rejected = collect_chEMBL(manifest, sources, excluded, set(base.parent_connectivity))
    paper, paper_rejected = collect_paper(manifest, sources, excluded)
    full = pd.concat([base, extra, paper], ignore_index=True)
    if set(full.SMILES.map(parent_key)) & excluded:
        raise ValueError('Competition compound entered the external pool')
    contexts = full[['parent_connectivity', 'Culture', 'strain']].fillna('UNKNOWN').astype(str)
    if contexts.duplicated().any() or not np.isfinite(full.model_target.to_numpy(float)).all():
        raise ValueError('Duplicate context or invalid model target')
    output.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(output, index=False)
    report = {'task_version': TASK_VERSION, 'status': 'candidate_pool_not_used_by_selected_model',
              'base_rows': len(base), 'new_chembl_rows': len(extra), 'new_2026_paper_rows': len(paper),
              'rows_retained': len(full), 'parent_structures': full.parent_connectivity.nunique(),
              'competition_parent_overlap': 0, 'chembl_rejections': chembl_rejected,
              'paper_rejections': paper_rejected, 'ambiguous_paper_CC50_left_missing': int(paper.CC50.isna().sum()),
              'base_sha256': BASE_SHA, 'manifest_sha256': sha(MANIFEST),
              'collector_sha256': sha(__file__), 'output_sha256': sha(output),
              'feature_allowlist': ALLOWED, 'used_by_selected_model': False,
              'limitations': ['Different papers use different assay contexts.',
                             'Printed selectivity is rounded; it is not a raw replicate.',
                             'Validation must exclude held-out scaffolds from external augmentation.',
                             'One explicit inactive base row has task-convention model_target=0, with measured SI left missing.']}
    save_json(output.with_suffix('.validation.json'), report)
    return full, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'cache/external_candidate_v3')
    parser.add_argument('--output', type=Path, default=DATA / 'external_training_candidate_v3.csv')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    _, result = build(args.cache_dir, args.output, args.offline)
    print(json.dumps(result, indent=2))
