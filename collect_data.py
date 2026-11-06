"""Collect traceable VACV IC50/CC50/SI data from explicitly matched ChEMBL assays."""
from pathlib import Path
import argparse, csv, hashlib, io, json, math, time, urllib.request, zipfile
from collections import Counter
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
API = 'https://www.ebi.ac.uk/chembl/api/data/'
# Assay descriptions explicitly identify the denominator, numerator and cell line.
# SI is the published value, not a ratio invented by joining unrelated experiments.
STUDIES = [
    dict(document='CHEMBL6092128', cell='Vero', strain='Copenhagen',
         ic='CHEMBL6092367', cc='CHEMBL6092368', si='CHEMBL6092369',
         duration='IC50 and CC50: 4 days'),
    dict(document='CHEMBL5625166', cell='Vero', strain='VACV; strain unspecified in assay record',
         ic='CHEMBL5625829', cc='CHEMBL5625830', si='CHEMBL5625831',
         duration='Published ratio: IC50 96 h; CC50 72 h'),
    dict(document='CHEMBL5625166', cell='A549', strain='VACV; strain unspecified in assay record',
         ic='CHEMBL5625832', cc='CHEMBL5625833', si='CHEMBL5625834',
         duration='Published ratio: IC50 96 h; CC50 72 h'),
    dict(document='CHEMBL5223146', cell='Vero', strain='Copenhagen',
         ic='CHEMBL5225279', cc='CHEMBL5225278', si='CHEMBL5225280',
         duration='IC50 and CC50: 4 days'),
    dict(document='CHEMBL4699424', cell='Vero', strain='Copenhagen',
         ic='CHEMBL4700293', cc='CHEMBL4700294', si='CHEMBL4700295',
         duration='Matched Table 1; antiviral assay: 4 days'),
    dict(document='CHEMBL4368876', cell='Vero', strain='Copenhagen',
         ic='CHEMBL4369494', cc='CHEMBL4369495', si='CHEMBL4369496',
         duration='IC50 and CC50: 4 days'),
    dict(document='CHEMBL2321819', cell='BSC1', strain='WR',
         ic='CHEMBL2330536', cc='CHEMBL2330528', si='CHEMBL2330527',
         duration='Published ratio: IC50 overnight; CC50 20 h'),
    dict(document='CHEMBL4257560', cell='Vero', strain='Copenhagen',
         ic='CHEMBL4260272', cc='CHEMBL4260273', si='CHEMBL4260274',
         cc_type='TC50', si_type='Selectivity Index',
         duration='IC50 and TC50: 4 days'),
    dict(document='CHEMBL4837211', cell='Vero', strain='Copenhagen',
         ic='CHEMBL4838274', cc='CHEMBL4838273', si='CHEMBL4838275',
         duration='IC50: 4 days; CC50 duration unspecified in assay record'),
    dict(document='CHEMBL4052722', cell='CV1', strain='LIVP-GFP',
         ic='CHEMBL4056706', cc='CHEMBL4056694', si='CHEMBL4056707',
         duration='Published ratio: IC50 48 h; CC50 24 h; 4 h pretreatment'),
]
FACTORS = {'nM': .001, 'uM': 1., 'µM': 1., 'mM': 1000., 'M': 1e6}


def measurement(record, kind, expected=None):
    if record is None:
        return '', '', 'missing_measurement'
    expected = expected or {'ic': 'IC50', 'cc': 'CC50', 'si': 'Ratio CC50/IC50'}[kind]
    if record['standard_type'] != expected:
        return '', '', 'wrong_endpoint'
    try:
        value = float(record['standard_value'])
    except (ValueError, TypeError):
        return '', '', 'missing_value'
    if not math.isfinite(value) or value <= 0:
        return '', '', 'invalid_value'
    unit = record.get('standard_units')
    if kind != 'si':
        if unit not in FACTORS:
            return '', '', 'unsupported_units'
        value *= FACTORS[unit]
    elif unit not in [None, '', 'dimensionless']:
        return '', '', 'unsupported_ratio_units'
    relation = record.get('standard_relation') or record.get('relation') or ''
    flag = ('data_validity_flag' if record.get('data_validity_comment')
            not in [None, '', 'Manually validated'] else '')
    return value, relation, flag


def fetch(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'TeaNuriHackathon/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            obj = json.loads(data)
            if 'error_message' in obj:
                raise ValueError(obj['error_message'])
            return data
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def load_sources(refresh=False, offline=False):
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / 'chembl_snapshot.zip'
    endpoints = {}
    for document in sorted({s['document'] for s in STUDIES}):
        endpoints[f'paper_{document}.json'] = API + f'activity.json?document_chembl_id={document}&limit=1000'
        endpoints[f'document_{document}.json'] = API + f'document/{document}.json'
    payloads = {}
    if archive.exists() and not refresh:
        with zipfile.ZipFile(archive) as z:
            payloads = {name: z.read(name) for name in endpoints if name in z.namelist()}
    # A saved snapshot must agree with its committed manifest before rebuilding.
    manifest_path = DATA / 'sources.json'
    if not refresh and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for name, raw in payloads.items():
            if name not in previous or hashlib.sha256(raw).hexdigest() != previous[name]['sha256']:
                raise ValueError(f'Snapshot integrity check failed: {name}')
    missing = set(endpoints)-set(payloads)
    if missing:
        if offline:
            raise FileNotFoundError('Incomplete data/chembl_snapshot.zip; missing: '+', '.join(sorted(missing)))
        for name in sorted(missing):
            print('Fetching '+endpoints[name],flush=True)
            payloads[name] = fetch(endpoints[name])
        temporary = archive.with_suffix('.tmp')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, raw in payloads.items():
                z.writestr(name,raw)
        temporary.replace(archive)
    manifest = {name: {'url': endpoints[name], 'sha256': hashlib.sha256(data).hexdigest()}
                for name, data in payloads.items()}
    (DATA / 'sources.json').write_text(json.dumps(manifest, indent=2))
    return {name: json.loads(data) for name, data in payloads.items()}


def build(payloads):
    rows = []
    for study in STUDIES:
        doc = study['document']
        paper = payloads[f'paper_{doc}.json']
        if paper['page_meta'].get('next'):
            raise ValueError('Paper has more than one page; extend pagination before use')
        records = paper['activities']
        lookup = {}
        for r in records:
            if r['document_chembl_id'] != doc:
                raise ValueError('Document mismatch')
            key = (r['molecule_chembl_id'], r['assay_chembl_id'])
            lookup.setdefault(key, []).append(r)
        molecules = sorted({r['molecule_chembl_id'] for r in records
                            if r['assay_chembl_id'] in [study['ic'], study['si']]})
        for molecule in molecules:
            selected = {}
            issues = []
            for kind in ['ic', 'cc', 'si']:
                candidates = lookup.get((molecule, study[kind]), [])
                if len(candidates) > 1:
                    raise ValueError(f'Ambiguous measurement: {doc}, {molecule}, {kind}')
                selected[kind] = candidates[0] if candidates else None
            si = selected['si']
            source = si or selected['ic']
            # The allowlisted SI assays explicitly refer to VACV and the paired cell line.
            description = source['assay_description'].lower()
            if not any(s in description for s in ['vaccinia', 'vacv', 'vavc']):
                raise ValueError('Selected SI assay does not describe VACV')
            if study['cell'].lower() not in description:
                raise ValueError('Cell context mismatch')
            mol = Chem.MolFromSmiles(source.get('canonical_smiles') or '')
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError(f'Invalid SMILES for {molecule}')
            row = dict(smiles=Chem.MolToSmiles(mol), IC50='', CC50='', SI='',
                       doi=payloads[f'document_{doc}.json'].get('doi') or '',
                       cell=study['cell'], strain=study['strain'], duration=study['duration'],
                       document_id=doc, molecule_id=molecule, concentration_units='uM')
            row['publication_url'] = 'https://doi.org/' + row['doi'] if row['doi'] else ''
            for kind, column in [('ic', 'IC50'), ('cc', 'CC50'), ('si', 'SI')]:
                val, relation, issue = measurement(selected[kind], kind, study.get(kind + '_type'))
                row[column] = val
                row[column + '_relation'] = relation
                row[column + '_activity_id'] = selected[kind]['activity_id'] if selected[kind] else ''
                row[column + '_assay_id'] = study[kind]
                row[column + '_source_type'] = selected[kind]['standard_type'] if selected[kind] else ''
                for field in ['activity_comment', 'standard_text_value', 'text_value']:
                    row[column + '_' + field] = (selected[kind].get(field) or '') if selected[kind] else ''
                if issue:
                    issues.append(column + ':' + issue)
                if relation != '=':
                    issues.append(column + ':bound_or_unknown_relation')
            row['SI_computed_for_check'] = row['CC50'] / row['IC50'] if row['CC50'] and row['IC50'] else ''
            if all(row[c] for c in ['IC50', 'CC50', 'SI']) and all(row[c + '_relation'] == '=' for c in ['IC50', 'CC50', 'SI']):
                if abs(row['SI'] - row['SI_computed_for_check']) > max(1., .05 * row['SI']):
                    issues.append('ratio_inconsistent_beyond_rounding')
            row['source_url'] = API + 'activity/' + str(source['activity_id']) + '.json'
            row['quality_flags'] = ';'.join(issues)
            row['usable_for_training'] = not issues
            rows.append(row)
    return rows


# Qualitative antiviral outcomes are kept separately from numerical SI values.
ANTIVIRAL_ENDPOINTS = {'Activity', 'Inhibition', 'Antiviral activity', 'IC50',
                       'EC50', 'EC90', 'MIC', 'MIC50', 'ID50', 'ED50'}
OUTCOME_FIELDS = ['activity_comment', 'standard_text_value', 'text_value']


def inactive_field(record):
    if record.get('standard_type') not in ANTIVIRAL_ENDPOINTS:
        return None
    return next((field for field in OUTCOME_FIELDS
                 if str(record.get(field)).strip().lower() == 'not active'), None)


def load_evidence_sources(refresh=False, offline=False):
    archive = DATA / 'vacv_evidence_snapshot.zip'
    payloads = {}
    if archive.exists() and not refresh:
        with zipfile.ZipFile(archive) as z:
            payloads = {name: z.read(name) for name in z.namelist()}
    manifest_path = DATA / 'evidence_sources.json'
    if not refresh and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for name, raw in payloads.items():
            if name not in previous or hashlib.sha256(raw).hexdigest() != previous[name]['sha256']:
                raise ValueError('Evidence snapshot integrity check failed: ' + name)
    endpoints, records = {}, []
    offset = 0
    changed = False

    def obtain(name, url):
        nonlocal changed
        endpoints[name] = url
        if name not in payloads:
            if offline:
                raise FileNotFoundError('Incomplete VACV evidence snapshot: ' + name)
            print('Fetching ' + url, flush=True)
            payloads[name] = fetch(url)
            changed = True
        return json.loads(payloads[name])

    while True:
        name = f'activity_{offset}.json'
        page = obtain(name, API + 'activity.json?target_organism=Orthopoxvirus+vaccinia'
                      + f'&limit=1000&offset={offset}')
        meta = page['page_meta']
        if meta['offset'] != offset or not page['activities']:
            raise ValueError('Invalid ChEMBL pagination')
        records.extend(page['activities'])
        if not meta.get('next'):
            if len(records) != meta['total_count']:
                raise ValueError('Incomplete ChEMBL activity snapshot')
            break
        offset += len(page['activities'])
    if len({r['activity_id'] for r in records}) != len(records):
        raise ValueError('Duplicate activity IDs in ChEMBL pagination')
    if any(r['target_organism'] != 'Orthopoxvirus vaccinia' for r in records):
        raise ValueError('Unexpected organism in VACV evidence')
    documents = {}
    for doc in sorted({r['document_chembl_id'] for r in records if inactive_field(r)}):
        documents[doc] = obtain(f'document_{doc}.json', API + f'document/{doc}.json')
    if changed:
        temporary = archive.with_suffix('.tmp')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in endpoints:
                z.writestr(name, payloads[name])
        temporary.replace(archive)
    manifest = {name: {'url': url, 'sha256': hashlib.sha256(payloads[name]).hexdigest()}
                for name, url in endpoints.items()}
    (DATA / 'evidence_sources.json').write_text(json.dumps(manifest, indent=2))
    return records, documents


def build_inactivity(records, documents):
    from collections import defaultdict
    document_sets = defaultdict(set)
    canonical = {}
    for record in records:
        s = record.get('canonical_smiles') or ''
        if s not in canonical:
            mol = Chem.MolFromSmiles(s)
            canonical[s] = Chem.MolToSmiles(mol) if mol and mol.GetNumAtoms() else None
        if canonical[s]:
            document_sets[canonical[s]].add(record['document_chembl_id'])
    rows = []
    for record in records:
        field = inactive_field(record)
        if field is None:
            continue
        s = canonical[record.get('canonical_smiles') or '']
        if s is None:
            raise ValueError('Invalid SMILES in inactivity evidence')
        doc = record['document_chembl_id']
        doi = documents[doc].get('doi') or ''
        if not doi:
            raise ValueError('Missing DOI for inactivity evidence: ' + doc)
        rows.append(dict(smiles=s, outcome='Not active', evidence_field=field,
                         reported_text=record[field], endpoint=record['standard_type'],
                         doi=doi, document_id=doc, molecule_id=record['molecule_chembl_id'],
                         activity_id=record['activity_id'], assay_id=record['assay_chembl_id'],
                         assay_description=record['assay_description'],
                         vacv_document_count=len(document_sets[s]),
                         single_source=len(document_sets[s]) == 1,
                         source_url=API + f"activity/{record['activity_id']}.json"))
    return rows


def collect_inactivity(refresh=False, offline=False):
    records, documents = load_evidence_sources(refresh, offline)
    rows = build_inactivity(records, documents)
    with (DATA / 'external_inactivity.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    report = dict(scanned_activities=len(records), records=len(rows),
                  unique_smiles=len({r['smiles'] for r in rows}),
                  documents=len(documents),
                  unique_single_source_smiles=len({r['smiles'] for r in rows if r['single_source']}),
                  limitations=[
                      'Only explicit Not active text in antiviral endpoints is accepted.',
                      'Cytotoxicity, CC50 and MCC50 outcomes are not antiviral inactivity.',
                      'Absence of a measurement never creates an inactivity annotation.',
                      'An external outcome is not automatically the competition SI=10000 target.',
                      'Different strains, cells and assay conditions may disagree.',
                      'Multiple VACV publications for a molecule prevent the single-source flag.',
                  ])
    (DATA / 'inactivity_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--refresh', action='store_true')
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()
    if args.refresh and args.offline:
        p.error('--refresh and --offline cannot be combined')
    rows = build(load_sources(args.refresh, args.offline))
    clean = [r for r in rows if r['usable_for_training']]
    with (DATA / 'external.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    report = dict(records=len(rows), exact_usable_records=len(clean),
                  unique_usable_smiles=len({r['smiles'] for r in clean}),
                  documents=len({r['doi'] for r in rows}), assay_mapping=STUDIES,
                  rejected_reasons=dict(Counter(flag for r in rows for flag in r['quality_flags'].split(';') if flag)),
                  per_document=dict(Counter(r['doi'] for r in rows)),
                  limitations=['Nine curated publications, not an exhaustive literature collection.',
                               'Published SI retained; bounds are excluded from exact-label training.',
                               'TC50 maps to CC50 only in the curated cytotoxicity assay; original types and text are retained.',
                               'Incomplete IC50 observations are retained for audit, never assigned SI=10000.',
                               '2023 paper reports 72 h CC50 / 96 h IC50; that context is retained.',
                               '2013 paper reports overnight IC50 / 20 h CC50; that context is retained.',
                               'Missing IC50 or SI is not automatically an inactivity label.',
                               'Distinct cell lines remain separate observations.'])
    (DATA / 'external_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    collect_inactivity(args.refresh, args.offline)


if __name__ == '__main__':
    main()
