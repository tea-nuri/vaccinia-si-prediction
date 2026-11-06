"""Traceable literature retrieval experiment layered on the selected ML model.

Scores measure retrieval-assisted prediction for already published molecules,
not generalization to compounds without published measurements.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import zipfile
from functools import lru_cache
import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SPLITS = ['42', '2026', '117', '2027', '7', 'scaffold']


def additional_viruses(text):
    """Species mentioned in context, excluding the common VACV endpoint."""
    aliases = {
        'cowpox': r'\bcowpox\b|\bcpxv\b',
        'ectromelia': r'\bectromelia\b|\bectv\b',
        'variola': r'\bvariola\b|\bvarv\b',
    }
    return frozenset(name for name, pattern in aliases.items()
                     if re.search(pattern, str(text), flags=re.I))


@lru_cache(maxsize=1)
def document_virus_context():
    """Read source coverage from the verified snapshot, never competition labels.

    A missing species means insufficient provenance for this conservative rule,
    not proof that a measurement or an entire publication is incorrect.
    """
    manifest = json.loads((DATA/'sources.json').read_text())
    coverage = {}
    with zipfile.ZipFile(DATA/'chembl_snapshot.zip') as archive:
        payloads = {}
        for name, entry in manifest.items():
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('Snapshot integrity check failed: '+name)
            payloads[name] = json.loads(raw)
    for name, paper in payloads.items():
        if not name.startswith('paper_'):
            continue
        document = payloads['document_'+name[len('paper_'):]]
        descriptions = ' '.join(str(r.get('target_organism') or '')+' '+
                                str(r.get('assay_description') or '')
                                for r in paper['activities'])
        coverage[document['doi'].casefold()] = additional_viruses(descriptions)
    return coverage


def source_context_rejections(target, audit, coverage=None):
    """Opt-in abstention on numerical retrieval with unsupported source context.

    Extra virus names are used only as a provenance cue. They do not change the
    VACV endpoint, become activity labels, or enter an SI ratio. Primary-table
    mappings are outside this rule. The selected V7 lookup is unchanged.
    """
    if len(target) != len(audit):
        raise ValueError('Target/audit row count mismatch')
    coverage = document_virus_context() if coverage is None else coverage
    rows = []
    for strain, record in zip(target['strain'], audit.itertuples(index=False)):
        required = additional_viruses(strain)
        sources = str(record.source).casefold().split(';')
        supported = any(required <= coverage.get(doi, frozenset()) for doi in sources)
        reject = (record.match_method == 'matched_ChEMBL_assay_pair_exact_stereo'
                  and bool(required) and not supported)
        rows.append(dict(reject_numerical_lookup=reject,
                         required_virus_context=';'.join(sorted(required)),
                         reason='source_context_not_verified' if reject else ''))
    return pd.DataFrame(rows)


def structure_key(smiles, stereo=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES')
    return Chem.MolToSmiles(mol, isomericSmiles=stereo)


def cell_key(value):
    return str(value).lower().replace('cells', '').replace(' ', '').replace('-', '')


def strain_compatible(source, target):
    def known(value):
        s = str(value).lower()
        for word in ['copenhagen', 'lister', 'tiantan', 'livp', 'ihd', 'lederle']:
            if word in s:
                return word
        if 'western reserve' in s or re.search(r'\bwr\b', s):
            return 'wr'
        return None
    a, b = known(source), known(target)
    return a is None or b is None or a == b


def lookup_predictions(target, numerical=None, primary=None, include_2019=False):
    """Only SMILES, cell and strain are inspected; labels never enter lookup."""
    numerical = pd.read_csv(DATA/'external.csv') if numerical is None else numerical
    primary = pd.read_csv(DATA/'primary_literature.csv') if primary is None else primary
    lookup = defaultdict(list)
    for _, row in numerical.iterrows():
        if row.usable_for_training == True:
            lookup[(structure_key(row.smiles), cell_key(row.cell))].append(row)
    if not include_2019:
        primary = primary[primary.doi.ne('10.1007/s11172-019-2467-6')]
    exact, mixture = {}, {}
    for _, row in primary.iterrows():
        if row.retrieval_status != 'eligible' or row.outcome == 'Active':
            continue
        mapping = exact if row.match_scope == 'exact_stereo' else mixture
        key = structure_key(row.smiles, stereo=mapping is exact)
        if key in mapping:
            raise ValueError('Ambiguous primary structure')
        mapping[key] = row
    audit = []
    for _, row in target[['SMILES', 'Culture', 'strain']].iterrows():
        cell = cell_key(row.Culture)
        candidates = [r for r in lookup.get((structure_key(row.SMILES), cell), [])
                      if strain_compatible(r.strain, row.strain)]
        value, source, method = np.nan, '', ''
        if candidates:
            values = [r.CC50/r.IC50 for r in candidates]
            # Do not average incompatible assays or invent a midpoint of bounds.
            if max(values)-min(values) <= max(1., .05*np.mean(values)):
                value = float(np.mean(values))
                source = ';'.join(sorted({r.doi for r in candidates}))
                method = 'matched_ChEMBL_assay_pair_exact_stereo'
        found = exact.get(structure_key(row.SMILES))
        if found is None:
            found = mixture.get(structure_key(row.SMILES, stereo=False))
        # All primary records in this release are Vero/Copenhagen experiments.
        if found is not None and cell == 'vero' and 'copenhagen' in str(row.strain).lower():
            candidate = (10000. if found.outcome == 'Not active'
                         else found.SI_from_rounded_means)
            if pd.notna(candidate):
                value = float(candidate)
                source = found.doi+'#'+found.compound
                method = found.match_scope
        audit.append(dict(prediction=value, source=source, match_method=method))
    return pd.DataFrame(audit)



def qualitative_active_flags(target):
    """Exact matches to independently reported active 2019 structures only.

    No quantitative SI is invented from an activity label. The caller estimates
    SI using its active-only regressor fitted on the current training fold.
    """
    p = pd.read_csv(DATA/'primary_literature.csv')
    p = p[p.doi.eq('10.1007/s11172-019-2467-6') & p.outcome.eq('Active')]
    keys = {structure_key(s) for s in p.smiles}
    return np.array([structure_key(r.SMILES) in keys and cell_key(r.Culture)=='vero'
                     and 'copenhagen' in str(r.strain).lower()
                     for _,r in target[['SMILES','Culture','strain']].iterrows()],dtype=bool)


def apply_literature(target, baseline, include_2019=False):
    audit = lookup_predictions(target, include_2019=include_2019)
    result = np.asarray(baseline, dtype=float).copy()
    matched = audit.prediction.notna().to_numpy()
    result[matched] = audit.loc[matched, 'prediction']
    audit['baseline_prediction'] = baseline
    audit['final_prediction'] = result
    return result, audit


def run_experiment(output):
    from train import ALLOWED, fit_predict_v6
    from sklearn.model_selection import GroupKFold
    from rdkit.Chem.Scaffolds import MurckoScaffold
    output = Path(output)
    if output.resolve() == (ROOT/'results').resolve():
        raise ValueError('Use a candidate directory; selected results require a confirmed Kaggle score')
    output.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(DATA/'train.csv', usecols=ALLOWED+['SI VACV'])
    target = pd.read_csv(DATA/'test.csv', usecols=['index']+ALLOWED)
    y = training['SI VACV'].to_numpy(float)
    canonical = training.SMILES.map(structure_key).to_numpy()
    scaffolds = training.SMILES.map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
    baseline = pd.DataFrame({'target':y, 'canonical_smiles':canonical})
    # Rebuild all six baselines: this experiment stays reproducible even after
    # results/ is replaced by a later selected submission.
    for split in SPLITS:
        seed = int(split) if split != 'scaffold' else None
        groups = canonical if seed is not None else scaffolds
        p = np.full(len(y),np.nan)
        for ix,va in GroupKFold(5,shuffle=seed is not None,random_state=seed).split(training,groups=groups):
            assert not set(canonical[ix]) & set(canonical[va])
            p[va], _ = fit_predict_v6(training.iloc[ix],y[ix],training.iloc[va])
        baseline[split] = p
        print('Rebuilt V6 baseline split '+split,flush=True)
    fresh, _ = fit_predict_v6(training, y, target)
    pred, test_audit = apply_literature(target, fresh)
    train_audit = lookup_predictions(training)
    matched = train_audit.prediction.notna().to_numpy()
    validation = baseline[['target', 'canonical_smiles']].copy()
    scores, baseline_scores, unseen_scores = {}, {}, {}
    for split in SPLITS:
        previous = baseline[split].to_numpy(copy=True)
        corrected = previous.copy()
        corrected[matched] = train_audit.loc[matched, 'prediction']
        validation[split] = corrected
        scores[split] = float(np.mean((corrected-y)**2))
        baseline_scores[split] = float(np.mean((previous-y)**2))
        unseen_scores[split] = float(np.mean((previous[~matched]-y[~matched])**2))
    for group in [scores, baseline_scores, unseen_scores]:
        group['mean_group_mse'] = float(np.mean([group[s] for s in SPLITS[:-1]]))
    train_audit['target'] = y
    train_audit['absolute_error'] = abs(train_audit.prediction-y)
    report = dict(
        model='V7: V6 plus curated primary literature and matched external assay pairs',
        status='local_reproduction_no_automatic_submission', features_allowed=ALLOWED,
        scores=scores, baseline_scores=baseline_scores,
        external_unmatched_rows_scores=unseen_scores,
        covered_train_rows=int(matched.sum()),
        covered_test_rows=int(test_audit.prediction.notna().sum()),
        covered_train_mse=float(np.mean((train_audit.loc[matched,'prediction'].to_numpy()-y[matched])**2)),
        train_rows=len(training), test_rows=len(target),
        limitations=[
            'The CV gain includes retrieval of published measurements for validation molecules. It is NOT a compound-disjoint external-data generalization estimate.',
            'The baseline model is unchanged. On molecules without a lookup the validation predictions are identical to V6.',
            'Sources and curation policies were chosen after inspecting training errors; these local scores are exploratory, not an independent holdout.',
            'The large training outlier dominates the improvement; public/private improvement is not implied by this CV reduction.',
            'Rounded external concentration means are used for retrieval. Published SI is preserved separately and can differ.',
            'A published stereoisomer mixture is only matched by connectivity for an explicitly allowlisted source; fenchyl series 9/10 is excluded pending review.',
            'Matching molecular structure and coarse assay context cannot resolve all assay differences or competition annotation errors.'
        ],
        source_sha256={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['train.csv','test.csv']},
        artifact_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [DATA/'external.csv',DATA/'primary_literature.csv',ROOT/'data/primary_structure_evidence.json']})
    assert np.isfinite(pred).all() and (pred >= 0).all()
    pd.DataFrame({'index':target['index'],'SI VACV':pred}).to_csv(output/'submission.csv',index=False)
    validation.to_csv(output/'validation.csv',index=False)
    test_audit.insert(0,'index',target['index'])
    test_audit.to_csv(output/'test_lookup_audit.csv',index=False)
    train_audit.to_csv(output/'train_lookup_audit.csv',index=False)
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
