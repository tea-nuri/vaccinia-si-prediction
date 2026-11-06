"""Traceable literature retrieval experiment layered on the selected ML model.

Scores measure retrieval-assisted prediction for already published molecules,
not generalization to compounds without published measurements.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SPLITS = ['42', '2026', '117', '2027', '7', 'scaffold']


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


def lookup_predictions(target, numerical=None, primary=None):
    """Only SMILES, cell and strain are inspected; labels never enter lookup."""
    numerical = pd.read_csv(DATA/'external.csv') if numerical is None else numerical
    primary = pd.read_csv(DATA/'primary_literature.csv') if primary is None else primary
    lookup = defaultdict(list)
    for _, row in numerical.iterrows():
        if row.usable_for_training == True:
            lookup[(structure_key(row.smiles), cell_key(row.cell))].append(row)
    exact, mixture = {}, {}
    for _, row in primary.iterrows():
        if row.retrieval_status != 'eligible':
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


def apply_literature(target, baseline):
    audit = lookup_predictions(target)
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
