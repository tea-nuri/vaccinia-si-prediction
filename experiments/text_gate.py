"""Compare probability models for the 10000 sentinel using fold-local features.

Uses the frozen V9 predictions and unchanged primary measurements. No uploads.
Discovery: 42/2026/scaffold; confirmation: 117/2027/7 (same molecules).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import ALLOWED, serialization_features, apply_inactivity_evidence, fit_predict_v6
from literature import lookup_predictions, structure_key, SPLITS
from experiments.literature_candidates import candidate_lookup
from experiments.camphor2022 import audit_matches, main as reproduce_v9


def v9_lookup(frame):
    audit = lookup_predictions(frame)
    for extra in [candidate_lookup(frame), audit_matches(frame)]:
        mask = extra.eligible_new_match
        audit.loc[mask, ['prediction', 'source', 'match_method']] = extra.loc[
            mask, ['prediction', 'source', 'match_method']]
    return audit


def probabilities(training, y, target, only=None):
    tf = TfidfVectorizer(analyzer='char', ngram_range=(2, 6),
                         lowercase=False, sublinear_tf=True)
    a = tf.fit_transform(training.SMILES)
    b = tf.transform(target.SMILES)
    enc = OneHotEncoder(handle_unknown='ignore')
    ca = enc.fit_transform(training[['strain', 'Culture']].fillna('UNKNOWN'))
    cb = enc.transform(target[['strain', 'Culture']].fillna('UNKNOWN'))
    result = {}
    for context_weight in [0., 1.]:
        aa, bb = hstack([a, context_weight*ca]), hstack([b, context_weight*cb])
        for c in [10., 100.]:
            key = f'logistic_C{c:g}_context{context_weight:g}'
            if only is not None and key != only:
                continue
            model = LogisticRegression(C=c, max_iter=2000, random_state=42)
            model.fit(aa, (y == 10000).astype(int))
            result[key] = model.predict_proba(bb)[:, 1]
        if context_weight == 1:
            for kernel in ['linear', 'rbf']:
                key = f'svc_{kernel}_context1'
                if only is not None and key != only:
                    continue
                model = SVC(C=10., kernel=kernel, probability=True, random_state=42)
                model.fit(aa, (y == 10000).astype(int))
                result[key] = model.predict_proba(bb)[:, 1]
    return result


def candidate_predict(training, y, target, baseline, selected, lookup_override=None):
    name, power, weight = selected.rsplit('_power', 1)[0], float(selected.split('_power')[1].split('_')[0]), float(selected.rsplit('_weight', 1)[1])
    prob = probabilities(training, y, target, only=name)[name]
    bounded = np.clip(prob, 1e-12, 1-1e-12)
    prob = 1/(1+np.exp(-power*(np.log(bounded)-np.log1p(-bounded))))
    active = y != 10000
    model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                               max_features=.5, random_state=42, n_jobs=4)
    model.fit(serialization_features(training.loc[active].SMILES), np.minimum(y[active], 5000))
    act = np.clip(model.predict(serialization_features(target.SMILES)), 0, 5000)
    p, _ = apply_inactivity_evidence(training, y, target, prob*10000+(1-prob)*act)
    lookup = v9_lookup(target).prediction.to_numpy() if lookup_override is None else np.asarray(lookup_override, dtype=float)
    matched = np.isfinite(lookup)
    p[matched] = lookup[matched]
    return (1-weight)*baseline+weight*p


def confirm(output):
    output = Path(output)
    report = json.loads((output/'metrics.json').read_text())
    selected = report['selected']
    if selected is None:
        raise ValueError('No discovery candidate')
    tr = pd.read_csv(ROOT/'data/train.csv', usecols=ALLOWED+['SI VACV'])
    te = pd.read_csv(ROOT/'data/test.csv', usecols=ALLOWED+['index'])
    y = tr['SI VACV'].to_numpy(float)
    canon = tr.SMILES.map(structure_key).to_numpy()
    lookup = v9_lookup(tr).prediction.to_numpy()
    extra = {}
    for seed in [23, 101, 509]:
        old, new = np.zeros(len(y)), np.zeros(len(y))
        for fold, (ix, va) in enumerate(GroupKFold(5, shuffle=True, random_state=seed).split(tr, groups=canon)):
            old[va], _ = fit_predict_v6(tr.iloc[ix], y[ix], tr.iloc[va])
            matched = np.isfinite(lookup[va])
            old[va[matched]] = lookup[va[matched]]
            new[va] = candidate_predict(tr.iloc[ix], y[ix], tr.iloc[va], old[va], selected)
            print('confirm', seed, fold, flush=True)
        extra[str(seed)] = {'baseline': float(np.mean((old-y)**2)), 'candidate': float(np.mean((new-y)**2))}
        pd.DataFrame({'target':y, 'baseline':old, 'candidate':new}).to_csv(output/f'confirmation_{seed}.csv', index=False)
        print(extra[str(seed)], flush=True)
    report['additional_confirmation'] = extra
    report['additional_confirmation_pass'] = all(v['candidate'] < v['baseline'] for v in extra.values())
    if report['additional_confirmation_pass']:
        submission = pd.read_csv(ROOT/'results/candidate_camphor2022.csv')
        np.testing.assert_array_equal(submission['index'], te['index'])
        submission['SI VACV'] = candidate_predict(tr, y, te, submission['SI VACV'].to_numpy(), selected)
        assert np.isfinite(submission['SI VACV']).all() and submission['SI VACV'].ge(0).all()
        submission.to_csv(output/'submission.csv', index=False)
        report['candidate_submission_sha256'] = hashlib.sha256((output/'submission.csv').read_bytes()).hexdigest()
    (output/'metrics.json').write_text(json.dumps(report, indent=2))
    print('Additional confirmation:', report['additional_confirmation_pass'], flush=True)


def main(output):
    output = Path(output)
    if output.resolve() == (ROOT/'results').resolve():
        raise ValueError('Use a separate experiment directory')
    output.mkdir(parents=True, exist_ok=True)
    reproduce_v9(output/'baseline_v9')
    tr = pd.read_csv(ROOT/'data/train.csv', usecols=ALLOWED+['SI VACV'])
    te = pd.read_csv(ROOT/'data/test.csv', usecols=ALLOWED+['index'])
    y = tr['SI VACV'].to_numpy(float)
    base = pd.read_csv(output/'baseline_v9/validation.csv')
    lookup = v9_lookup(tr).prediction.to_numpy()
    matched = np.isfinite(lookup)
    canon = tr.SMILES.map(structure_key).to_numpy()
    scaffolds = tr.SMILES.map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
    report = {'status': 'local_only_not_submitted', 'scores': {}, 'selection':
              'Strict improvement on 42/2026/scaffold, then 117/2027/7; lowest mean discovery molecular MSE among passing candidates.',
              'features': 'Fold-local raw SMILES char TFIDF 2-6 and optional one-hot strain/Culture. Active regressor unchanged: ExtraTrees 300 leaf3 max_features .5 cap5000.',
              'limitations': ['Repeated exploratory CV on the same molecules, not independent confirmation.',
                              'SVC internal probability calibration uses training-fold data only.',
                              'No public-score optimization or uploads.'],
              'source_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                  ['data/train.csv', 'data/test.csv', 'results/candidate_camphor2022.csv']}}
    predictions = {}
    for split in ['42', '2026', 'scaffold', '117', '2027', '7']:
        seed = None if split == 'scaffold' else int(split)
        groups = scaffolds if seed is None else canon
        pp = {}
        for fold, (ix, va) in enumerate(GroupKFold(5, shuffle=seed is not None, random_state=seed).split(tr, groups=groups)):
            assert not set(canon[ix]) & set(canon[va])
            active = y[ix] != 10000
            model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                       max_features=.5, random_state=42, n_jobs=4)
            model.fit(serialization_features(tr.iloc[ix[active]].SMILES), np.minimum(y[ix][active], 5000))
            act = np.clip(model.predict(serialization_features(tr.iloc[va].SMILES)), 0, 5000)
            probs = probabilities(tr.iloc[ix], y[ix], tr.iloc[va])
            for name, prob in probs.items():
                for power in [1., 1.5, 2.]:
                    bounded = np.clip(prob, 1e-12, 1-1e-12)
                    calibrated = 1/(1+np.exp(-power*(np.log(bounded)-np.log1p(-bounded))))
                    p = calibrated*10000+(1-calibrated)*act
                    p, _ = apply_inactivity_evidence(tr.iloc[ix], y[ix], tr.iloc[va], p)
                    p[matched[va]] = lookup[va][matched[va]]
                    pp.setdefault(f'{name}_power{power:g}', np.zeros(len(y)))[va] = p
            print(split, 'fold', fold, flush=True)
        scores = {'baseline': float(np.mean((base[split]-y)**2))}
        for name, p in pp.items():
            for weight in [.25, .5, 1.]:
                key = f'{name}_weight{weight:g}'
                mixed = (1-weight)*base[split].to_numpy()+weight*p
                scores[key] = float(np.mean((mixed-y)**2))
                predictions[f'{split}:{key}'] = mixed
        report['scores'][split] = scores
        print(split, sorted(scores.items(), key=lambda x: x[1])[:4], flush=True)
        (output/'metrics.json').write_text(json.dumps(report, indent=2))
    report['discovery_pass'] = [k for k in report['scores']['42'] if k != 'baseline' and all(
        report['scores'][s][k] < report['scores'][s]['baseline'] for s in ['42', '2026', 'scaffold'])]
    report['confirmation_pass'] = [k for k in report['discovery_pass'] if all(
        report['scores'][s][k] < report['scores'][s]['baseline'] for s in ['117', '2027', '7'])]
    report['selected'] = min(report['confirmation_pass'], key=lambda k: np.mean([
        report['scores'][s][k] for s in ['42', '2026']])) if report['confirmation_pass'] else None
    np.savez_compressed(output/'predictions.npz', **predictions)
    (output/'metrics.json').write_text(json.dumps(report, indent=2))
    print('Selected:', report['selected'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'run_text_gate')
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    (confirm if args.confirm else main)(args.output)
