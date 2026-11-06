"""Audit scaffold stress errors and how its novelty differs from the test set."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import ALLOWED, evidence_flags, apply_inactivity_evidence
from literature import cell_key, structure_key
from experiments.measurement_agreement import agreement_lookup
from experiments.bounded_features import calibrated_transfer


def prior_probe(tr, y, fps, scaffolds):
    """Two frozen low-similarity fallbacks, with the same evidence corrections."""
    canonical = tr.SMILES.map(structure_key).to_numpy()
    fixed = agreement_lookup(tr).prediction.to_numpy(float)
    report = {'threshold': .3, 'weights': [.25, .5],
              'prior': 'Mean capped-at-10000 training target after averaging by canonical molecule',
              'scores': {}, 'known_culture_scores': {}}
    for split in ['scaffold','42','2026','117','2027','7','23','101','509']:
        cache = pd.read_csv(ROOT/f'run_bounded_features/validation_{split}.csv')
        np.testing.assert_array_equal(cache.target, y)
        baseline = cache['1.0'].to_numpy()
        predictions = {str(w): baseline.copy() for w in [.25,.5]}
        known = np.zeros(len(tr), dtype=bool)
        seed = None if split == 'scaffold' else int(split)
        for ix, va in GroupKFold(5, shuffle=seed is not None, random_state=seed).split(
                tr, groups=scaffolds if seed is None else canonical):
            known[va] = tr.iloc[va].Culture.map(cell_key).isin(tr.iloc[ix].Culture.map(cell_key))
            prior = pd.DataFrame({'key': canonical[ix], 'y': np.minimum(y[ix],10000)}).groupby('key').y.mean().mean()
            pool = [fps[i] for i in ix]
            remote = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i],pool)) < .3 for i in va])
            for weight in [.25,.5]:
                p = baseline[va].copy()
                p[remote] = (1-weight)*p[remote] + weight*prior
                p, _ = apply_inactivity_evidence(tr.iloc[ix],y[ix],tr.iloc[va],p)
                measured = np.isfinite(fixed[va])
                p[measured] = fixed[va][measured]
                p, _ = calibrated_transfer(tr.iloc[ix],y[ix],tr.iloc[va],p)
                predictions[str(weight)][va] = p
        predictions['baseline'] = baseline
        report['scores'][split] = {key: float(np.mean((p-y)**2)) for key,p in predictions.items()}
        report['known_culture_scores'][split] = {key: float(np.mean((p[known]-y[known])**2)) for key,p in predictions.items()}
    report['decision'] = 'Rejected: both weights improve whole original scaffold but worsen known-Culture scaffold and all eight molecular protocols.'
    return report


def main():
    output = ROOT/'run_scaffold_audit'
    output.mkdir(exist_ok=True)
    tr = pd.read_csv(ROOT/'data/train.csv', usecols=ALLOWED+['SI VACV'])
    te = pd.read_csv(ROOT/'data/test.csv', usecols=ALLOWED+['index'])
    validation = pd.read_csv(ROOT/'run_bounded_features/validation_scaffold.csv')
    y = tr['SI VACV'].to_numpy(float)
    np.testing.assert_array_equal(validation.target, y)
    pred = validation['1.0'].to_numpy()
    expected = json.loads((ROOT/'results/bounded_features_experiment.json').read_text())
    np.testing.assert_allclose(np.mean((pred-y)**2), expected['scores']['scaffold']['1.0'], atol=1e-6, rtol=0)
    scaffolds = lambda f: f.SMILES.map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
    groups, test_groups = scaffolds(tr), scaffolds(te)
    cells, test_cells = tr.Culture.map(cell_key).to_numpy(), te.Culture.map(cell_key).to_numpy()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr.SMILES]
    test_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in te.SMILES]
    similarity = lambda queries, pool: np.array([max(DataStructs.BulkTanimotoSimilarity(q, pool)) for q in queries])
    known = np.zeros(len(tr), dtype=bool)
    nearest = np.zeros(len(tr))
    fold_ids = np.zeros(len(tr), dtype=int)
    folds = []
    for fold, (ix, va) in enumerate(GroupKFold(5).split(tr, groups=groups)):
        assert not set(groups[ix]) & set(groups[va])
        known[va] = np.isin(cells[va], cells[ix])
        nearest[va] = similarity([fps[i] for i in va], [fps[i] for i in ix])
        fold_ids[va] = fold
        folds.append({'fold': fold, 'rows': len(va), 'mse': float(np.mean((pred[va]-y[va])**2)),
                      'unseen_culture_rows': int((~known[va]).sum())})
    errors = (pred-y)**2
    audit = pd.DataFrame({'row': np.arange(len(tr)), 'target': y, 'prediction': pred,
                          'squared_error': errors, 'scaffold': groups, 'fold': fold_ids,
                          'known_culture': known, 'nearest_similarity': nearest})
    summary = audit.groupby('scaffold').agg(rows=('row','size'), sse=('squared_error','sum'))
    dominant = summary.sse.idxmax()
    dominant_mask = groups == dominant
    outside = ~dominant_mask
    test_new = ~np.isin(test_groups, groups)
    test_nearest = similarity(test_fps, fps)
    report = {'status': 'diagnostic_not_test_labels',
        'overall_mse': float(errors.mean()), 'folds': folds,
        'known_culture_rows': int(known.sum()), 'known_culture_mse': float(errors[known].mean()),
        'dominant_scaffold': {'key': dominant, 'rows': int(dominant_mask.sum()),
            'sentinel_rows': int((y[dominant_mask] == 10000).sum()),
            'share_of_sse': float(errors[dominant_mask].sum()/errors.sum()),
            'contribution_to_overall_mse': float(errors[dominant_mask].sum()/len(y)),
            'mean_prediction': float(pred[dominant_mask].mean()), 'mean_target': float(y[dominant_mask].mean()),
            'existing_inactivity_matches': int(evidence_flags(tr.loc[dominant_mask,'SMILES']).sum()),
            'test_rows': int((test_groups == dominant).sum())},
        'outside_dominant_mse': float(errors[outside].mean()),
        'sentinel_share_of_sse': float(errors[y == 10000].sum()/errors.sum()),
        'outside_dominant_sentinel_mse': float(errors[outside & (y == 10000)].mean()),
        'outside_dominant_numerical_mse': float(errors[outside & (y != 10000)].mean()),
        'test_new_scaffold_rows': int(test_new.sum()),
        'test_new_culture_rows': int((~np.isin(test_cells, cells)).sum()),
        'scaffold_cv_nearest_similarity_median': float(np.median(nearest)),
        'test_new_scaffold_nearest_similarity_median': float(np.median(test_nearest[test_new])),
        'test_known_scaffold_nearest_similarity_median': float(np.median(test_nearest[~test_new])),
        'test_similarity_below_point3': int((test_nearest < .3).sum()),
        'limitations': ['This is exploratory reuse of the training dataset and external retrieval.',
            'No test targets or forbidden competition concentrations are read.',
            'Known Culture is diagnostic only: no row or culture-specific prediction exception is added.',
            'Both full scaffold and known-Culture scaffold scores must be reported; neither estimates Public Score.'],
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['data/train.csv','data/test.csv','run_bounded_features/validation_scaffold.csv','results/bounded_features_experiment.json']}}
    report['uncertainty_prior_probe'] = prior_probe(tr, y, fps, groups)
    audit.to_csv(output/'rows.csv', index=False)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['overall_mse','known_culture_mse','outside_dominant_mse','test_new_scaffold_rows','test_new_culture_rows']}, indent=2))


if __name__ == '__main__':
    main()
