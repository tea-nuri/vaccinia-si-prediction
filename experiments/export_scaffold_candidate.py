"""Export a user-requested experiment without promoting it or submitting it."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import ALLOWED
from experiments.scaffold_generalization import (
    CANDIDATES, agreement_lookup, fit_current_baseline, neighbors,
    predict_candidates, unseen_scaffold_mask)


def export(candidate, output):
    if candidate not in CANDIDATES:
        raise ValueError('Unknown experiment')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((ROOT/'results/scaffold_generalization_experiment.json').read_text())
    for name, digest in report['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Source changed since validation: {name}')
    tr = pd.read_csv(ROOT/'data/train.csv', usecols=ALLOWED+['SI VACV'])
    te = pd.read_csv(ROOT/'data/test.csv', usecols=ALLOWED+['index'])
    y = tr['SI VACV'].to_numpy(float)
    sub = pd.read_csv(ROOT/'results/candidate_bounded_features.csv')
    sample = pd.read_csv(ROOT/'data/sample_submit.csv')
    np.testing.assert_array_equal(sub['index'], te['index'])
    np.testing.assert_array_equal(sub['index'], sample['index'])
    if list(sub.columns) != list(sample.columns):
        raise ValueError('Submission schema mismatch')
    fixed = agreement_lookup(te).prediction.to_numpy(float)
    baseline = fit_current_baseline(tr, y, te, fixed)
    np.testing.assert_allclose(baseline, sub['SI VACV'], rtol=0, atol=1e-6)
    nearest = neighbors(tr[ALLOWED], y, te[ALLOWED])['text',7]
    prediction = predict_candidates(tr, y, te, baseline, nearest, fixed)[candidate]
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Invalid predictions')
    mask = unseen_scaffold_mask(tr, te)
    np.testing.assert_allclose(prediction[~mask], baseline[~mask], rtol=0, atol=1e-6)
    np.testing.assert_allclose(prediction[np.isfinite(fixed)], fixed[np.isfinite(fixed)], rtol=0, atol=1e-6)
    sub['SI VACV'] = prediction
    path = output/'submission_scaffold_experimental.csv'
    sub.to_csv(path, index=False)
    meta = {
        'status': 'user_requested_experimental_export_not_submitted',
        'candidate': candidate, 'promoted': False,
        'passed_selection': candidate in report['passing'],
        'rejection_reasons': report['rejection_reasons'].get(candidate, []),
        'rows': len(sub), 'unseen_scaffold_rows': int(mask.sum()),
        'changed_rows': int((np.abs(prediction-baseline)>1e-6).sum()),
        'baseline_full_refit_max_abs_difference': float(np.max(np.abs(baseline-pd.read_csv(ROOT/'results/candidate_bounded_features.csv')['SI VACV']))),
        'local_scores': {key: value[candidate] for key,value in report['scores'].items()},
        'source_sha256': report['source_sha256'],
        'submission_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'limitations': ['Rejected by robustness criteria; exported at user request for manual inspection or submission.',
                       'Scaffold and molecular CV are exploratory local scores, not Kaggle Public Score.']}
    (output/'scaffold_export.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({key:meta[key] for key in ['candidate','rows','changed_rows','submission_sha256','passed_selection']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=CANDIDATES, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'run_scaffold_export')
    args = parser.parse_args()
    export(args.candidate, args.output)
