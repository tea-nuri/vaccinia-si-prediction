"""Audit selected OOF errors without reading prohibited concentration columns."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from train import ALLOWED

ROOT = Path(__file__).resolve().parent


def diagnose(output):
    training = pd.read_csv(ROOT/'data/train.csv', usecols=ALLOWED+['SI VACV'])
    validation = pd.read_csv(ROOT/'results/validation.csv')
    y = training['SI VACV'].to_numpy(float)
    np.testing.assert_array_equal(validation.target.to_numpy(), y)
    splits = [str(s) for s in [42, 2026, 117, 2027, 7]] + ['scaffold']
    report = {
        'baseline_model': json.loads((ROOT/'results/metrics.json').read_text())['model'],
        'metric': 'MSE on original uncapped targets; supplementary slices do not replace it',
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ['data/train.csv', 'results/validation.csv']},
        'train_rows': len(y), 'inactive_rows': int((y == 10000).sum()),
        'targets_above_prediction_cap': int((y > 10000).sum()),
        'largest_target': float(y.max()),
        'minimum_possible_train_mse_with_cap_10000': float(np.mean(np.maximum(y-10000, 0)**2)),
        'splits': {},
        'limitations': [
            'Training OOF error decomposition does not identify errors on the hidden Kaggle test.',
            'Removing the largest target is a diagnostic slice only, not a replacement metric.',
            'The lower bound applies to these training targets, not to the unseen test distribution.',
            'A low leaderboard score alone does not establish how a team generated predictions.',
            'No CC50, IC50, concentration missingness, Title or index is used as a predictor.',
        ],
    }
    rows = training[ALLOWED].copy()
    rows['target'] = y
    for split in splits:
        p = validation[split].to_numpy(float)
        error = (y-p)**2
        assert np.isfinite(error).all()
        rows[f'prediction_{split}'] = p
        rows[f'squared_error_{split}'] = error
        report['splits'][split] = {
            'mse': float(error.mean()),
            'largest_target_error_share': float(error[np.argmax(y)]/error.sum()),
            'top20_error_share': float(np.sort(error)[-20:].sum()/error.sum()),
            'mse_on_targets_at_most_10000': float(error[y <= 10000].mean()),
            'inactive_mse': float(error[y == 10000].mean()),
            'other_target_mse': float(error[y != 10000].mean()),
            'threshold_5000_inactivity_disagreements': int(((p >= 5000) != (y == 10000)).sum()),
        }
    error = rows['squared_error_42']
    report['culture_error_seed42'] = (
        training.assign(squared_error=error).groupby('Culture', dropna=False)
        .agg(rows=('squared_error', 'size'), mse=('squared_error', 'mean'))
        .sort_values('mse', ascending=False).reset_index().to_dict('records'))
    report['largest_errors_seed42'] = [
        {'source_row': int(i), 'target': float(y[i]),
         'prediction': float(validation.loc[i, '42']), 'squared_error': float(error[i]),
         'culture': str(training.loc[i, 'Culture'])}
        for i in np.argsort(error.to_numpy())[-20:][::-1]]
    output.mkdir(parents=True, exist_ok=True)
    (output/'error_analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    rows.to_csv(output/'error_rows.csv', index=False)
    print(json.dumps({k: report[k] for k in [
        'train_rows', 'inactive_rows', 'largest_target',
        'minimum_possible_train_mse_with_cap_10000', 'splits']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'run_diagnostics')
    diagnose(parser.parse_args().output)
