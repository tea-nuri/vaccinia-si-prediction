"""V5: equal blend of v4 and assay-context inactivity classifier.

python train_context_ensemble.py --output run_context_ensemble
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from train_ensemble import fit_predict as predict_v4
from experiments.context_models import predict as predict_context, split_groups, ALLOWED

ROOT = Path(__file__).resolve().parent
CONFIG = {'name': 'cat_False_spike_5', 'family': 'cat',
          'columns': ['strain', 'Culture'], 'mode': 'spike',
          'depth': 5, 'normalize': False}


def fit_predict(training, y, target):
    a, m4 = predict_v4(training.SMILES.to_numpy(), y, target.SMILES.to_numpy())
    b, mc = predict_context(CONFIG, training, y, target)
    return .5*a+.5*b, {'v4': m4, 'context': mc, 'context_config': CONFIG, 'context_weight': .5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'run_context_ensemble')
    parser.add_argument('--skip-cv', action='store_true')
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(ROOT/'train.csv', usecols=ALLOWED+['SI VACV'])
    y = tr.pop('SI VACV').to_numpy()
    te = pd.read_csv(ROOT/'test.csv', usecols=['index']+ALLOWED)
    scores = {}; oof = pd.DataFrame({'target': y, 'canonical_smiles': split_groups(tr)})
    if not args.skip_cv:
        for seed in [42,2026,117,2027,7,None]:
            groups = np.array(split_groups(tr, scaffold=seed is None)); p = np.zeros(len(y))
            for ix, va in GroupKFold(5, shuffle=seed is not None, random_state=seed).split(tr, y, groups):
                assert not set(groups[ix]) & set(groups[va])
                p[va], _ = fit_predict(tr.iloc[ix], y[ix], tr.iloc[va])
            name = str(seed) if seed is not None else 'scaffold'
            scores[name] = float(np.mean((y-p)**2)); oof[name] = p
            print(name, scores[name], flush=True)
        scores['mean_group_mse'] = float(np.mean([scores[str(s)] for s in [42,2026,117,2027,7]]))
        oof.to_csv(out/'oof.csv', index=False)
    p, models = fit_predict(tr, y, te)
    sample = pd.read_csv(ROOT/'sample_submit.csv')
    assert sample.columns.tolist()==['index','SI VACV']
    assert sample['index'].equals(te['index']) and np.isfinite(p).all()
    sample['SI VACV'] = p; sample.to_csv(out/'submission_v5.csv', index=False)
    joblib.dump(models, out/'model.joblib')
    used = tr.copy(); used['original_SI'] = y; used['inactive_label'] = (y==10000).astype(int)
    used['v4_training_target'] = np.minimum(y,10000)
    used['active_regression_target'] = np.where(y!=10000,np.minimum(y,5000),np.nan)
    used.to_csv(out/'training_used_context.csv', index=False)
    report = dict(context_config=CONFIG, context_weight=.5, features_allowed=ALLOWED,
                  scores=scores, test_rows=len(te),
                  source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['train.csv','test.csv']},
                  limitation='Selection reuses the same small dataset across repeated CV. Raw SMILES and assay descriptions retain source conventions. Validation uses original uncapped SI values.')
    (out/'metrics.json').write_text(json.dumps(report, indent=2))


if __name__=='__main__': main()
