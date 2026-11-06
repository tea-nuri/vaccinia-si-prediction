"""85% SMILES Ridge + 15% serialization-count ExtraTrees, calibrated on CV."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from train_text import make_model

ROOT = Path(__file__).resolve().parent
CHARACTERS = list('cCnNoOpPsSFIBr[]=#@/\\123456789%()+-.')


def serialization_features(smiles):
    return np.asarray([[len(s)] + [s.count(c) for c in CHARACTERS]
                       + [s.find('c'), s.find('@'), s.count('C1'), s.count('c1')]
                       for s in smiles], dtype=float)


def fit_predict(x, y, target):
    text_model = make_model().fit(x, np.minimum(y, 10000))
    tree_model = ExtraTreesRegressor(n_estimators=500, max_features=1.0,
                                    min_samples_leaf=1, n_jobs=4, random_state=42)
    tree_model.fit(serialization_features(x), np.minimum(y, 10000))
    text_pred = np.clip(text_model.predict(target), 0, 10000)
    tree_pred = np.clip(tree_model.predict(serialization_features(target)), 0, 10000)
    blended = 0.85 * text_pred + 0.15 * tree_pred
    prediction = np.clip((blended - 5000) * 1.5 + 5000, 0, 10000)
    return prediction, {'text': text_model, 'trees': tree_model,
                        'tree_weight': 0.15, 'slope': 1.5, 'center': 5000,
                        'characters': CHARACTERS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'run_ensemble')
    parser.add_argument('--skip-cv', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(ROOT/'train.csv', usecols=['SMILES', 'SI VACV'])
    te = pd.read_csv(ROOT/'test.csv', usecols=['index', 'SMILES'])
    x = tr.SMILES.to_numpy(); y = tr['SI VACV'].to_numpy()
    mols = [Chem.MolFromSmiles(s) for s in x]
    assert all(m is not None for m in mols)
    canonical = np.array([Chem.MolToSmiles(m) for m in mols])
    scaffold = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m) or 'ACYCLIC' for m in mols])
    scores = {}; oof = pd.DataFrame({'target': y, 'canonical_smiles': canonical})
    if not args.skip_cv:
        for seed in [42, 2026, 117, 2027, 7, None]:
            p = np.zeros(len(y)); name = str(seed) if seed is not None else 'scaffold'
            groups = canonical if seed is not None else scaffold
            for ix, va in GroupKFold(5, shuffle=seed is not None, random_state=seed).split(x, y, groups):
                assert not set(canonical[ix]) & set(canonical[va])
                p[va], _ = fit_predict(x[ix], y[ix], x[va])
            scores[name] = float(np.mean((y-p)**2)); oof[name] = p
            print(name, scores[name], flush=True)
        scores['mean_group_mse'] = float(np.mean([scores[str(s)] for s in [42, 2026, 117, 2027, 7]]))
        oof.to_csv(args.output/'oof.csv', index=False)
    pred, model = fit_predict(x, y, te.SMILES.to_numpy())
    pd.DataFrame({'SMILES': x, 'original_SI': y,
                  'training_target': np.minimum(y, 10000),
                  'source': 'competition_train'}).to_csv(
                      args.output/'training_used_text.csv', index=False)
    sample = pd.read_csv(ROOT/'sample_submit.csv')
    assert sample.columns.tolist() == ['index', 'SI VACV']
    assert sample['index'].equals(te['index']) and np.isfinite(pred).all()
    sample['SI VACV'] = pred
    sample.to_csv(args.output/'submission_v4.csv', index=False)
    joblib.dump(model, args.output/'model.joblib')
    report = {'features_allowed': ['SMILES'], 'tree_weight': .15,
              'calibration_slope': 1.5, 'calibration_center': 5000,
              'target_cap': 10000, 'prediction_bounds': [0, 10000],
              'scores': scores, 'test_rows': len(te),
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['train.csv', 'test.csv']},
              'limitations': 'Model/weight/calibration chosen using repeated CV on the same small training set. Scores are optimistic. Original SMILES serialization affects predictions. MSE uses original uncapped targets.'}
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
