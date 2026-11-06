"""Reproduce selected V5 using SMILES and allowed assay context."""
from pathlib import Path
import argparse
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
ALLOWED = ['SMILES', 'strain', 'Culture']
CHARACTERS = list('cCnNoOpPsSFIBr[]=#@/\\123456789%()+-.')


def make_model():
    return make_pipeline(
        TfidfVectorizer(analyzer='char', ngram_range=(2, 6),
                        lowercase=False, sublinear_tf=True),
        Ridge(alpha=0.03))


def serialization_features(smiles):
    return np.asarray([[len(s)] + [s.count(c) for c in CHARACTERS]
                       + [s.find('c'), s.find('@'), s.count('C1'), s.count('c1')]
                       for s in smiles], dtype=float)


def _predict_smiles(train_smiles, targets, predict_smiles):
    text_model = make_model().fit(train_smiles, np.minimum(targets, 10000))
    tree_model = ExtraTreesRegressor(n_estimators=500, max_features=1.0,
                                    min_samples_leaf=1, n_jobs=4, random_state=42)
    tree_model.fit(serialization_features(train_smiles), np.minimum(targets, 10000))
    text_pred = np.clip(text_model.predict(predict_smiles), 0, 10000)
    tree_pred = np.clip(tree_model.predict(serialization_features(predict_smiles)), 0, 10000)
    blended = .85*text_pred+.15*tree_pred
    prediction = np.clip((blended-5000)*1.5+5000, 0, 10000)
    return prediction, {'text':text_model, 'trees':tree_model,
                        'tree_weight':.15, 'slope':1.5, 'center':5000,
                        'characters':CHARACTERS}


def fit_predict(training, targets, target):
    # The allowlist excludes concentrations, missingness and identifiers even
    # when a caller passes a full competition dataframe.
    training = training[ALLOWED]
    target = target[ALLOWED]
    smiles_pred, smiles_models = _predict_smiles(
        training.SMILES.to_numpy(), targets, target.SMILES.to_numpy())
    xt = pd.DataFrame(serialization_features(training.SMILES)).astype(float)
    xv = pd.DataFrame(serialization_features(target.SMILES)).astype(float)
    categorical = []
    for column in ['strain', 'Culture']:
        categorical.append(xt.shape[1])
        xt[column] = training[column].fillna('UNKNOWN').astype(str).to_numpy()
        xv[column] = target[column].fillna('UNKNOWN').astype(str).to_numpy()
    inactive = CatBoostClassifier(iterations=700, depth=5, learning_rate=.035,
                                 l2_leaf_reg=5, verbose=False, thread_count=4,
                                 allow_writing_files=False, random_seed=42)
    inactive.fit(xt, (targets == 10000).astype(int), cat_features=categorical)
    inactive_part = 10000 * inactive.predict_proba(xv)[:, 1]
    active = targets != 10000
    active_model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                      max_features=.5, random_state=42, n_jobs=4)
    active_model.fit(serialization_features(training.loc[active, 'SMILES']),
                     np.minimum(targets[active], 5000))
    active_pred = np.clip(active_model.predict(serialization_features(target.SMILES)), 0, 5000)
    context_pred = np.clip(inactive_part + (1-inactive_part/10000)*active_pred, 0, 10000)
    return .5*smiles_pred+.5*context_pred, {
        'smiles_models': smiles_models, 'inactive_classifier': inactive,
        'active_regressor': active_model, 'categorical_positions': categorical,
        'context_weight': .5, 'features_allowed': ALLOWED}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'run')
    parser.add_argument('--skip-cv', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Only context explicitly permitted by the organizers joins SMILES.
    train = pd.read_csv(DATA / 'train.csv', usecols=ALLOWED+['SI VACV'])
    test = pd.read_csv(DATA / 'test.csv', usecols=['index']+ALLOWED)
    sample = pd.read_csv(DATA / 'sample_submit.csv')
    x = train.SMILES.to_numpy()
    y = train['SI VACV'].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid training target')
    mols = [Chem.MolFromSmiles(s) for s in x]
    assert all(m is not None for m in mols)
    groups = np.array([Chem.MolToSmiles(m) for m in mols])
    scaffolds = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m)
                          or 'ACYCLIC' for m in mols])
    scores = {}
    oof = pd.DataFrame({'target': y, 'canonical_smiles': groups})
    if not args.skip_cv:
        for seed in [42, 2026, 117, 2027, 7, None]:
            label = str(seed) if seed is not None else 'scaffold'
            splitter = GroupKFold(5, shuffle=seed is not None, random_state=seed)
            p = np.zeros(len(y))
            for ix, va in splitter.split(x, y, groups if seed is not None else scaffolds):
                assert not set(groups[ix]) & set(groups[va])
                # Vocabulary and IDF are fitted strictly inside this training fold.
                p[va], _ = fit_predict(train.iloc[ix], y[ix], train.iloc[va])
            scores[label] = float(np.mean((y-p)**2))
            oof[label] = p
            print(label, scores[label], flush=True)
        scores['mean_group_mse'] = float(np.mean([scores[str(s)] for s in [42, 2026, 117, 2027, 7]]))
        oof.to_csv(args.output / 'validation.csv', index=False)
    pred, model = fit_predict(train, y, test)
    used = train[ALLOWED].copy()
    used['original_SI'] = y
    used['inactive_label'] = (y==10000).astype(int)
    used['v4_training_target'] = np.minimum(y,10000)
    used['active_regression_target'] = np.where(y!=10000, np.minimum(y,5000), np.nan)
    used.to_csv(args.output / 'training_used.csv', index=False)
    assert sample.columns.tolist() == ['index', 'SI VACV']
    assert sample['index'].equals(test['index'])
    assert np.isfinite(pred).all() and len(pred) == len(test)
    sample['SI VACV'] = pred
    sample.to_csv(args.output / 'submission.csv', index=False)
    joblib.dump(model, args.output / 'model.joblib')
    report = {
        'model': 'V5: equal blend of calibrated SMILES model and assay-context inactivity classifier',
        'context_weight': .5,
        'inactive_classifier': {'iterations':700,'depth':5,'learning_rate':.035,'l2_leaf_reg':5,'seed':42},
        'tree_weight': .15, 'calibration_slope': 1.5, 'calibration_center': 5000,
        'ngram_range': [2, 6], 'alpha': 0.03,
        'training_target_cap': 10000, 'prediction_bounds': [0, 10000],
        'features_allowed': ALLOWED, 'train_rows': len(y),
        'test_rows': len(test), 'scores': scores,
        'source_sha256': {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                          for name in ['train.csv', 'test.csv']},
        'limitations': [
            'CV selection uses the same small set of molecules; it is optimistic.',
            'Raw SMILES serialization carries source conventions. Equivalent rewritten SMILES can receive different predictions.',
            'Train targets are capped, but all validation MSE values use the ORIGINAL uncapped targets.',
            'The target cap can underestimate genuinely high SI values.',
            'SI=10000 is an inactivity sentinel per organizer clarification, not measured high selectivity.',
        ],
    }
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
