"""Allowed assay context + SMILES, prompted by organizers' captain-chat clarification.

python experiments/context_models.py --search
python experiments/context_models.py --config path/to/config.json --output run_context
CC50/IC50 and their missingness are never read.
"""
from pathlib import Path
import sys, json, argparse, hashlib
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from catboost import CatBoostRegressor, CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_ensemble import serialization_features

ALLOWED = ['SMILES', 'strain', 'Culture']


def context(frame, columns, normalize=False):
    result = frame[columns].fillna('UNKNOWN').astype(str).copy()
    if normalize:
        for column in columns:
            result[column] = result[column].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    return result


def predict(config, training, y, target):
    """Fit all representations on training only. Return predictions and fitted objects."""
    cols = config['columns']
    train_context = context(training, cols, config.get('normalize', False))
    target_context = context(target, cols, config.get('normalize', False))
    mode = config.get('mode', 'cap')
    labels = (y == 10000).astype(int) if mode == 'spike' else np.minimum(y, 10000)
    if config['family'] == 'linear':
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 6), lowercase=False, sublinear_tf=True)
        xt = vectorizer.fit_transform(training.SMILES); xv = vectorizer.transform(target.SMILES)
        encoder = OneHotEncoder(handle_unknown='ignore')
        cat_t = encoder.fit_transform(train_context); cat_v = encoder.transform(target_context)
        xt = sparse.hstack([xt, config['weight'] * cat_t]).tocsr()
        xv = sparse.hstack([xv, config['weight'] * cat_v]).tocsr()
        m = LogisticRegression(C=1/config['alpha'], max_iter=2000) if mode == 'spike' else Ridge(alpha=config['alpha'])
        m.fit(xt, labels)
        p = 10000*m.predict_proba(xv)[:, 1] if mode == 'spike' else m.predict(xv)
        objects = {'model': m, 'vectorizer': vectorizer, 'encoder': encoder}
    else:
        xt = pd.DataFrame(serialization_features(training.SMILES)).astype(float)
        xv = pd.DataFrame(serialization_features(target.SMILES)).astype(float)
        categorical = []
        for col in cols:
            categorical.append(xt.shape[1])
            xt[col] = train_context[col].to_numpy(); xv[col] = target_context[col].to_numpy()
        cls = CatBoostClassifier if mode == 'spike' else CatBoostRegressor
        m = cls(iterations=700, depth=config['depth'], learning_rate=.035,
                l2_leaf_reg=5, verbose=False, thread_count=4,
                allow_writing_files=False, random_seed=42)
        m.fit(xt, labels, cat_features=categorical)
        p = 10000*m.predict_proba(xv)[:, 1] if mode == 'spike' else m.predict(xv)
        objects = {'model': m, 'categorical_positions': categorical}
    if mode == 'spike':
        # Residual SI for active compounds. Fit strictly on active training rows.
        active = y != 10000
        low_model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                       max_features=.5, random_state=42, n_jobs=4)
        low_model.fit(serialization_features(training.loc[active, 'SMILES']), np.minimum(y[active], 5000))
        low = np.clip(low_model.predict(serialization_features(target.SMILES)), 0, 5000)
        p = p + (1-p/10000)*low
        objects['active_regressor'] = low_model
    return np.clip(p, 0, 10000), objects


def split_groups(frame, scaffold=False):
    mols = [Chem.MolFromSmiles(s) for s in frame.SMILES]
    assert all(m is not None for m in mols)
    if scaffold:
        return [MurckoScaffold.MurckoScaffoldSmiles(mol=m) or 'ACYCLIC' for m in mols]
    return [Chem.MolToSmiles(m) for m in mols]


def validate(c, frame, y, seed):
    groups = np.array(split_groups(frame, scaffold=seed is None))
    p = np.zeros(len(y))
    for ix, va in GroupKFold(5, shuffle=seed is not None, random_state=seed).split(frame, y, groups):
        assert not set(groups[ix]) & set(groups[va])
        p[va], _ = predict(c, frame.iloc[ix], y[ix], frame.iloc[va])
    return p


def candidates():
    for columns in [['Culture'], ['strain'], ['strain', 'Culture']]:
        for mode in ['cap', 'spike']:
            for weight in [.1, .3, 1., 3.]:
                yield dict(name=f'linear_{"_".join(columns)}_{mode}_{weight}', family='linear',
                           columns=columns, mode=mode, weight=weight, alpha=.03)
    for normalize in [False, True]:
        for mode in ['cap', 'spike']:
            for depth in [3, 5, 7]:
                yield dict(name=f'cat_{normalize}_{mode}_{depth}', family='cat',
                           columns=['strain', 'Culture'], mode=mode, depth=depth, normalize=normalize)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'run_context')
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT/'train.csv', usecols=ALLOWED+['SI VACV'])
    y = frame.pop('SI VACV').to_numpy()
    if args.search:
        results = []
        for config in candidates():
            path = out/f'{config["name"]}_42.joblib'
            if path.exists(): p = joblib.load(path)
            else: p = validate(config, frame, y, 42); joblib.dump(p, path)
            row = dict(config=config, mse42=float(np.mean((y-p)**2)))
            results.append(row); print(config['name'], round(row['mse42']), flush=True)
            (out/'search.json').write_text(json.dumps(sorted(results, key=lambda r:r['mse42']), indent=2))
        for row in sorted(results, key=lambda r:r['mse42'])[:6]:
            for seed in [2026, 117, 2027, 7]:
                path = out/f'{row["config"]["name"]}_{seed}.joblib'
                if path.exists(): p = joblib.load(path)
                else: p = validate(row['config'], frame, y, seed); joblib.dump(p, path)
                row[f'mse{seed}'] = float(np.mean((y-p)**2))
            row['mean'] = float(np.mean([row[f'mse{s}'] for s in [42,2026,117,2027,7]]))
            print('CONFIRMED', row, flush=True)
            (out/'search.json').write_text(json.dumps(sorted(results, key=lambda r:r.get('mean',1e20)), indent=2))
    elif args.config:
        config = json.loads(args.config.read_text())
        scores = {}; oof = pd.DataFrame({'target': y})
        for seed in [42, 2026, 117, 2027, 7, None]:
            p = validate(config, frame, y, seed); label = str(seed) if seed is not None else 'scaffold'
            scores[label] = float(np.mean((y-p)**2)); oof[label] = p
        te = pd.read_csv(ROOT/'test.csv', usecols=['index']+ALLOWED)
        pred, models = predict(config, frame, y, te)
        sample = pd.read_csv(ROOT/'sample_submit.csv')
        assert sample['index'].equals(te['index']) and np.isfinite(pred).all()
        sample['SI VACV'] = pred; sample.to_csv(out/'submission_context.csv', index=False)
        joblib.dump(dict(config=config, models=models), out/'model.joblib')
        oof.to_csv(out/'oof.csv', index=False)
        used = frame.copy(); used['original_SI'] = y; used['inactive_label'] = (y==10000).astype(int)
        used['training_target_capped'] = np.minimum(y,10000); used.to_csv(out/'training_used.csv', index=False)
        report = dict(config=config, scores=scores, features_allowed=ALLOWED,
                      target_meaning='SI=10000 denotes inactive against VACV, per organizer chat supplied by user.',
                      limitation='Repeated grouped CV used for selection; not an unbiased independent estimate.',
                      source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['train.csv','test.csv']})
        (out/'metrics.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    else: parser.error('Choose --search or --config')


if __name__ == '__main__': main()
