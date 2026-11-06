"""Bounded classification-head comparison on the selected V6 grouped folds.

Run from repository root: python experiments/activity_heads.py
Only SMILES, strain and Culture enter features. SI enters training labels only.
TF-IDF and category encoders are fitted inside each training fold. V6 external
inactivity corrections are retained for every candidate. This repeated-data
comparison is exploratory, not an independent final holdout estimate.
Artifacts go to ignored run_heads/, never to the selected results automatically.
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train as core
from literature import lookup_predictions
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from scipy.sparse import hstack, csr_matrix
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
OUT = ROOT / 'run_heads'
OUT.mkdir(exist_ok=True)

def run():
    t = time.perf_counter()
    tr = pd.read_csv(ROOT / 'data/train.csv', usecols=core.ALLOWED + ['SI VACV'])
    y = tr['SI VACV'].to_numpy(float)
    label = (y == 10000).astype(int)
    mols = [Chem.MolFromSmiles(s) for s in tr.SMILES]
    canonical = np.array([Chem.MolToSmiles(m) for m in mols])
    gen = F.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    X = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=float)
    dot = X @ X.T
    norms = X.sum(1)
    molK = dot / np.maximum(norms[:, None] + norms[None, :] - dot, 1e-12)
    raw = tr[['strain', 'Culture']].fillna('UNKNOWN').astype(str)
    context = sum(((raw[c].to_numpy()[:, None] == raw[c].to_numpy()[None, :]).astype(float) for c in raw)) / 2
    previous = pd.read_csv(ROOT / 'results/validation.csv')
    literature = lookup_predictions(tr[core.ALLOWED]).prediction.to_numpy(float)
    matched = np.isfinite(literature)
    report = {'protocols': {}, 'literature_protocols': {}, 'covered_training_rows': int(matched.sum()), 'caveat': 'Repeated CV on previously explored data; not independent holdout.'}
    names = ['text_1', 'text_10', 'text_100', 'hybrid_1', 'hybrid_10', 'morgan_cat']
    for seed in [42, 2026, None]:
        protocol = str(seed) if seed else 'scaffold'
        groups = canonical if seed else tr.SMILES.map(lambda s: core.MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
        ps = {n: np.zeros(len(y)) for n in names}
        active = np.zeros(len(y))
        protected = np.zeros(len(y), dtype=bool)
        base = previous[protocol].to_numpy()
        for fold, (ix, va) in enumerate(GroupKFold(5, shuffle=seed is not None, random_state=seed).split(tr, groups=groups)):
            assert not set(canonical[ix]) & set(canonical[va])
            original, models = core.fit_predict_v6(tr.iloc[ix], y[ix], tr.iloc[va])
            np.testing.assert_allclose(original, base[va], atol=1e-06, rtol=1e-10)
            active[va] = np.clip(models['active_regressor'].predict(core.serialization_features(tr.iloc[va].SMILES)), 0, 5000)
            protected[va] = core.apply_inactivity_evidence(tr.iloc[ix], y[ix], tr.iloc[va], np.zeros(len(va)))[0] == 10000
            vector = TfidfVectorizer(analyzer='char', ngram_range=(2, 6), lowercase=False, sublinear_tf=True)
            a = vector.fit_transform(tr.iloc[ix].SMILES)
            b = vector.transform(tr.iloc[va].SMILES)
            enc = OneHotEncoder(handle_unknown='ignore')
            ca = enc.fit_transform(raw.iloc[ix])
            cb = enc.transform(raw.iloc[va])
            fa = hstack([a, 0.5 * ca], format='csr')
            fb = hstack([b, 0.5 * cb], format='csr')
            for c in [1, 10, 100]:
                lr = LogisticRegression(C=c, max_iter=1000, random_state=42).fit(fa, label[ix])
                ps['text_' + str(c)][va] = lr.predict_proba(fb)[:, 1]
            ktrain = 0.5 * molK[np.ix_(ix, ix)] + 0.3 * (a @ a.T).toarray() + 0.2 * context[np.ix_(ix, ix)]
            kval = 0.5 * molK[np.ix_(va, ix)] + 0.3 * (b @ a.T).toarray() + 0.2 * context[np.ix_(va, ix)]
            for c in [1, 10]:
                svc = SVC(C=c, kernel='precomputed', probability=True, random_state=42).fit(ktrain, label[ix])
                ps['hybrid_' + str(c)][va] = svc.predict_proba(kval)[:, 1]
            fa = pd.DataFrame(X[ix])
            fb = pd.DataFrame(X[va])
            cats = []
            for col in raw:
                cats.append(fa.shape[1])
                fa[col] = raw.iloc[ix][col].to_numpy()
                fb[col] = raw.iloc[va][col].to_numpy()
            model = CatBoostClassifier(iterations=400, depth=4, learning_rate=0.04, l2_leaf_reg=5, thread_count=4, verbose=False, random_seed=42, allow_writing_files=False)
            model.fit(fa, label[ix], cat_features=cats)
            ps['morgan_cat'][va] = model.predict_proba(fb)[:, 1]
            print(protocol, fold, 'elapsed', round(time.perf_counter() - t, 1), flush=True)
        rows = {}
        saved = pd.DataFrame({'actual': y, 'baseline': base, 'active_estimate': active})

        def metrics(pred):
            return dict(mse=float(np.mean((pred - y) ** 2)), ordinary_mse=float(np.mean((pred[y <= 10000] - y[y <= 10000]) ** 2)), inactive_mse=float(np.mean((pred[label == 1] - y[label == 1]) ** 2)))
        rows['baseline'] = metrics(base)
        combined_rows = {}
        combined_base = base.copy()
        combined_base[matched] = literature[matched]
        combined_rows['baseline'] = metrics(combined_base)
        for name, p in ps.items():
            pred = 10000 * p + (1 - p) * active
            pred[protected] = 10000
            saved[name + '_probability'] = p
            for weight in [0.25, 0.5, 1.0]:
                v = (1 - weight) * base + weight * pred
                key = name + '_blend' + str(weight)
                rows[key] = dict(**metrics(v), brier=float(np.mean((p - label) ** 2)))
                saved[key] = v
                combined = v.copy()
                combined[matched] = literature[matched]
                combined_rows[key] = metrics(combined)
                saved['literature_' + key] = combined
        report['protocols'][protocol] = rows
        report['literature_protocols'][protocol] = combined_rows
        saved.to_csv(OUT / (protocol + '_oof.csv'), index=False)
        print(protocol, json.dumps(sorted(rows.items(), key=lambda kv: kv[1]['ordinary_mse'])[:5]), flush=True)
    report['stable_improvements'] = [name for name in report['protocols']['42'] if name != 'baseline' and all((report['protocols'][p][name]['mse'] < report['protocols'][p]['baseline']['mse'] for p in report['protocols']))]
    report['stable_literature_improvements'] = [name for name in report['literature_protocols']['42'] if name != 'baseline' and all(report['literature_protocols'][p][name]['mse'] < report['literature_protocols'][p]['baseline']['mse'] for p in report['literature_protocols'])]
    report['runtime_seconds'] = time.perf_counter() - t
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print('done', report['runtime_seconds'], flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
