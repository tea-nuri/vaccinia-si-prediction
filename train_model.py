"""Train and validate the SMILES-only model, with optional external-data ablation."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from features import features

ROOT = Path(__file__).resolve().parent


def canonical(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError('Invalid SMILES')
    return Chem.MolToSmiles(mol)


def prepare_external(frame, original_groups):
    required = ['smiles', 'SI', 'IC50_relation', 'CC50_relation', 'SI_relation', 'quality_flags']
    if any(c not in frame for c in required):
        raise ValueError('External data must come from collect_data.py')
    frame = frame.copy()
    exact = (frame[['IC50_relation', 'CC50_relation', 'SI_relation']] == '=').all(axis=1)
    clean = frame['quality_flags'].fillna('').eq('')
    frame = frame[exact & clean].copy()
    frame['canonical'] = frame['smiles'].map(canonical)
    # Exclude all competition-train molecules before any CV split, preventing
    # validation molecules from entering through a second literature record.
    frame = frame[~frame['canonical'].isin(set(original_groups))]
    return frame.groupby('canonical', as_index=False).agg(SI=('SI', 'mean'), doi=('doi', lambda v: ';'.join(sorted(set(v)))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--external', action='store_true', help='Compare external weights 0, 0.1, 0.3, 1 on two CV splits')
    p.add_argument('--output', default='run')
    args = p.parse_args()
    out = ROOT / args.output
    out.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / 'train.csv')
    test = pd.read_csv(ROOT / 'test.csv')
    sample = pd.read_csv(ROOT / 'sample_submit.csv')
    y = train['SI VACV'].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid target')
    _, x, groups, scaffolds = features(train.SMILES)
    _, xt, _, _ = features(test.SMILES)
    groups = np.array(groups)
    ext = prepare_external(pd.read_csv(ROOT / 'external_training.csv'), groups) if args.external else pd.DataFrame()
    xe = features(ext.canonical)[1] if len(ext) else np.empty((0, x.shape[1]))
    ye = ext.SI.to_numpy() if len(ext) else np.empty(0)
    prototype = make_pipeline(SimpleImputer(keep_empty_features=True),
        ExtraTreesRegressor(n_estimators=350, min_samples_leaf=3,
                            max_features=.5, random_state=42, n_jobs=4))
    weights = [0., .1, .3, 1.] if args.external and len(ext) else [0.]
    def fit(indices, weight):
        model = clone(prototype)
        xx, yy = x[indices], y[indices]
        sw = np.ones(len(indices))
        if weight > 0:
            xx = np.vstack([xx, xe]); yy = np.r_[yy, ye]
            sw = np.r_[sw, np.full(len(ye), weight)]
        model.fit(xx, yy, extratreesregressor__sample_weight=sw)
        return model
    metrics = {}; oof = {}
    for weight in weights:
        scores = []
        for seed in [42, 2026]:
            prediction = np.zeros(len(y))
            for tr, va in GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y, groups):
                assert not set(groups[tr]) & set(groups[va])
                prediction[va] = np.maximum(fit(tr, weight).predict(x[va]), 0)
            scores.append(float(mean_squared_error(y, prediction)))
            oof[f'weight_{weight}_seed_{seed}'] = prediction
        metrics[str(weight)] = {'mse_seed42': scores[0], 'mse_seed2026': scores[1], 'mean_mse': float(np.mean(scores))}
        print(weight, metrics[str(weight)], flush=True)
    chosen = min(weights, key=lambda w: metrics[str(w)]['mean_mse'])
    scaffold_pred = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=5).split(x, y, scaffolds):
        # Remove external molecules with a validation scaffold for this diagnostic.
        if chosen > 0:
            _, _, _, ext_scaffolds = features(ext.canonical)
            allowed = ~np.isin(ext_scaffolds, np.array(scaffolds)[va])
            xx = np.vstack([x[tr], xe[allowed]])
            yy = np.r_[y[tr], ye[allowed]]
            model = clone(prototype).fit(xx, yy, extratreesregressor__sample_weight=np.r_[np.ones(len(tr)), np.full(allowed.sum(), chosen)])
        else:
            model = fit(tr, 0)
        scaffold_pred[va] = np.maximum(model.predict(x[va]), 0)
    model = fit(np.arange(len(y)), chosen)
    prediction = np.maximum(model.predict(xt), 0)
    assert sample.columns.tolist() == ['index', 'SI VACV']
    assert sample['index'].equals(test['index']) and np.isfinite(prediction).all()
    sample['SI VACV'] = prediction
    sample.to_csv(out / 'submission.csv', index=False)
    training_used = pd.DataFrame({'smiles': train.SMILES, 'canonical': groups, 'SI': y, 'source': 'competition', 'weight': 1.})
    if chosen > 0:
        training_used = pd.concat([training_used, pd.DataFrame({'smiles': ext.canonical, 'canonical': ext.canonical,
                'SI': ye, 'source': ext.doi, 'weight': chosen})], ignore_index=True)
    training_used.to_csv(out / 'training_used.csv', index=False)
    pd.DataFrame({'target': y, **oof, 'scaffold_prediction': scaffold_pred}).to_csv(out / 'oof.csv', index=False)
    joblib.dump(model, out / 'model.joblib')
    report = dict(cv=metrics, selected_external_weight=chosen,
        external_unique_molecules_after_overlap_removal=len(ext), training_rows=len(training_used),
        scaffold_mse=float(mean_squared_error(y, scaffold_pred)), submission_rows=len(sample),
        input_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['train.csv','test.csv','external_training.csv'] if (ROOT/n).exists()},
        features=['RDKit descriptors from SMILES'],
        excluded_features=['CC50_uM','IC50_uM_VACV','Title','strain','Culture'],
        caveat='Two CV partitions share molecules. Model selection is optimistic; scaffold diagnostic is stricter. Public leaderboard is only 30% of test data.')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
