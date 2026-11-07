"""Regression for the official train_v2 labels; no literature test lookup.

Only SMILES, strain and Culture enter a predictor. All target transforms and
preprocessors are fitted inside the training fold. Reported errors always use
the original, untransformed SI, including zeros and the largest observation.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
ALLOWED = ['SMILES', 'strain', 'Culture']
TARGET = 'SI VACV'
TASK_VERSION = 'official_train_v2_zero_inactivity_2026-11-07'
TRAIN_SHA256 = 'd9a876a67b5623d9cdb6592fc53d31ffaa7c535b09968c3a538138740fe246e8'
SCAFFOLD = ['scaffold', 'scaffold_shuffle42', 'scaffold_shuffle2026']
MOLECULAR = ['42', '2026', '117', '2027', '7', '23', '101', '509']
DISCOVERY = SCAFFOLD + MOLECULAR[:2]
BASELINE = 'rf_log'
MODELS = ['rf_log', 'rf_sqrt', 'rf_raw', 'et_log', 'et_sqrt', 'et_raw',
          'text_log', 'text_raw', 'tanimoto_log', 'tanimoto_raw']
SELECTED_FILE = ROOT / 'results/corrected_selection.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


@lru_cache(maxsize=None)
def molecule_features(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError('Invalid molecular structure')
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False) or 'ACYCLIC'
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fingerprint = generator.GetFingerprintAsNumPy(molecule).astype(float)
    descriptors = []
    for _, function in Descriptors._descList:
        try:
            value = float(function(molecule))
        except (ValueError, OverflowError, ZeroDivisionError):
            value = np.nan
        descriptors.append(value if np.isfinite(value) else np.nan)
    return canonical, scaffold, fingerprint, np.asarray(descriptors)


def allowed_frame(frame):
    if not set(ALLOWED) <= set(frame.columns):
        raise ValueError('Missing SMILES/strain/Culture')
    result = frame[ALLOWED].copy()
    if result.SMILES.isna().any():
        raise ValueError('Missing SMILES')
    return result


def feature_arrays(frame):
    frame = allowed_frame(frame)
    records = [molecule_features(str(s)) for s in frame.SMILES]
    return (np.array([r[0] for r in records]), np.array([r[1] for r in records]),
            np.stack([r[2] for r in records]), np.stack([r[3] for r in records]))


def category_values(frame):
    # Keep culture and strain as separate categories. No target-dependent aliases.
    return frame[['Culture', 'strain']].fillna('UNKNOWN').astype(str).map(
        lambda s: ''.join(c for c in s.casefold() if c.isalnum()) or 'unknown').to_numpy()


def target_transform(values, transform):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Targets must be finite and nonnegative')
    return np.log1p(values) if transform == 'log' else np.sqrt(values) if transform == 'sqrt' else values.copy()


def inverse_target(values, transform):
    values = np.asarray(values, dtype=float)
    if transform == 'log':
        values = np.expm1(values)
    elif transform == 'sqrt':
        values = np.maximum(values, 0.) ** 2
    values = np.maximum(values, 0.)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite predictions')
    return values


def tanimoto(a, b):
    intersection = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


class Regressor:
    def __init__(self, name):
        if name not in MODELS:
            raise ValueError(f'Unknown model: {name}')
        self.name = name
        self.family, self.transform = name.split('_')

    def fit(self, frame, labels, sample_weight=None):
        frame = allowed_frame(frame)
        labels = np.asarray(labels, dtype=float)
        if len(frame) != len(labels):
            raise ValueError('Labels and feature rows differ')
        self.training_rows = len(frame)
        y = target_transform(labels, self.transform)
        if self.family == 'text':
            self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 6),
                                             lowercase=False, sublinear_tf=True)
            x = self.vectorizer.fit_transform(frame.SMILES)
            self.model = Ridge(alpha=1.).fit(x, y, sample_weight=sample_weight)
        else:
            _, _, fp, desc = feature_arrays(frame)
            if self.family == 'tanimoto':
                self.fingerprint = fp
                self.model = KernelRidge(alpha=.1, kernel='precomputed').fit(tanimoto(fp, fp), y,
                                                                         sample_weight=sample_weight)
            else:
                self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                x = np.column_stack([fp, self.imputer.fit_transform(desc),
                                     self.encoder.fit_transform(category_values(frame))])
                cls = RandomForestRegressor if self.family == 'rf' else ExtraTreesRegressor
                self.model = cls(n_estimators=400, min_samples_leaf=2, max_features=.5,
                                 random_state=42, n_jobs=2).fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, frame):
        frame = allowed_frame(frame)
        if self.family == 'text':
            x = self.vectorizer.transform(frame.SMILES)
        else:
            _, _, fp, desc = feature_arrays(frame)
            x = tanimoto(fp, self.fingerprint) if self.family == 'tanimoto' else np.column_stack([
                fp, self.imputer.transform(desc), self.encoder.transform(category_values(frame))])
        return inverse_target(self.model.predict(x), self.transform)


def read_inputs():
    if sha(DATA / 'train_v2.csv') != TRAIN_SHA256:
        raise ValueError('train_v2.csv changed: audit a new official version before training')
    training = pd.read_csv(DATA / 'train_v2.csv', usecols=ALLOWED + [TARGET])
    test = pd.read_csv(DATA / 'test.csv', usecols=['index'] + ALLOWED)
    sample = pd.read_csv(DATA / 'sample_submit.csv', usecols=['index'])
    np.testing.assert_array_equal(test['index'], sample['index'])
    if not test['index'].is_unique:
        raise ValueError('Duplicate test index')
    target_transform(training[TARGET].to_numpy(), 'raw')
    return training, test


def metrics(y, p):
    error = np.asarray(p) - np.asarray(y)
    squared = error ** 2
    return {'mse': float(squared.mean()), 'mae': float(abs(error).mean()),
            'rmse': float(np.sqrt(squared.mean())),
            'largest_row_share_of_sse': float(squared.max() / squared.sum()) if squared.sum() else 0.}


def folds_for(frame, split):
    canonical, scaffolds, _, _ = feature_arrays(frame)
    seed = None if split == 'scaffold' else int(split.replace('scaffold_shuffle', ''))
    groups = scaffolds if split in SCAFFOLD else canonical
    for fold, (ix, va) in enumerate(GroupKFold(5, shuffle=seed is not None, random_state=seed).split(frame, groups=groups)):
        assert not set(canonical[ix]) & set(canonical[va])
        if split in SCAFFOLD:
            assert not set(scaffolds[ix]) & set(scaffolds[va])
        yield fold, ix, va


def validate(names, splits, output):
    output.mkdir(parents=True, exist_ok=True)
    frame, _ = read_inputs()
    y = frame[TARGET].to_numpy(float)
    report = {'task_version': TASK_VERSION, 'status': 'running', 'models': names, 'splits': splits,
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                                [DATA / 'train_v2.csv', DATA / 'test.csv', Path(__file__)]},
              'selection_rule_frozen_before_discovery': 'Compare to fixed rf_log. A model may be selected only if scaffold mean improves >=10%, each scaffold worsens <=5%, known-Culture scaffold mean does not worsen, and two molecular mean does not worsen. Choose smallest scaffold mean, then confirm all eight molecular protocols.',
              'feature_allowlist': ALLOWED, 'external_test_retrieval': False,
              'external_measurements_used': False, 'target_clipping': False,
              'metrics': {}, 'known_culture_metrics': {}, 'folds': {},
              'rdkit_version': rdBase.rdkitVersion,
              'limitations': ['Only 320 training rows; split protocols reuse the same dataset.',
                             'The largest target remains in all aggregate MSE calculations.',
                             'Legacy sentinel-based metrics are not comparable to train_v2 metrics.',
                             'No Public improvement is established by local validation.']}
    for split in splits:
        predictions = {name: np.full(len(y), np.nan) for name in names}
        predictions['fold_mean'] = np.full(len(y), np.nan)
        predictions['zero'] = np.zeros(len(y))
        known, ids = np.zeros(len(y), bool), np.full(len(y), -1, int)
        report['folds'][split] = []
        for fold, ix, va in folds_for(frame, split):
            start = time.perf_counter()
            ids[va] = fold
            known[va] = np.isin(category_values(frame.iloc[va])[:, 0], category_values(frame.iloc[ix])[:, 0])
            predictions['fold_mean'][va] = y[ix].mean()
            for name in names:
                model = Regressor(name).fit(frame.iloc[ix], y[ix])
                predictions[name][va] = model.predict(frame.iloc[va])
            report['folds'][split].append({'fold': fold, 'train_positions': ix.tolist(),
                                          'validation_positions': va.tolist()})
            print(split, fold, 'seconds', round(time.perf_counter() - start, 2), flush=True)
        assert (ids >= 0).all()
        report['metrics'][split] = {name: metrics(y, p) for name, p in predictions.items()}
        report['known_culture_metrics'][split] = {name: metrics(y[known], p[known]) for name, p in predictions.items()}
        pd.DataFrame({'target': y, 'fold': ids, 'known_culture': known, **predictions}).to_csv(output / f'oof_{split}.csv', index=False)
        save_json(output / 'metrics.json', report)
    for group, protocols in [('scaffold_mean', SCAFFOLD), ('molecular2_mean', MOLECULAR[:2]),
                             ('molecular5_mean', MOLECULAR[:5]), ('molecular_extra3_mean', MOLECULAR[5:])]:
        if not set(protocols) <= set(splits):
            continue
        report[group] = {name: {key: float(np.mean([report['metrics'][s][name][key] for s in protocols]))
                                for key in ['mse', 'mae']} for name in predictions}
        report[group + '_known_culture'] = {name: {key: float(np.mean([report['known_culture_metrics'][s][name][key] for s in protocols]))
                                                 for key in ['mse', 'mae']} for name in predictions}
    report['material_checks'] = {}
    if set(DISCOVERY) <= set(splits):
        for name in names:
            if name == BASELINE:
                continue
            report['material_checks'][name] = {
                'scaffold_mean_improves10percent': report['scaffold_mean'][name]['mse'] <= .9 * report['scaffold_mean'][BASELINE]['mse'],
                'each_scaffold_no_more_than5percent_worse': all(report['metrics'][s][name]['mse'] <= 1.05 * report['metrics'][s][BASELINE]['mse'] for s in SCAFFOLD),
                'known_culture_scaffold_mean_no_worse': report['scaffold_mean_known_culture'][name]['mse'] <= report['scaffold_mean_known_culture'][BASELINE]['mse'],
                'molecular2_mean_no_worse': report['molecular2_mean'][name]['mse'] <= report['molecular2_mean'][BASELINE]['mse'],
            }
    report['passing'] = [name for name, checks in report['material_checks'].items() if all(checks.values())]
    report['selected_for_confirmation'] = min(report['passing'], key=lambda n: report['scaffold_mean'][n]['mse']) if report['passing'] else BASELINE
    report['status'] = 'completed'
    save_json(output / 'metrics.json', report)
    print(json.dumps({key: report.get(key) for key in ['scaffold_mean', 'molecular2_mean', 'passing', 'selected_for_confirmation']}, indent=2), flush=True)
    return report


def export(name, output):
    output.mkdir(parents=True, exist_ok=True)
    frame, test = read_inputs()
    y = frame[TARGET].to_numpy(float)
    model = Regressor(name).fit(frame, y)
    pred = model.predict(test)
    pd.DataFrame({'index': test['index'], TARGET: pred}).to_csv(output / 'submission.csv', index=False)
    frame.to_csv(output / 'training_used.csv', index=False)
    joblib.dump(model, output / 'model.joblib')
    report = {'task_version': TASK_VERSION, 'model': name, 'train_rows': len(frame), 'test_rows': len(test),
              'feature_allowlist': ALLOWED, 'external_measurements_used': False,
              'inactivity_prediction': 'SI is a nonnegative regression target; 10000 has no sentinel role',
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [DATA / 'train_v2.csv', DATA / 'test.csv', Path(__file__)]},
              'submission_sha256': sha(output / 'submission.csv'),
              'training_used_sha256': sha(output / 'training_used.csv'),
              'prediction_quantiles': np.quantile(pred, [0, .5, .9, .95, .99, 1]).tolist(),
              'status': 'local_not_submitted', 'public_mse': None}
    save_json(output / 'export.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'run_v2')
    parser.add_argument('--models', nargs='+', choices=MODELS)
    parser.add_argument('--all-splits', action='store_true')
    parser.add_argument('--skip-cv', action='store_true')
    parser.add_argument('--export-model', choices=MODELS)
    args = parser.parse_args()
    if args.models:
        validate(args.models, SCAFFOLD + MOLECULAR if args.all_splits else DISCOVERY, args.output)
    else:
        name = args.export_model
        if name is None:
            if not SELECTED_FILE.exists():
                raise ValueError('No corrected selected model yet; first run --models rf_log et_log ...')
            selected = json.loads(SELECTED_FILE.read_text())
            if selected['task_version'] != TASK_VERSION:
                raise ValueError('Selected model targets another dataset version')
            name = selected['model']
        if not args.skip_cv:
            validate([BASELINE] if name == BASELINE else [BASELINE, name], SCAFFOLD + MOLECULAR, args.output)
        export(name, args.output)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
