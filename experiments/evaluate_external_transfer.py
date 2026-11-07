"""Compare external transfer under molecular and scaffold holdouts.

External parents overlapping ANY official train/test parent are excluded before
training. Scaffold holdouts also remove external validation scaffolds. Molecular
holdouts retain different molecules on a known scaffold, reflecting that task.
No test outcomes, assay concentrations, titles, or IDs enter any predictor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import corrected_pipeline as cp
from collect_training_v2 import parent_key


def fingerprints(frame):
    generators = [rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048),
                  rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048),
                  rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)]
    molecules = [Chem.MolFromSmiles(parent_key(s)) for s in frame.SMILES]
    out = [np.stack([g.GetFingerprintAsNumPy(m) for m in molecules]).astype(float)
           for g in generators]
    out.append(np.stack([np.array(MACCSkeys.GenMACCSKeys(m)) for m in molecules]).astype(float))
    return out


def inverse(values, transform):
    if transform == 'log':
        return np.maximum(np.expm1(values), 0.)
    if transform == 'sqrt':
        return np.maximum(values, 0.)**2
    return np.maximum(values, 0.)


def kernel_predict(ktt, kvt, labels, weights, alpha, transform):
    # Weighted kernel ridge, including an unpenalized fold-fitted intercept.
    w = weights / weights.sum()
    yt = cp.target_transform(labels, transform)
    mean_y = float(w @ yt)
    column_mean = w @ ktt
    grand_mean = float(column_mean @ w)
    a = ktt - column_mean[None, :] - column_mean[:, None] + grand_mean
    b = kvt - column_mean[None, :] - (kvt @ w)[:, None] + grand_mean
    a = a.copy()
    a.flat[::len(a)+1] += alpha / weights
    coefficients = cho_solve(cho_factor(a, lower=True, check_finite=False),
                             yt - mean_y, check_finite=False)
    return inverse(mean_y + b @ coefficients, transform)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    train, test = cp.read_inputs()
    external = pd.read_csv(cp.DATA/'external_training_candidate_v3.csv')
    source_report = json.loads((cp.DATA/'external_training_candidate_v3.validation.json').read_text())
    assert cp.sha(cp.DATA/'external_training_candidate_v3.csv') == source_report['output_sha256']
    official_parents = set(pd.concat([train.SMILES, test.SMILES]).map(parent_key))
    assert not set(external.SMILES.map(parent_key)) & official_parents
    combined = pd.concat([train[cp.ALLOWED], external[cp.ALLOWED]], ignore_index=True)
    y = train[cp.TARGET].to_numpy(float)
    all_y = np.r_[y, external.model_target.to_numpy(float)]
    _, scaffolds, _, descriptors = cp.feature_arrays(combined)
    fps = fingerprints(combined)
    kernels = [cp.tanimoto(x, x) for x in fps]
    contexts = cp.category_values(combined)
    # Fixed positive semidefinite sum kernel; equality indicators use no labels.
    context = .5 * (contexts[:, None, 0] == contexts[None, :, 0]) + .5 * (
        contexts[:, None, 1] == contexts[None, :, 1])
    reps = {'morgan': kernels[0], 'multifp': sum(kernels)/len(kernels)}
    configs = {f'{rep}_{pool}_a{alpha}_{transform}':
               {'representation':rep, 'pool':pool, 'alpha':alpha, 'transform':transform}
               for rep in ['morgan', 'multifp', 'descriptors']
               for pool in ['official', 'external'] for alpha in [.1, 1.]
               for transform in ['raw', 'sqrt', 'log']}
    splits = cp.SCAFFOLD + cp.MOLECULAR if args.all_splits else cp.DISCOVERY
    report = {'status':'running', 'task_version':cp.TASK_VERSION, 'configs':configs,
              'feature_allowlist':cp.ALLOWED, 'splits':splits,
              'source_sha256':{'script':cp.sha(__file__), 'train':cp.TRAIN_SHA256,
                               'external':cp.sha(cp.DATA/'external_training_candidate_v3.csv')},
              'external_weight':.25, 'competition_parent_overlap':0,
              'external_exclusion':{'molecular':'all official parents excluded globally',
                                    'scaffold':'also remove all validation scaffolds'},
              'test_predictions_generated':False, 'target_clipping':False,
              'selection':'Exploratory comparison. No automatic production selection or Kaggle upload.',
              'metrics':{}, 'known_culture_metrics':{}, 'folds':{},
              'limitations':['Repeated splits share the same 320 labels.',
                             'The best of many exploratory configurations is optimistic.',
                             'Molecular and unseen-scaffold risks are different estimands.']}
    cp.save_json(args.output/'plan.json', report)
    for split in splits:
        start = time.monotonic()
        ref = pd.read_csv(args.reference/f'oof_{split}.csv')
        np.testing.assert_array_equal(ref.target, y)
        predictions = {n:np.full(len(y),np.nan) for n in configs}
        predictions['rf_log'] = ref.rf_log.to_numpy()
        known = ref.known_culture.to_numpy(bool)
        ids = ref.fold.to_numpy(int)
        report['folds'][split] = []
        for fold, ix, va in cp.folds_for(train,split):
            np.testing.assert_array_equal(np.flatnonzero(ids==fold), va)
            ex = np.arange(len(train), len(combined))
            if split in cp.SCAFFOLD:
                ex = ex[~np.isin(scaffolds[ex],scaffolds[va])]
                assert not set(scaffolds[ex]) & set(scaffolds[va])
            report['folds'][split].append({'fold':fold, 'train_positions':ix.tolist(),
                                          'validation_positions':va.tolist(),
                                          'external_positions':(ex-len(train)).tolist()})
            for pool in ['official','external']:
                tr = ix if pool=='official' else np.r_[ix,ex]
                weights = np.ones(len(tr)); weights[len(ix):] = .25
                # Imputation and scaling fit only the rows used in this fold.
                imp = SimpleImputer(strategy='median',keep_empty_features=True)
                scaler = StandardScaler()
                da = scaler.fit_transform(imp.fit_transform(descriptors[tr]))
                db = scaler.transform(imp.transform(descriptors[va]))
                # Cauchy/Rational Quadratic kernel has a longer tail than RBF.
                sqt = np.maximum(0., ((da[:,None,:]-da[None,:,:])**2).mean(2))
                sqv = np.maximum(0., ((db[:,None,:]-da[None,:,:])**2).mean(2))
                for rep in ['morgan','multifp','descriptors']:
                    if rep=='descriptors':
                        a,b=1/(1+sqt),1/(1+sqv)
                    else:
                        a,b=reps[rep][np.ix_(tr,tr)],reps[rep][np.ix_(va,tr)]
                    a=.8*a+.2*context[np.ix_(tr,tr)]
                    b=.8*b+.2*context[np.ix_(va,tr)]
                    for alpha in [.1,1.]:
                        for transform in ['raw','sqrt','log']:
                            name=f'{rep}_{pool}_a{alpha}_{transform}'
                            predictions[name][va]=kernel_predict(a,b,all_y[tr],weights,alpha,transform)
        assert all(np.isfinite(p).all() for p in predictions.values())
        report['metrics'][split]={n:cp.metrics(y,p) for n,p in predictions.items()}
        report['known_culture_metrics'][split]={n:cp.metrics(y[known],p[known]) for n,p in predictions.items()}
        pd.DataFrame({'target':y,'fold':ids,'known_culture':known,**predictions}).to_csv(
            args.output/f'oof_{split}.csv',index=False)
        cp.save_json(args.output/'metrics.json',report)
        print(split,round(time.monotonic()-start,1),sorted(
            [(n,round(m['mse'])) for n,m in report['metrics'][split].items()],key=lambda x:x[1])[:5],flush=True)
    for label,protocols in [('scaffold_mean',cp.SCAFFOLD),('molecular2_mean',cp.MOLECULAR[:2]),
                            ('molecular8_mean',cp.MOLECULAR)]:
        if set(protocols)<=set(splits):
            report[label]={n:{k:float(np.mean([report['metrics'][s][n][k] for s in protocols]))
                              for k in ['mse','mae']} for n in predictions}
    report['status']='completed'
    cp.save_json(args.output/'metrics.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--all-splits',action='store_true')
    with threadpool_limits(limits=1):main(parser.parse_args())
