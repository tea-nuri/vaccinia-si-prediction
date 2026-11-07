"""Paired ablation of the seven-row extension on all established protocols.

No validation target is clipped or omitted. Each scaffold fold also removes
external compounds with held-out scaffolds. Test outcomes are never loaded.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import corrected_pipeline as cp
from collect_training_v2 import parent_key
from molformer_candidate import rbf
from experiments.evaluate_external_transfer import kernel_predict


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    train, test = cp.read_inputs()
    source = cp.DATA / 'external_training_high_si_v4.csv'
    ext = pd.read_csv(source)
    base = pd.read_csv(cp.DATA / 'external_training_candidate_v3.csv')
    assert ext.iloc[:len(base)].SMILES.tolist() == base.SMILES.tolist()
    forbidden = set(pd.concat([train.SMILES, test.SMILES]).map(parent_key))
    assert not set(ext.SMILES.map(parent_key)) & forbidden
    frame = pd.concat([train[cp.ALLOWED], ext[cp.ALLOWED]], ignore_index=True)
    y = train[cp.TARGET].to_numpy(float)
    labels = np.r_[y, ext.model_target.to_numpy(float)]
    meta = json.loads(args.embeddings.with_suffix('.json').read_text())
    assert meta['embedding_sha256'] == cp.sha(args.embeddings)
    assert meta['raw_inputs'] == frame.SMILES.tolist()
    assert meta['source_sha256'] == {'train':cp.TRAIN_SHA256, 'external':cp.sha(source)}
    assert meta['revision'] == '361063d0ad524ef77cf39b08469f6be770dc550f'
    vectors = np.load(args.embeddings, allow_pickle=False)['vectors'].astype(float)
    assert vectors.shape == (len(frame), 768) and np.isfinite(vectors).all()
    _, scaffolds, fp, _ = cp.feature_arrays(frame)
    chemical = cp.tanimoto(fp, fp)
    context = cp.category_values(frame)
    ctx = .5*(context[:,None,0] == context[None,:,0]) + .5*(context[:,None,1] == context[None,:,1])
    # Fixed before observing any new OOF results: paired data ablation across
    # four target powers, plus two sensitivity weights for the default head.
    configs = {f'{pool}_p{power}_w0.25':dict(pool=pool, power=power, weight=.25)
               for pool in ['base60', 'expanded67'] for power in [.5, .75, .9, 1.]}
    configs.update({f'{pool}_p0.5_w{w}':dict(pool=pool, power=.5, weight=w)
                    for pool in ['base60', 'expanded67'] for w in [.1, 1.]})
    if args.kernel_sweep:
        configs = {'base60_p0.5_w0.25':dict(pool='base60', power=.5, weight=.25)}
        configs.update({f'expanded67_gamma{g}_fp{p}_mix{mix}':dict(
            pool='expanded67', power=.5, weight=.25, gamma=g, fp_power=p, fp_fraction=mix)
            for g in [.25, 1., 4.] for p in [1, 2] for mix in [.25, .75]})
    report = dict(status='running', configs=configs, metrics={}, known_culture_metrics={},
                  tail_metrics={}, folds={}, task_version=cp.TASK_VERSION,
                  feature_allowlist=cp.ALLOWED, competition_parent_overlap=0,
                  source_sha256=dict(script=cp.sha(__file__), train=cp.TRAIN_SHA256,
                                     external=cp.sha(source), embeddings=cp.sha(args.embeddings)),
                  target_clipping=False, test_predictions_generated=False,
                  selection='Research ablation; production selection remains unchanged.',
                  limitations=['Repeated protocols share the same 320 targets.',
                               'Large-SI subgroup metrics are diagnostic, not a replacement for full MSE.',
                               'No Kaggle improvement follows from local scores.'])
    cp.save_json(args.output / 'plan.json', report)
    for split in cp.SCAFFOLD + cp.MOLECULAR:
        started = time.monotonic()
        predictions = {name:np.full(len(y), np.nan) for name in configs}
        ids = np.full(len(y), -1)
        known = np.zeros(len(y), dtype=bool)
        report['folds'][split] = []
        for fold, ix, va in cp.folds_for(train, split):
            ids[va] = fold
            known[va] = np.isin(cp.category_values(train.iloc[va])[:,0], cp.category_values(train.iloc[ix])[:,0])
            ex = np.arange(len(train), len(frame))
            if split in cp.SCAFFOLD:
                ex = ex[~np.isin(scaffolds[ex], scaffolds[va])]
                assert not set(scaffolds[ex]) & set(scaffolds[va])
            report['folds'][split].append(dict(fold=fold, validation_positions=va.tolist(),
                                              train_positions=ix.tolist(), external_positions=ex.tolist()))
            for pool in ['base60', 'expanded67']:
                active = ex if pool == 'expanded67' else ex[ex < len(train) + len(base)]
                tr = np.r_[ix, active]
                scaler = StandardScaler()
                a = scaler.fit_transform(vectors[tr])
                b = scaler.transform(vectors[va])
                ktt = .4*rbf(a,a) + .4*chemical[np.ix_(tr,tr)] + .2*ctx[np.ix_(tr,tr)]
                kvt = .4*rbf(b,a) + .4*chemical[np.ix_(va,tr)] + .2*ctx[np.ix_(va,tr)]
                for name, config in configs.items():
                    if config['pool'] != pool:
                        continue
                    weights = np.r_[np.ones(len(ix)), np.full(len(active), config['weight'])]
                    power = config['power']
                    if 'gamma' in config:
                        g, p, mix = config['gamma'], config['fp_power'], config['fp_fraction']
                        at = .8*((1-mix)*rbf(a,a)**g + mix*chemical[np.ix_(tr,tr)]**p) + .2*ctx[np.ix_(tr,tr)]
                        bt = .8*((1-mix)*rbf(b,a)**g + mix*chemical[np.ix_(va,tr)]**p) + .2*ctx[np.ix_(va,tr)]
                    else:
                        at, bt = ktt, kvt
                    z = kernel_predict(at, bt, labels[tr]**power, weights, .1, 'raw')
                    predictions[name][va] = z**(1/power)
        assert all(np.isfinite(p).all() for p in predictions.values())
        report['metrics'][split] = {n:cp.metrics(y,p) for n,p in predictions.items()}
        report['known_culture_metrics'][split] = {n:cp.metrics(y[known],p[known]) for n,p in predictions.items()}
        report['tail_metrics'][split] = {
            n:{'high_SI':cp.metrics(y[y>=1000],p[y>=1000]),
               'other':cp.metrics(y[y<1000],p[y<1000]),
               'largest_target_prediction':float(p[np.argmax(y)])}
            for n,p in predictions.items()}
        pd.DataFrame(dict(target=y, fold=ids, known_culture=known, **predictions)).to_csv(args.output/f'oof_{split}.csv', index=False)
        cp.save_json(args.output/'metrics.json', report)
        print(split, round(time.monotonic()-started,2),
              sorted((n,round(m['mse'])) for n,m in report['metrics'][split].items()), flush=True)
    for label, protocols in [('scaffold_mean', cp.SCAFFOLD), ('molecular8_mean', cp.MOLECULAR)]:
        report[label] = {n:{k:float(np.mean([report['metrics'][s][n][k] for s in protocols]))
                          for k in ['mse','mae']} for n in configs}
    report['status'] = 'completed'
    cp.save_json(args.output/'metrics.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kernel-sweep', action='store_true', help='Separate exploratory kernel bandwidth/mixture sweep.')
    with threadpool_limits(limits=1):
        main(parser.parse_args())
