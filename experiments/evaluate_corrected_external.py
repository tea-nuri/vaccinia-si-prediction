"""Evaluate a filtered external candidate pool with grouped scaffold exclusions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import corrected_pipeline as cp
from collect_training_v2 import parent_key

HERE = ROOT/'run_corrected_candidates'

def forest_x(model, frame):
    _, _, fp, desc = cp.feature_arrays(frame)
    return np.column_stack([fp, model.imputer.transform(desc),
                            model.encoder.transform(cp.category_values(frame))])

def forest_means(model, train, y, validation, weights=None):
    """Estimate a conditional arithmetic mean in log-trained tree partitions.

    Tree structure is learned from training labels only. Leaf estimates use all
    training observations, never held-out labels. This is not an OOB estimate.
    """
    tx, vx = forest_x(model, train), forest_x(model, validation)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    result = np.zeros(len(validation))
    for tree in model.model.estimators_:
        leaf_train, leaf_val = tree.apply(tx), tree.apply(vx)
        counts = np.bincount(leaf_train, weights=w, minlength=tree.tree_.node_count)
        sums = np.bincount(leaf_train, weights=w*y, minlength=tree.tree_.node_count)
        assert (counts[leaf_val] > 0).all()
        result += sums[leaf_val] / counts[leaf_val]
    return result / len(model.model.estimators_)
WEIGHTS = {'025':.25,'1':1.,'4':4.}


def main(source=None, output=None, reference=None):
    output = output or HERE/'expanded_external_validation'
    output.mkdir(parents=True, exist_ok=True)
    reference = reference or HERE/'baseline'
    source = source or cp.DATA/'external_training_candidate_v3.csv'
    meta = json.loads(source.with_suffix('.validation.json').read_text())
    assert cp.sha(source)==meta.get('expanded_pool_sha256',meta.get('output_sha256'))
    ext = pd.read_csv(source)
    train,test = cp.read_inputs()
    forbidden = set(pd.concat([train.SMILES,test.SMILES]).map(parent_key))
    assert not set(ext.SMILES.map(parent_key))&forbidden
    _,es,_,_ = cp.feature_arrays(ext)
    _,ts,_,_ = cp.feature_arrays(train)
    y = train[cp.TARGET].to_numpy(float)
    names = [f'expanded_w{w}_{t}' for w in WEIGHTS for t in ['log','leafmean']]
    plan = {'task_version':cp.TASK_VERSION,'models':names,'splits':cp.SCAFFOLD+cp.MOLECULAR,
            'external_rows':len(ext),'new_external_rows':meta.get('new_rows',meta.get('new_chembl_rows',0)+meta.get('new_2026_paper_rows',0)),
            'source_sha256':{'script':cp.sha(__file__),
                              'external':cp.sha(source),'train':cp.sha(cp.DATA/'train_v2.csv'),
                              'pipeline':cp.sha(cp.__file__)},
            'external_weights':WEIGHTS,'external_validation_scaffolds_excluded':True,
            'competition_parent_overlap':0,'test_predictions_generated':False,'target_clipping':False,
            'selection_rule':'Prior corrected-task gate: >=10% scaffold MSE gain, each scaffold <=5% worse, known-Culture scaffold mean and molecular2 mean no worse.'}
    cp.save_json(output/'plan.json',plan)
    report = dict(plan,status='running',metrics={},known_culture_metrics={},folds={})
    for split in plan['splits']:
        start = time.monotonic()
        ref = pd.read_csv(reference/f'oof_{split}.csv')
        np.testing.assert_array_equal(ref.target,y)
        preds = {n:np.full(len(y),np.nan) for n in names}
        preds['rf_log'] = ref.rf_log.to_numpy()
        ids = np.full(len(y),-1,int)
        known = np.zeros(len(y),bool)
        report['folds'][split] = []
        for fold,ix,va in cp.folds_for(train,split):
            np.testing.assert_array_equal(np.flatnonzero(ref.fold.to_numpy()==fold),va)
            ex = np.flatnonzero(~np.isin(es,ts[va]))
            assert not set(es[ex])&set(ts[va])
            frame = pd.concat([train.iloc[ix][cp.ALLOWED],ext.iloc[ex][cp.ALLOWED]],ignore_index=True)
            labels = np.r_[y[ix],ext.iloc[ex].model_target.to_numpy(float)]
            for label,weight in WEIGHTS.items():
                sw = np.r_[np.ones(len(ix)),np.full(len(ex),weight)]
                model = cp.Regressor('rf_log').fit(frame,labels,sample_weight=sw)
                preds[f'expanded_w{label}_log'][va] = model.predict(train.iloc[va])
                preds[f'expanded_w{label}_leafmean'][va] = forest_means(model,frame,labels,train.iloc[va],sw)
            known[va] = np.isin(cp.category_values(train.iloc[va])[:,0],cp.category_values(train.iloc[ix])[:,0])
            ids[va] = fold
            report['folds'][split].append({'fold':fold,'training_positions':ix.tolist(),
                                          'validation_positions':va.tolist(),'external_positions':ex.tolist()})
        np.testing.assert_array_equal(ref.known_culture,known)
        assert (ids>=0).all() and all(np.isfinite(p).all() for p in preds.values())
        report['metrics'][split] = {n:cp.metrics(y,p) for n,p in preds.items()}
        report['known_culture_metrics'][split] = {n:cp.metrics(y[known],p[known]) for n,p in preds.items()}
        pd.DataFrame({'target':y,'fold':ids,'known_culture':known,**preds}).to_csv(output/f'oof_{split}.csv',index=False)
        cp.save_json(output/'metrics.json',report)
        print(split,'seconds',round(time.monotonic()-start,1),
              json.dumps({n:round(v['mse']) for n,v in report['metrics'][split].items()}),flush=True)
    for label,protocols in [('scaffold_mean',cp.SCAFFOLD),('molecular2_mean',cp.MOLECULAR[:2]),
                            ('molecular5_mean',cp.MOLECULAR[:5]),('molecular_extra3_mean',cp.MOLECULAR[5:])]:
        for field,suffix in [('metrics',''),('known_culture_metrics','_known_culture')]:
            report[label+suffix] = {n:{k:float(np.mean([report[field][s][n][k] for s in protocols]))
                                      for k in ['mse','mae']} for n in preds}
    report['material_checks'] = {}
    for name in names:
        report['material_checks'][name] = {
            'scaffold_mean_improves10percent':report['scaffold_mean'][name]['mse']<=.9*report['scaffold_mean']['rf_log']['mse'],
            'each_scaffold_no_more_than5percent_worse':all(report['metrics'][s][name]['mse']<=1.05*report['metrics'][s]['rf_log']['mse'] for s in cp.SCAFFOLD),
            'known_culture_scaffold_mean_no_worse':report['scaffold_mean_known_culture'][name]['mse']<=report['scaffold_mean_known_culture']['rf_log']['mse'],
            'molecular2_mean_no_worse':report['molecular2_mean'][name]['mse']<=report['molecular2_mean']['rf_log']['mse']}
    report['passing'] = [n for n,c in report['material_checks'].items() if all(c.values())]
    report['status']='completed'
    cp.save_json(output/'metrics.json',report)
    print(json.dumps({'scaffold_mean':report['scaffold_mean'],'passing':report['passing']},indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.source,args.output,args.reference)
