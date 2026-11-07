"""Learn molecular similarity from external endpoint measurements, then SI.

CC50 and reporter potency remain separate endpoints. No cross-assay ratios or
new SI labels are constructed. Every competition parent is excluded globally;
scaffold validation also excludes held-out scaffolds from each teacher.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import corrected_pipeline as cp
from collect_training_v2 import parent_key
from experiments.evaluate_external_transfer import kernel_predict


def structural(frame):
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    return np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(parent_key(s)))
                     for s in frame.SMILES]).astype(float)


def fit_teacher(frame,x,target,query,qx,endpoint):
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    col='Culture' if endpoint=='CC50_uM' else 'reporter_assay'
    a=np.column_stack([x,enc.fit_transform(frame[[col]].fillna('UNKNOWN').astype(str))])
    b=np.column_stack([qx,enc.transform(query[[col]].fillna('UNKNOWN').astype(str))])
    model=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,max_features=.5,
                              random_state=42,n_jobs=2)
    model.fit(a,np.log10(target),sample_weight=frame.sample_weight.to_numpy(float))
    return model,model.predict(b),model.apply(b)


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    train,test=cp.read_inputs();y=train[cp.TARGET].to_numpy(float)
    official_parents=set(pd.concat([train.SMILES,test.SMILES]).map(parent_key))
    tox=pd.read_csv(args.source/'toxicity_exact.csv')
    pot=pd.read_csv(args.source/'potency_quantitative.csv')
    mt=json.loads((args.source/'toxicity_provenance.json').read_text())
    mp=json.loads((args.source/'potency_provenance.json').read_text())
    assert cp.sha(args.source/'toxicity_exact.csv')==mt['exact_sha256']
    assert cp.sha(args.source/'potency_quantitative.csv')==mp['output_sha256']
    assert not set(tox.SMILES.map(parent_key))&official_parents
    assert not set(pot.SMILES.map(parent_key))&official_parents
    assert tox.relation.eq('=').all()
    tx,px,x=structural(tox),structural(pot),structural(train)
    _,scaffolds,_,_=cp.feature_arrays(train)
    # Recompute exact same Murcko convention for all datasets, rather than
    # assuming that saved source scaffolds follow the competition convention.
    tox['strain']='UNKNOWN'
    _,ts,_,_=cp.feature_arrays(tox)
    pot['Culture']='UNKNOWN';pot['strain']='UNKNOWN'
    _,ps,_,_=cp.feature_arrays(pot)
    chem=cp.tanimoto(x,x)
    configs={f'{rep}_a{alpha}_{transform}':{'rep':rep,'alpha':alpha,'transform':transform}
             for rep in ['toxleaf','potleaf','endpointmix'] for alpha in [.1,1.]
             for transform in ['raw','sqrt','log']}
    splits=cp.SCAFFOLD+cp.MOLECULAR if args.all_splits else cp.DISCOVERY
    plan={'task_version':cp.TASK_VERSION,'configs':configs,'splits':splits,
          'toxicity_rows':len(tox),'potency_rows':len(pot),
          'source_sha256':{'script':cp.sha(__file__), 'kernel_implementation':cp.sha(ROOT/'experiments/evaluate_external_transfer.py'),
                           'toxicity':mt['exact_sha256'],'potency':mp['output_sha256'],
                           'train':cp.TRAIN_SHA256},
          'feature_allowlist':cp.ALLOWED,'SI_labels_created':False,
          'competition_parent_overlap':0,'test_predictions_generated':False,
          'target_clipping':False,'teacher_target':'log10 of a separate measured endpoint',
          'teacher_model':'ExtraTrees, 200 trees, min_samples_leaf=2, max_features=.5, seed=42',
          'validation':'All official parents excluded globally; held-out scaffolds additionally removed for scaffold protocols.',
          'selection':'Exploratory, no automatic production replacement or Kaggle upload.',
          'limitations':['Reporter potency is not cellular selectivity.',
                         'Teachers retain source assay and cell context; unresolved duration remains a limitation.',
                         'Repeated outer splits are not independent data.']}
    cp.save_json(args.output/'plan.json',plan)
    report=dict(plan,status='running',metrics={},known_culture_metrics={},folds={},auxiliary_validation={})
    # Independent external-only scaffold check, without any competition labels.
    for label,data,mat,endpoint in [('toxicity',tox,tx,'CC50_uM'),('potency',pot,px,'auxiliary_potency_uM')]:
        target=np.log10(data[endpoint].to_numpy(float));pred=np.full(len(data),np.nan);dummy=pred.copy()
        groups=ts if label=='toxicity' else ps
        for it,iv in GroupKFold(5).split(data,groups=groups):
            _,pred[iv],_=fit_teacher(data.iloc[it],mat[it],data[endpoint].to_numpy()[it],data.iloc[iv],mat[iv],endpoint)
            dummy[iv]=np.average(target[it],weights=data.sample_weight.iloc[it])
        report['auxiliary_validation'][label]={'rows':len(data),'log10_mse':float(np.mean((pred-target)**2)),
                                              'dummy_log10_mse':float(np.mean((dummy-target)**2))}
    for split in splits:
        start=time.monotonic();ref=pd.read_csv(args.reference/f'oof_{split}.csv')
        np.testing.assert_array_equal(ref.target,y)
        predictions={n:np.full(len(y),np.nan) for n in configs};predictions['rf_log']=ref.rf_log.to_numpy()
        known=ref.known_culture.to_numpy(bool);ids=ref.fold.to_numpy(int);report['folds'][split]=[]
        for fold,ix,va in cp.folds_for(train,split):
            np.testing.assert_array_equal(np.flatnonzero(ids==fold),va)
            kt=np.flatnonzero(~np.isin(ts,scaffolds[va])) if split in cp.SCAFFOLD else np.arange(len(tox))
            kp=np.flatnonzero(~np.isin(ps,scaffolds[va])) if split in cp.SCAFFOLD else np.arange(len(pot))
            if split in cp.SCAFFOLD:
                assert not set(ts[kt])&set(scaffolds[va]) and not set(ps[kp])&set(scaffolds[va])
            _,_,tl=fit_teacher(tox.iloc[kt],tx[kt],tox.CC50_uM.to_numpy()[kt],train,x,'CC50_uM')
            q=train.copy();q['reporter_assay']=720579
            _,_,pl=fit_teacher(pot.iloc[kp],px[kp],pot.auxiliary_potency_uM.to_numpy()[kp],q,x,'auxiliary_potency_uM')
            tk=np.mean(tl[:,None,:]==tl[None,:,:],axis=2)
            pk=np.mean(pl[:,None,:]==pl[None,:,:],axis=2)
            kernels={'toxleaf':tk,'potleaf':pk,'endpointmix':.5*chem+.25*tk+.25*pk}
            for name,cfg in configs.items():
                k=kernels[cfg['rep']]
                predictions[name][va]=kernel_predict(k[np.ix_(ix,ix)],k[np.ix_(va,ix)],
                                                    y[ix],np.ones(len(ix)),cfg['alpha'],cfg['transform'])
            report['folds'][split].append({'fold':fold,'toxicity_rows':len(kt),'potency_rows':len(kp)})
        assert all(np.isfinite(p).all() for p in predictions.values())
        report['metrics'][split]={n:cp.metrics(y,p) for n,p in predictions.items()}
        report['known_culture_metrics'][split]={n:cp.metrics(y[known],p[known]) for n,p in predictions.items()}
        pd.DataFrame({'target':y,'fold':ids,'known_culture':known,**predictions}).to_csv(args.output/f'oof_{split}.csv',index=False)
        cp.save_json(args.output/'metrics.json',report)
        print(split,round(time.monotonic()-start,1),sorted([(n,round(m['mse'])) for n,m in report['metrics'][split].items()],key=lambda x:x[1])[:4],flush=True)
    for label,protocols in [('scaffold_mean',cp.SCAFFOLD),('molecular2_mean',cp.MOLECULAR[:2]),('molecular8_mean',cp.MOLECULAR)]:
        if set(protocols)<=set(splits):
            report[label]={n:{k:float(np.mean([report['metrics'][s][n][k] for s in protocols])) for k in ['mse','mae']} for n in predictions}
    report['status']='completed';cp.save_json(args.output/'metrics.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--all-splits',action='store_true')
    with threadpool_limits(limits=1):main(p.parse_args())
