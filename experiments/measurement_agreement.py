"""Abstain from fixed retrieval when independent eligible measurements disagree.

Uses the existing numerical agreement tolerance (absolute 1 or relative 5%).
Does not select a study using competition labels or additional virus names.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train import ALLOWED, fit_predict_v6
from literature import lookup_predictions, structure_key
from experiments.text_gate import v9_lookup, candidate_predict
from experiments.camphor2022 import main as reproduce_v9

SELECTED_GATE='logistic_C100_context0_power2_weight0.25'


def conflicting_measurements(values):
    """Rows need at least two finite measurements; missing is not disagreement."""
    frame=pd.DataFrame(values).replace([np.inf,-np.inf],np.nan)
    spread=frame.max(axis=1)-frame.min(axis=1)
    threshold=np.maximum(1.,.05*frame.mean(axis=1).abs())
    return (frame.notna().sum(axis=1).ge(2)&spread.gt(threshold)).to_numpy()


def agreement_lookup(target):
    numerical=pd.read_csv(ROOT/'data/external.csv')
    primary=pd.read_csv(ROOT/'data/primary_literature.csv')
    observations=[lookup_predictions(target,numerical=numerical,primary=primary.iloc[:0])]
    for name in ['primary_literature.csv','literature_candidates.csv','camphor2022.csv']:
        observations.append(lookup_predictions(target,numerical=numerical.iloc[:0],
                            primary=pd.read_csv(ROOT/'data'/name),include_2019=True))
    values=np.column_stack([a.prediction.to_numpy() for a in observations])
    conflicts=conflicting_measurements(values)
    audit=v9_lookup(target)
    audit['v9_prediction']=audit.prediction
    audit['conflict']=conflicts
    audit['observations']=[json.dumps([{'value':float(a.prediction.iloc[i]),'source':a.source.iloc[i]}
                          for a in observations if pd.notna(a.prediction.iloc[i])]) for i in range(len(target))]
    audit.loc[conflicts,'prediction']=np.nan
    return audit


def main():
    output=ROOT/'run_measurement_agreement';output.mkdir(exist_ok=True)
    reproduce_v9(output/'baseline_v9')
    tr=pd.read_csv(ROOT/'data/train.csv',usecols=ALLOWED+['SI VACV'])
    te=pd.read_csv(ROOT/'data/test.csv',usecols=ALLOWED+['index'])
    y=tr['SI VACV'].to_numpy(float)
    ta,ea=agreement_lookup(tr),agreement_lookup(te)
    fixed=ta.prediction.to_numpy();previous=ta.v9_prediction.to_numpy()
    canon=tr.SMILES.map(structure_key).to_numpy()
    scaffold=tr.SMILES.map(lambda s:MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
    report={'status':'local_only_not_submitted','scores':{},
            'rule':'Refuse fixed lookup if eligible measurements disagree by more than max(1, .05*mean); use fold-trained ML. No row/source-ID exceptions.',
            'fixed_classifier':SELECTED_GATE,
            'train_conflicts':int(ta.conflict.sum()),'test_conflicts':int(ea.conflict.sum()),
            'limitations':['Rule developed after error analysis; all splits reuse the same molecules.',
                           'Effect of agreement guard currently depends on one training and one test molecule.',
                           'Disagreeing numerical measurements can still both indicate activity; ML fallback is not biological ground truth.',
                           'No claim of Public Score below one million; no Kaggle submission.'],
            'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                ['data/train.csv','data/test.csv','data/external.csv','data/primary_literature.csv',
                 'data/literature_candidates.csv','data/camphor2022.csv','results/candidate_camphor2022.csv']}}
    for split in ['42','2026','117','2027','7','scaffold','23','101','509']:
        seed=None if split=='scaffold' else int(split)
        preds={k:np.zeros(len(y)) for k in ['v9','agreement','agreement_and_gate']}
        for fold,(ix,va) in enumerate(GroupKFold(5,shuffle=seed is not None,random_state=seed).split(tr,groups=scaffold if seed is None else canon)):
            raw,_=fit_predict_v6(tr.iloc[ix],y[ix],tr.iloc[va])
            old=raw.copy();known=np.isfinite(previous[va]);old[known]=previous[va][known]
            guarded=raw.copy();known=np.isfinite(fixed[va]);guarded[known]=fixed[va][known]
            preds['v9'][va]=old;preds['agreement'][va]=guarded
            preds['agreement_and_gate'][va]=candidate_predict(tr.iloc[ix],y[ix],tr.iloc[va],
                                                            guarded,SELECTED_GATE,lookup_override=fixed[va])
            print(split,fold,flush=True)
        report['scores'][split]={k:float(np.mean((p-y)**2)) for k,p in preds.items()}
        if split in ['42','2026','117','2027','7','scaffold']:
            expected=pd.read_csv(output/'baseline_v9/validation.csv')[split].to_numpy()
            np.testing.assert_allclose(preds['v9'],expected,rtol=0,atol=1e-6)
        pd.DataFrame({'target':y,**preds}).to_csv(output/f'validation_{split}.csv',index=False)
        (output/'metrics.json').write_text(json.dumps(report,indent=2))
        print(split,report['scores'][split],flush=True)
    for group,splits in [('mean_group_mse',['42','2026','117','2027','7']),
                         ('additional_mean_mse',['23','101','509'])]:
        report[group]={k:float(np.mean([report['scores'][s][k] for s in splits])) for k in preds}
    report['all_splits_improved']={k:all(v[k]<v['v9'] for v in report['scores'].values())
                                 for k in ['agreement','agreement_and_gate']}
    if report['all_splits_improved']['agreement_and_gate']:
        raw,_=fit_predict_v6(tr,y,te)
        known=ea.prediction.notna().to_numpy();raw[known]=ea.loc[known,'prediction']
        pred=candidate_predict(tr,y,te,raw,SELECTED_GATE,lookup_override=ea.prediction.to_numpy())
        submission=pd.DataFrame({'index':te['index'],'SI VACV':pred})
        assert len(submission)==len(te) and np.isfinite(pred).all() and (pred>=0).all()
        submission.to_csv(output/'submission.csv',index=False)
        report['candidate_submission_sha256']=hashlib.sha256((output/'submission.csv').read_bytes()).hexdigest()
    ta.to_csv(output/'train_audit.csv',index=False);ea.to_csv(output/'test_audit.csv',index=False)
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(report['mean_group_mse'],report['additional_mean_mse'],flush=True)


if __name__=='__main__':
    main()
