"""Reweight V7's existing SMILES/context branches, retaining all retrievals.
Only training-fold fits; all five molecular splits and scaffold are required.
Run: python experiments/branch_calibration.py
"""
import sys,json,itertools,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import train as core
from literature import lookup_predictions
OUT=ROOT/'run_branch_calibration'

def heads(tr,y,va):
    original,m=core.fit_predict_v6(tr,y,va)
    sm=m['smiles_models'];text=np.clip(sm['text'].predict(va.SMILES.to_numpy()),0,10000)
    tree=np.clip(sm['trees'].predict(core.serialization_features(va.SMILES)),0,10000)
    raw=.85*text+.15*tree
    x=pd.DataFrame(core.serialization_features(va.SMILES)).astype(float)
    for col in ['strain','Culture']:x[col]=va[col].fillna('UNKNOWN').astype(str).to_numpy()
    p=m['inactive_classifier'].predict_proba(x)[:,1]
    a=np.clip(m['active_regressor'].predict(core.serialization_features(va.SMILES)),0,5000)
    protected=core.apply_inactivity_evidence(tr,y,va,np.zeros(len(va)))[0]==10000
    check=.5*np.clip((raw-5000)*1.5+5000,0,10000)+.5*(10000*p+(1-p)*a)
    check[protected]=10000
    np.testing.assert_allclose(check,original,rtol=1e-10,atol=1e-6)
    return raw,p,a,protected

def predict(parts,slope,weight,power,lookup):
    raw,p,a,protected=parts
    # Calibrate probability odds; preserve [0,1] bounds.
    t=np.clip(p,1e-12,1-1e-12);odds=(t/(1-t))**power;q=odds/(1+odds)
    v=(1-weight)*np.clip((raw-5000)*slope+5000,0,10000)+weight*(10000*q+(1-q)*a)
    v[protected]=10000;found=np.isfinite(lookup);v[found]=lookup[found]
    return v

def run():
 start=time.perf_counter();OUT.mkdir(exist_ok=True)
 tr=pd.read_csv(core.DATA/'train.csv',usecols=core.ALLOWED+['SI VACV']);y=tr['SI VACV'].to_numpy(float)
 lookup=lookup_predictions(tr).prediction.to_numpy(float);v=pd.read_csv(ROOT/'results/validation.csv');np.testing.assert_allclose(v.target,y)
 canonical=tr.SMILES.map(lambda s:core.Chem.MolToSmiles(core.Chem.MolFromSmiles(s))).to_numpy();scaff=tr.SMILES.map(lambda s:core.MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
 configs={f'slope{s}_context{w}_odds{p}':(s,w,p) for s,w,p in itertools.product([1.,1.5,2.,3.],[.25,.5,.75,1.],[.7,1.,1.5])}
 report=dict(model='V7 branch and probability calibration, fixed literature predictions',scores={},selection_rule='Improve all five molecular splits and scaffold; no source or row-specific edits.',caveat='Hyperparameter search on repeatedly explored CV; not independent holdout.')
 for seed in [42,2026,117,2027,7,None]:
  name=str(seed) if seed is not None else 'scaffold';parts=[np.zeros(len(y)),np.zeros(len(y)),np.zeros(len(y)),np.zeros(len(y),dtype=bool)]
  for ix,va in GroupKFold(5,shuffle=seed is not None,random_state=seed).split(tr,groups=canonical if seed is not None else scaff):
   z=heads(tr.iloc[ix],y[ix],tr.iloc[va]);
   for out,part in zip(parts,z):out[va]=part
  original=predict(parts,1.5,.5,1.,lookup);np.testing.assert_allclose(original,v[name],rtol=1e-10,atol=1e-6)
  rows={'baseline':float(np.mean((original-y)**2))}
  for key,(s,w,p) in configs.items():rows[key]=float(np.mean((predict(parts,s,w,p,lookup)-y)**2))
  report['scores'][name]=rows;np.savez(OUT/f'{name}_heads.npz',raw=parts[0],probability=parts[1],active=parts[2],protected=parts[3])
  print(name,'baseline',rows['baseline'],'best',min((val,key) for key,val in rows.items()),flush=True)
 report['passing']=[key for key in configs if all(rows[key]<rows['baseline'] for rows in report['scores'].values())]
 report['ranking']=[dict(config=key,mean_molecular_mse=float(np.mean([report['scores'][str(s)][key] for s in [42,2026,117,2027,7]]))) for key in report['passing']]
 report['ranking'].sort(key=lambda r:r['mean_molecular_mse']);report['runtime_seconds']=time.perf_counter()-start
 (OUT/'report.json').write_text(json.dumps(report,indent=2));print('passing',json.dumps(report['ranking'][:5]),flush=True)
def confirm():
 OUT.mkdir(exist_ok=True)
 tr=pd.read_csv(core.DATA/'train.csv',usecols=core.ALLOWED+['SI VACV']);y=tr['SI VACV'].to_numpy(float)
 lookup=lookup_predictions(tr).prediction.to_numpy(float)
 groups=tr.SMILES.map(lambda s:core.Chem.MolToSmiles(core.Chem.MolFromSmiles(s))).to_numpy()
 report=dict(config='slope1.5_context0.5_odds1.5',seeds=[13,73,811],scores={},caveat='New split seeds of the same previously explored molecules; not new independent compounds.')
 for seed in report['seeds']:
  baseline=np.zeros(len(y));candidate=np.zeros(len(y))
  for ix,va in GroupKFold(5,shuffle=True,random_state=seed).split(tr,groups=groups):
   parts=heads(tr.iloc[ix],y[ix],tr.iloc[va]);baseline[va]=predict(parts,1.5,.5,1.,lookup[va]);candidate[va]=predict(parts,1.5,.5,1.5,lookup[va])
  rows=dict(baseline=float(np.mean((baseline-y)**2)),candidate=float(np.mean((candidate-y)**2)))
  report['scores'][str(seed)]=rows
  pd.DataFrame({'target':y,'baseline':baseline,'candidate':candidate}).to_csv(OUT/f'confirmation_{seed}.csv',index=False)
  print(seed,rows,flush=True)
 report['all_improve']=all(r['candidate']<r['baseline'] for r in report['scores'].values())
 (OUT/'confirmation.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):
  confirm() if '--confirm' in sys.argv else run()
