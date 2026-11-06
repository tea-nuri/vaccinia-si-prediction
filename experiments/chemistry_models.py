"""Local V7 comparison with chemistry features; never submits to Kaggle.

Run: OPENBLAS_NUM_THREADS=1 python experiments/chemistry_models.py
The >50% gate must pass every tested protocol before considering submission.
"""
import sys,json,re
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors,rdFingerprintGenerator
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from catboost import CatBoostRegressor,CatBoostClassifier
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from train import ALLOWED,serialization_features,apply_inactivity_evidence
from literature import lookup_predictions
(root/'run_promising').mkdir(exist_ok=True)
tr=pd.read_csv(root/'data/train.csv',usecols=ALLOWED+['SI VACV']);y=tr['SI VACV'].to_numpy();base=pd.read_csv(root/'results/validation.csv');lit=lookup_predictions(tr).prediction.to_numpy();mask=np.isfinite(lit);mols=[Chem.MolFromSmiles(s) for s in tr.SMILES];canon=np.array([Chem.MolToSmiles(m) for m in mols]);desc=np.array([[f(m) for _,f in Descriptors._descList] for m in mols]);desc=np.nan_to_num(desc,nan=0,posinf=0,neginf=0);gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=512,includeChirality=True);fp=np.array([gen.GetFingerprintAsNumPy(m) for m in mols]);features=pd.DataFrame(np.c_[serialization_features(tr.SMILES),desc,fp]).clip(-1e6,1e6)
for c in ['strain','Culture']:
 features[c]=tr[c].fillna('UNKNOWN').astype(str).map(lambda s:re.sub('[^a-z0-9]','',s.lower().replace('cells','')))
cat=[features.shape[1]-2,features.shape[1]-1];out={};predictions={}
for seed in [42,2026,117,None]:
 from rdkit.Chem.Scaffolds import MurckoScaffold
 label=str(seed) if seed is not None else 'scaffold';groups=canon if seed is not None else np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m) or 'ACYCLIC' for m in mols]);pp={}
 for ix,va in GroupKFold(5,shuffle=seed is not None,random_state=seed).split(tr,groups=groups):
  a,b=features.iloc[ix],features.iloc[va]
  for depth in [3,5,7]:
   for mode in ['direct','two_stage']:
    key=f'cat_d{depth}_{mode}';params=dict(iterations=800,depth=depth,learning_rate=.025,l2_leaf_reg=10,verbose=False,thread_count=4,random_seed=2026,allow_writing_files=False)
    if mode=='direct':model=CatBoostRegressor(**params).fit(a,np.minimum(y[ix],10000),cat_features=cat);p=np.clip(model.predict(b),0,10000)
    else:
     model=CatBoostClassifier(**params).fit(a,(y[ix]==10000).astype(int),cat_features=cat);prob=model.predict_proba(b)[:,1];active=y[ix]!=10000;reg=CatBoostRegressor(**params).fit(a.iloc[np.flatnonzero(active)],np.log1p(np.minimum(y[ix][active],5000)),cat_features=cat);p=10000*prob+(1-prob)*np.clip(np.expm1(reg.predict(b)),0,5000)
    p,_=apply_inactivity_evidence(tr.iloc[ix],y[ix],tr.iloc[va],p);p[mask[va]]=lit[va][mask[va]];pp.setdefault(key,np.zeros(len(y)))[va]=p
  for leaf in [1,3]:
   a=features.iloc[ix,:-2];b=features.iloc[va,:-2];model=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=leaf,max_features=.5,n_jobs=4,random_state=2026).fit(a,np.minimum(y[ix],10000));p=np.clip(model.predict(b),0,10000);p,_=apply_inactivity_evidence(tr.iloc[ix],y[ix],tr.iloc[va],p);p[mask[va]]=lit[va][mask[va]];pp.setdefault(f'extratrees_leaf{leaf}',np.zeros(len(y)))[va]=p
 for k,p in pp.items():
  for w in [.25,.5,1.]:
   pred=(1-w)*base[label].to_numpy()+w*p;name=k+f'_blend{w}';out.setdefault(name,{})[label]=float(np.mean((y-pred)**2));predictions[name+'_'+label]=pred
 print(label,'best',min((np.mean((y-p)**2),k) for k,p in pp.items()),'baseline',np.mean((y-base[label])**2),flush=True)
report={'baseline':{s:float(np.mean((y-base[s])**2)) for s in ['42','2026','117','scaffold']},'scores':out,'settings':'Raw serialization + RDKit descriptors + 512-bit chiral Morgan r2; normalized strain/culture. CatBoost d3/5/7 800 lr.025 l2=10 seed2026: direct cap10000 or inactive classifier + log1p active regression cap5000. ExtraTrees 500 max_features.5 leaf1/3. Blend .25/.5/1. All preprocessing target-independent; train fitted within folds. V7 fixed retrieval.'}
report['passes_submission_gate']=[k for k,v in out.items() if all(v[s]<.5*report['baseline'][s] for s in v)]
(root/'run_promising/models.json').write_text(json.dumps(report,indent=2));np.savez(root/'run_promising/predictions.npz',**predictions)
print('50% gate',report['passes_submission_gate'],flush=True)
