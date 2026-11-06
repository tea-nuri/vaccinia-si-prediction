"""SMILES-only model search; identical canonical molecules never cross folds."""
from pathlib import Path
import json, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT
OUT=ROOT/'run_search/v3'; OUT.mkdir(parents=True,exist_ok=True)
def load_spaces():
    cache=ROOT/'run_search/features.joblib'
    if cache.exists(): return joblib.load(cache)
    sys.path.insert(0,str(ROOT))
    from features import features
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
    smiles=pd.concat([tr.SMILES,te.SMILES],ignore_index=True)
    fp,desc,canonical,scaffolds=features(smiles)
    spaces={'desc':desc,'morgan2':fp}
    mols=[Chem.MolFromSmiles(s) for s in smiles]
    for radius in [1,2,3]:
        gen=rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=2048)
        for count in [False,True]:
            name=f'morgan{radius}_{"count" if count else "bit"}'
            spaces[name]=np.array([gen.GetCountFingerprintAsNumPy(m) if count else gen.GetFingerprintAsNumPy(m) for m in mols],dtype=float)
    spaces['rdk']=np.array([list(Chem.RDKFingerprint(m,fpSize=2048)) for m in mols],dtype=float)
    spaces['maccs']=np.array([list(MACCSkeys.GenMACCSKeys(m)) for m in mols],dtype=float)
    cache.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump((spaces,canonical,scaffolds),cache)
    return spaces,canonical,scaffolds

tr=pd.read_csv(DATA/'train.csv'); te=pd.read_csv(DATA/'test.csv')
y=tr['SI VACV'].to_numpy(); n=len(y)
spaces, groups, scaffolds=load_spaces()
groups=np.asarray(groups); scaffolds=np.asarray(scaffolds)
for k in spaces:
    spaces[k]=np.asarray(spaces[k],dtype=np.float32)
spaces['combined']=np.hstack([spaces['desc'], spaces['morgan2_count']])
strings=np.asarray(pd.concat([tr.SMILES,te.SMILES]))
configs=[]
for sp in ['desc','combined','morgan2_count','maccs']:
 for mode in ['raw','cap','spike']:
  if sp=='maccs' and mode=='raw': continue
  for leaf in [1,3]:
   configs.append(dict(name=f'et_{sp}_{mode}_{leaf}',family='et',space=sp,mode=mode,leaf=leaf))
for sp in ['desc','combined']:
 for mode in ['raw','cap','spike']:
  for depth in [3,5,7]:
   configs.append(dict(name=f'cat_{sp}_{mode}_{depth}',family='cat',space=sp,mode=mode,depth=depth))
for mode in ['cap','spike']:
 for depth in [2,4]:
  configs.append(dict(name=f'xgb_desc_{mode}_{depth}',family='xgb',space='desc',mode=mode,depth=depth))
for canonical in [False,True]:
 for mode in ['cap','spike']:
  for alpha in [.1,1.,10.]:
   configs.append(dict(name=f'text_{canonical}_{mode}_{alpha}',family='text',space='text',mode=mode,canonical=canonical,alpha=alpha))

def model(c):
 cl=c['mode']=='spike'
 if c['family']=='et':
  cls=ExtraTreesClassifier if cl else ExtraTreesRegressor
  return make_pipeline(SimpleImputer(keep_empty_features=True),cls(n_estimators=400,min_samples_leaf=c['leaf'],max_features=.5,n_jobs=4,random_state=42))
 if c['family']=='cat':
  cls=CatBoostClassifier if cl else CatBoostRegressor
  return cls(iterations=600,depth=c['depth'],learning_rate=.035,l2_leaf_reg=5,verbose=False,thread_count=4,random_seed=42,allow_writing_files=False)
 if c['family']=='xgb':
  cls=XGBClassifier if cl else XGBRegressor
  return cls(n_estimators=400,max_depth=c['depth'],learning_rate=.025,min_child_weight=3,reg_lambda=10,subsample=.9,colsample_bytree=.8,n_jobs=4,random_state=42)
 if c['family']=='text':
  cls=LogisticRegression(C=c['alpha'],max_iter=1000) if cl else Ridge(alpha=c['alpha'])
  return make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(2,6),lowercase=False,sublinear_tf=True),cls)

# Non-spike predictions: fitted using ONLY the non-spike training rows in each fold.
def low_predictions(ix, va):
 good=ix[y[ix]!=10000]
 m=ExtraTreesRegressor(n_estimators=350,max_features=.5,min_samples_leaf=3,n_jobs=4,random_state=42)
 m=make_pipeline(SimpleImputer(keep_empty_features=True),m)
 m.fit(spaces['desc'][good], np.minimum(y[good],5000))
 return np.maximum(m.predict(spaces['desc'][va]),0)

def run(c,seed):
 if c['space']=='text': x=groups if c['canonical'] else strings
 else: x=np.nan_to_num(spaces[c['space']],nan=0,posinf=0,neginf=0)
 oof=np.zeros(n); pt=[]
 splits=list(GroupKFold(5,shuffle=True,random_state=seed).split(x[:n],y,groups[:n]))
 for ix,va in splits:
  m=model(c)
  target=(y==10000).astype(int) if c['mode']=='spike' else np.minimum(y,10000) if c['mode']=='cap' else y
  m.fit(x[ix],target[ix])
  if c['mode']=='spike':
   p=m.predict_proba(x[va])[:,1]
   oof[va]=10000*p+(1-p)*low_predictions(ix,va)
  else: oof[va]=np.maximum(m.predict(x[va]),0)
 return oof

def main():
 results=[]; preds={}
 for c in configs:
  pth=OUT/(c['name']+'.joblib')
  if pth.exists(): p=joblib.load(pth)
  else:
   p=run(c,42); joblib.dump(p,pth)
  mse=float(np.mean((y-p)**2)); preds[c['name']]=p
  results.append(dict(**c,mse42=mse,mse42_without_outlier=float(np.mean((y[y<20000]-p[y<20000])**2))))
  print(c['name'],round(mse),flush=True)
  (OUT/'results.json').write_text(json.dumps(sorted(results,key=lambda r:r['mse42']),indent=2))
 top=sorted(results,key=lambda r:r['mse42'])[:8]
 for c in top:
  for seed in [2026,117]:
   pth=OUT/(c['name']+f'_{seed}.joblib')
   if pth.exists(): p=joblib.load(pth)
   else: p=run(c,seed); joblib.dump(p,pth)
   c[f'mse{seed}']=float(np.mean((y-p)**2))
  c['mean']=float(np.mean([c['mse42'],c['mse2026'],c['mse117']]))
  print('CONFIRM',c,flush=True)
  (OUT/'finalists.json').write_text(json.dumps(sorted(top,key=lambda r:r.get('mean',1e20)),indent=2))

if __name__=='__main__': main()
