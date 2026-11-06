from pathlib import Path
import numpy as np,pandas as pd,json,joblib
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import experiment_v3 as base
OUT=base.ROOT/'run_search/text_trees';OUT.mkdir(parents=True,exist_ok=True)
x=base.strings[:base.n];y=base.y
configs=[]
for ng in [(1,3),(2,6)]:
 for kind in ['cat','xgb','et']:
  for depth in ([3,5,7] if kind!='et' else [1,3]):
   configs.append(dict(name=f'{kind}_{ng[0]}{ng[1]}_{depth}',kind=kind,ng=ng,depth=depth))

def run(c,seed):
 p=np.zeros(len(y))
 for ix,va in GroupKFold(5,shuffle=True,random_state=seed).split(x,y,base.groups[:len(y)]):
  v=CountVectorizer(analyzer='char',ngram_range=c['ng'],lowercase=False,min_df=2,max_features=2000)
  xt=v.fit_transform(x[ix]).toarray(); xv=v.transform(x[va]).toarray()
  if c['kind']=='cat':m=CatBoostRegressor(iterations=700,depth=c['depth'],learning_rate=.035,l2_leaf_reg=5,thread_count=4,verbose=False,random_seed=42,allow_writing_files=False)
  elif c['kind']=='xgb':m=XGBRegressor(n_estimators=600,max_depth=c['depth'],learning_rate=.025,min_child_weight=3,reg_lambda=10,colsample_bytree=.8,n_jobs=4,random_state=42)
  else:m=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=c['depth'],max_features=.5,n_jobs=4,random_state=42)
  m.fit(xt,np.minimum(y[ix],10000));p[va]=np.clip(m.predict(xv),0,10000)
 return p

def main():
 r=[]
 for c in configs:
  p=run(c,42);joblib.dump(p,OUT/(c['name']+'_42.joblib'))
  r.append(dict(**c,mse42=float(np.mean((y-p)**2))))
  print(c['name'],r[-1]['mse42'],flush=True)
  (OUT/'results.json').write_text(json.dumps(sorted(r,key=lambda z:z['mse42']),indent=2))
 for c in sorted(r,key=lambda z:z['mse42'])[:4]:
  for seed in [2026,117]:
   p=run(c,seed);joblib.dump(p,OUT/(c['name']+f'_{seed}.joblib'));c[f'mse{seed}']=float(np.mean((y-p)**2))
  c['mean']=float(np.mean([c[f'mse{s}'] for s in [42,2026,117]]))
  print('CONFIRMED',c,flush=True)
  (OUT/'results.json').write_text(json.dumps(sorted(r,key=lambda z:z.get('mean',1e20)),indent=2))

if __name__=='__main__':main()
