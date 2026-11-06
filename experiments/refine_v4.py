from pathlib import Path
import sys,json,joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import experiment_v3 as base

OUT=base.ROOT/'run_search/v4';OUT.mkdir(parents=True,exist_ok=True)
y=base.y;n=len(y)
chars=list('cCnNoOpPsSFIBr[]=#@/\\123456789%()+-.')
style=np.array([[len(s)]+[s.count(c) for c in chars]+[s.find('c'),s.find('@'),s.count('C1'),s.count('c1')] for s in base.strings])
xs={'style':style,'desc_style':np.hstack([np.nan_to_num(base.spaces['desc']),style])}
configs=[]
for ng in [(2,4),(3,8),(2,6)]:
 for a in [.01,.03,.1,.3]:
  configs.append(dict(name=f'text_{ng[0]}{ng[1]}_{a}',kind='text',ng=ng,alpha=a,cap=10000))
configs.append(dict(name='text_24_raw',kind='text',ng=(2,4),alpha=.03,cap=1e10))
for sp in xs:
 for frac in [.5,1.]:
  configs.append(dict(name=f'et_{sp}_{frac}',kind='et',space=sp,frac=frac,cap=10000))
 for depth in [3,5,7]:
  configs.append(dict(name=f'cat_{sp}_{depth}',kind='cat',space=sp,depth=depth,cap=10000))

def get_model(c):
 if c['kind']=='text':
  return make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=tuple(c['ng']),lowercase=False,sublinear_tf=True),Ridge(alpha=c['alpha']))
 if c['kind']=='et':
  return ExtraTreesRegressor(n_estimators=500,max_features=c['frac'],min_samples_leaf=1,n_jobs=4,random_state=42)
 return CatBoostRegressor(iterations=800,depth=c['depth'],learning_rate=.035,l2_leaf_reg=5,verbose=False,thread_count=4,random_seed=42,allow_writing_files=False)

def run(c,seed):
 x=base.strings if c['kind']=='text' else xs[c['space']]
 p=np.zeros(n)
 for ix,va in GroupKFold(5,shuffle=True,random_state=seed).split(x[:n],y,base.groups[:n]):
  m=get_model(c).fit(x[ix],np.minimum(y[ix],c['cap']))
  p[va]=np.clip(m.predict(x[va]),0,c['cap'])
 return p

def main():
 results=[]
 for c in configs:
  row=dict(c)
  for seed in [42,2026,117]:
   path=OUT/f'{c["name"]}_{seed}.joblib'
   if path.exists(): p=joblib.load(path)
   else: p=run(c,seed);joblib.dump(p,path)
   row[str(seed)]=float(np.mean((y-p)**2))
  row['mean']=float(np.mean([row[str(s)] for s in [42,2026,117]]))
  results.append(row);print(json.dumps(row),flush=True)
  (OUT/'results.json').write_text(json.dumps(sorted(results,key=lambda r:r['mean']),indent=2))

if __name__=='__main__':main()
