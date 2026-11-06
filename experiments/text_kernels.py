from pathlib import Path
import numpy as np,pandas as pd,json,joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'run_search/kernels';OUT.mkdir(parents=True,exist_ok=True)
tr=pd.read_csv(ROOT/'train.csv',usecols=['SMILES','SI VACV'])
x=tr.SMILES.to_numpy();y=tr['SI VACV'].to_numpy();groups=[Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in x]
results=[]
for ng in [(2,4),(2,6),(3,8)]:
 for mode in ['tfidf','binary']:
  preds={}
  for seed in [42,2026,117]:
   for ix,va in GroupKFold(5,shuffle=True,random_state=seed).split(x,y,groups):
    v=TfidfVectorizer(analyzer='char',ngram_range=ng,lowercase=False,sublinear_tf=mode=='tfidf',binary=mode=='binary')
    xt=v.fit_transform(x[ix]);xv=v.transform(x[va]);ktr=(xt@xt.T).toarray();kva=(xv@xt.T).toarray()
    for power in [1,2,3,5]:
     for a in [.01,.03,.1,.3]:
      name=f'{ng[0]}{ng[1]}_{mode}_p{power}_a{a}'
      if name not in preds:preds[name]={s:np.zeros(len(y)) for s in [42,2026,117]}
      m=KernelRidge(alpha=a,kernel='precomputed').fit(ktr**power,np.minimum(y[ix],10000))
      preds[name][seed][va]=np.clip(m.predict(kva**power),0,10000)
  for name,ps in preds.items():
   r={'name':name,'ng':list(ng),'mode':mode,'scores':{str(s):float(np.mean((y-p)**2)) for s,p in ps.items()}}
   r['mean']=float(np.mean(list(r['scores'].values())))
   results.append(r);joblib.dump(ps,OUT/f'{name}.joblib')
  (OUT/'results.json').write_text(json.dumps(sorted(results,key=lambda r:r['mean']),indent=2))
  print('DONE',ng,mode,'TOP',sorted(results,key=lambda r:r['mean'])[:3],flush=True)
