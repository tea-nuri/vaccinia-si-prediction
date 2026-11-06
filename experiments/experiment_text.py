"""Raw/canonical SMILES strings and chemical kernels, without metadata."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from scipy import sparse
sys.path.insert(0,str(Path(__file__).parent))
import experiment_v3 as base

OUT=base.ROOT/'run_search/text';OUT.mkdir(parents=True,exist_ok=True)
y=base.y;n=len(y); groups=base.groups;strings=base.strings
results=[]

def evaluate(name,p):
 score=float(np.mean((y-p)**2)); results.append(dict(name=name,mse42=score))
 joblib.dump(p,OUT/(name+'.joblib'))
 print(name,round(score),flush=True)
 (OUT/'results.json').write_text(json.dumps(sorted(results,key=lambda x:x['mse42']),indent=2))

# Fold-fitted vocabularies, including the original SMILES serialization.
splits=list(GroupKFold(5,shuffle=True,random_state=42).split(strings[:n],y,groups[:n]))
for canon in [False,True]:
 for ng in [(2,4),(3,8),(1,3)]:
  x=groups if canon else strings
  matrices=[]
  for ix,va in splits:
   v=TfidfVectorizer(analyzer='char',ngram_range=ng,lowercase=False,sublinear_tf=True)
   matrices.append((ix,va,v.fit_transform(x[ix]),v.transform(x[va])))
  for a in [.03,.3,3,30]:
   for mode in ['cap','spike']:
    name=f'text_c{canon}_n{ng[0]}{ng[1]}_{mode}_a{a}'
    p=np.zeros(n)
    for ix,va,xt,xv in matrices:
     m=LogisticRegression(C=1/a,max_iter=1200) if mode=='spike' else Ridge(alpha=a)
     target=(y[ix]==10000).astype(int) if mode=='spike' else np.minimum(y[ix],10000)
     m.fit(xt,target)
     p[va]=10000*m.predict_proba(xv)[:,1] if mode=='spike' else np.clip(m.predict(xv),0,10000)
    evaluate(name,p)

# Tanimoto kernels with local similarity weighting and target clipping.
for space in ['morgan1_bit','morgan2_bit','morgan3_bit','morgan2_count','rdk','maccs']:
 x=base.spaces[space][:n].astype(float)
 dot=x@x.T;s=(x*x).sum(1);k=dot/np.maximum(s[:,None]+s-dot,1e-9)
 for power in [1,2,4]:
  for a in [.03,.3,3]:
   p=np.zeros(n)
   for ix,va in splits:
    m=KernelRidge(kernel='precomputed',alpha=a).fit(k[np.ix_(ix,ix)]**power,np.minimum(y[ix],10000))
    p[va]=np.clip(m.predict(k[np.ix_(va,ix)]**power),0,10000)
   evaluate(f'krr_{space}_p{power}_a{a}',p)

# Serialization counts describe only the supplied SMILES, without other columns.
chars=list('cCnNoOpPsSFIBr[]=#@/\\123456789%()+-.')
style=np.array([[len(s)]+[s.count(c) for c in chars]+[s.find('c'),s.find('@'),s.count('C1'),s.count('c1')] for s in strings])
for sp in ['style','desc_style']:
 x=style if sp=='style' else np.hstack([np.nan_to_num(base.spaces['desc']),style])
 for leaf in [1,3,6]:
  p=np.zeros(n)
  for ix,va in splits:
   m=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=leaf,max_features=.7,n_jobs=4,random_state=42).fit(x[ix],np.minimum(y[ix],10000))
   p[va]=np.clip(m.predict(x[va]),0,10000)
  evaluate(f'et_{sp}_l{leaf}',p)
