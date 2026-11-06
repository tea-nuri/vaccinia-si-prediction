"""Context normalization ablation. Run from repository root. Artifacts: run_context/."""
import re,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import train as core
from literature import apply_literature

def clean(t,mode):
 t=t[core.ALLOWED].copy()
 for c in ['strain','Culture']:
  t[c]=t[c].fillna('UNKNOWN').map(lambda s:re.sub('[^a-z0-9]','',str(s).lower()))
 if mode=='biological':
  def strain(s):
   for w in ['copenhagen','tiantan','lister','livp','ihdj','wr','westernreserve']:
    if w in s:return 'wr' if w=='westernreserve' else w
   return s
  t.strain=t.strain.map(strain)
  t.Culture=t.Culture.str.replace('cells','',regex=False).str.replace('cell','',regex=False)
 return t
tr=pd.read_csv(core.DATA/'train.csv',usecols=core.ALLOWED+['SI VACV']);y=tr['SI VACV'].to_numpy(float)
baseline=pd.read_csv('results/validation.csv');canonical=tr.SMILES.map(lambda s:core.Chem.MolToSmiles(core.Chem.MolFromSmiles(s))).to_numpy();scaff=tr.SMILES.map(lambda s:core.MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
report={}
Path('run_context').mkdir(exist_ok=True)
with threadpool_limits(limits=1):
 for seed in [42,2026,None]:
  name=str(seed) if seed is not None else 'scaffold';report[name]={'baseline':float(np.mean((baseline[name]-y)**2))}
  for mode in ['normalized','biological']:
   norm=clean(tr,mode);p=np.zeros(len(y))
   for ix,va in GroupKFold(5,shuffle=seed is not None,random_state=seed).split(tr,groups=canonical if seed is not None else scaff):
    q,_=core.fit_predict_v6(norm.iloc[ix],y[ix],norm.iloc[va]);p[va],_=apply_literature(tr.iloc[va],q)
   for w in [.25,.5,1.]:report[name][mode+'_'+str(w)]=float(np.mean(((1-w)*baseline[name].to_numpy()+w*p-y)**2))
   np.save('run_context/'+name+'_'+mode+'.npy',p)
  print(name,json.dumps(report[name]),flush=True)
Path('run_context/report.json').write_text(json.dumps(report,indent=2))
