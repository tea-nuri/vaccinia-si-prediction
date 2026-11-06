"""Test whether physicochemical descriptors improve V7 inactivity prediction.

Fixed V7 literature and active-regression predictions; descriptors use only
SMILES. Candidate selection requires all five molecule splits AND scaffold.
"""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from rdkit.Chem import Descriptors
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
import train as core
from literature import lookup_predictions
OUT=ROOT/'run_descriptor_classifier'

def run():
 start=time.perf_counter();OUT.mkdir(exist_ok=True)
 tr=pd.read_csv(core.DATA/'train.csv',usecols=core.ALLOWED+['SI VACV']);y=tr['SI VACV'].to_numpy(float);baseline=pd.read_csv(ROOT/'results/validation.csv')
 lookup=lookup_predictions(tr).prediction.to_numpy(float);found=np.isfinite(lookup)
 mols=[core.Chem.MolFromSmiles(s) for s in tr.SMILES]
 desc=np.asarray([[f(m) for n,f in Descriptors._descList] for m in mols],float)
 desc=np.nan_to_num(desc,nan=0.,posinf=1e6,neginf=-1e6)
 serialization=core.serialization_features(tr.SMILES)
 canonical=np.array([core.Chem.MolToSmiles(m) for m in mols]);scaff=np.array([core.MurckoScaffold.MurckoScaffoldSmiles(mol=m) or 'ACYCLIC' for m in mols])
 configs={f'{mode}_depth{d}':(mode,d) for mode in ['serialization','descriptors'] for d in [3,4,6]}
 pending=set(configs)
 report=dict(scores={},selection_rule='Require MSE improvement in every molecule split and scaffold; stop failing variants.',caveat='Repeated CV and hyperparameter selection; no independent holdout.')
 for seed in [42,2026,117,2027,7,None]:
  label=str(seed) if seed is not None else 'scaffold'
  parts=np.load(ROOT/'run_branch_calibration'/f'{label}_heads.npz');raw=parts['raw'];active=parts['active'];protected=parts['protected']
  text=np.clip((raw-5000)*1.5+5000,0,10000)
  pred={k:np.zeros(len(y)) for k in pending}
  for ix,va in GroupKFold(5,shuffle=seed is not None,random_state=seed).split(tr,groups=canonical if seed is not None else scaff):
   for name in list(pending):
    mode,depth=configs[name]
    x=pd.DataFrame(serialization if mode=='serialization' else np.c_[serialization,desc]).astype(float)
    cats=[]
    for col in ['strain','Culture']:cats.append(x.shape[1]);x[col]=tr[col].fillna('UNKNOWN').astype(str).to_numpy()
    model=CatBoostClassifier(iterations=700,depth=depth,learning_rate=.035,l2_leaf_reg=5,verbose=False,thread_count=4,allow_writing_files=False,random_seed=42)
    model.fit(x.iloc[ix],(y[ix]==10000).astype(int),cat_features=cats)
    pred[name][va]=model.predict_proba(x.iloc[va])[:,1]
  rows={'baseline':float(np.mean((baseline[label]-y)**2))}
  survived=set()
  for name,p in pred.items():
   for power in [1.,1.5]:
    t=np.clip(p,1e-12,1-1e-12);odds=(t/(1-t))**power;q=odds/(1+odds)
    proposal=.5*text+.5*(10000*q+(1-q)*active);proposal[protected]=10000;proposal[found]=lookup[found]
    for w in [.25,.5,1.]:
     key=name+f'_odds{power}_blend{w}';candidate=(1-w)*baseline[label].to_numpy()+w*proposal
     rows[key]=float(np.mean((candidate-y)**2))
     if rows[key]<rows['baseline'] and all(key in previous and previous[key]<previous['baseline'] for previous in report['scores'].values()):survived.add(name)
   np.save(OUT/f'{label}_{name}.npy',p)
  report['scores'][label]=rows;pending=survived
  print(label,'baseline',rows['baseline'],'best',min((v,k) for k,v in rows.items()),'survivors',sorted(pending),flush=True)
  if not pending:break
 allkeys=set.intersection(*(set(r) for r in report['scores'].values()))
 report['passing']=[k for k in allkeys if k!='baseline' and len(report['scores'])==6 and all(r[k]<r['baseline'] for r in report['scores'].values())]
 report['runtime_seconds']=time.perf_counter()-start
 (OUT/'report.json').write_text(json.dumps(report,indent=2));print('passing',report['passing'],flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):run()
