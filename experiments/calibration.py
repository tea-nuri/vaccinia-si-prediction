"""Exploratory V7 calibration search; outputs stay in ignored run_tune_v8/."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from train import fit_predict_v6,serialization_features,apply_inactivity_evidence,ALLOWED
from literature import lookup_predictions
(root/'run_tune_v8').mkdir(exist_ok=True)
tr=pd.read_csv(root/'data/train.csv',usecols=ALLOWED+['SI VACV']);y=tr['SI VACV'].to_numpy();lookup=lookup_predictions(tr).prediction.to_numpy();matched=np.isfinite(lookup);groups=tr.SMILES.map(lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s))).to_numpy();scaf=tr.SMILES.map(lambda s:MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy();results={};allpred={};baselines={}
for seed in [42,2026,117,None]:
 label=str(seed) if seed is not None else 'scaffold';pp={};base=np.zeros(len(y))
 for ix,va in GroupKFold(5,shuffle=seed is not None,random_state=seed).split(tr,groups=groups if seed is not None else scaf):
  t=tr.iloc[ix];v=tr.iloc[va];p,m=fit_predict_v6(t,y[ix],v);p[matched[va]]=lookup[va][matched[va]];base[va]=p
  xt=pd.DataFrame(serialization_features(t.SMILES)).astype(float);xv=pd.DataFrame(serialization_features(v.SMILES)).astype(float)
  for c in ['strain','Culture']:xt[c]=t[c].fillna('UNKNOWN').astype(str).to_numpy();xv[c]=v[c].fillna('UNKNOWN').astype(str).to_numpy()
  prob=m['inactive_classifier'].predict_proba(xv)[:,1];ap=np.clip(m['active_regressor'].predict(serialization_features(v.SMILES)),0,5000)
  tree=np.clip(m['smiles_models']['trees'].predict(serialization_features(v.SMILES)),0,10000)
  vector=TfidfVectorizer(analyzer='char',ngram_range=(2,6),lowercase=False,sublinear_tf=True);a=vector.fit_transform(t.SMILES);b=vector.transform(v.SMILES)
  protected=apply_inactivity_evidence(t,y[ix],v,np.zeros(len(va)))[0]==10000
  for alpha in [.003,.03,.3]:
   txt=np.clip(Ridge(alpha=alpha).fit(a,np.minimum(y[ix],10000)).predict(b),0,10000)
   for slope in [1.,1.5,2.,3.]:
    sp=np.clip((.85*txt+.15*tree-5000)*slope+5000,0,10000)
    for power in [1.,1.5,2.]:
     # Symmetric odds sharpening; preserves probability ordering.
     pr=prob**power/(prob**power+(1-prob)**power);cp=10000*pr+(1-pr)*ap
     for weight in [0.,.25,.5,.75,1.]:
      key=f'a{alpha}_s{slope}_p{power}_w{weight}';out=(1-weight)*sp+weight*cp
      out[protected]=10000;out[matched[va]]=lookup[va][matched[va]]
      pp.setdefault(key,np.zeros(len(y)))[va]=out
 for key,p in pp.items():results.setdefault(key,{})[label]=float(np.mean((y-p)**2));allpred[key+'_'+label]=p
 baselines[label]=float(np.mean((y-base)**2));print(label,baselines[label],min((np.mean((y-p)**2),k) for k,p in pp.items()),flush=True)
report={'baseline':baselines,'variants':results,'settings':'3 ridge alphas x 4 SMILES slopes x 3 probability powers x 5 component weights; fixed audited literature lookup; GroupKFold; original targets'}
passing=[k for k,s in results.items() if all(s[label]<baselines[label] for label in baselines)]
report['passing']=passing;report['selected']=min(passing,key=lambda k:np.mean([results[k][s] for s in ['42','2026','117']])) if passing else None
print('SELECTED',report['selected'],results.get(report['selected']),flush=True)
(root/'run_tune_v8/search.json').write_text(json.dumps(report,indent=2));np.savez(root/'run_tune_v8/oof.npz',**allpred)
