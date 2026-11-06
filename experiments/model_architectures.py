"""Local architecture comparison; never changes selected results or submits.

All fitted feature transforms stay inside each grouped fold. Fixed literature
retrieval is identical across candidates. Seeds 42/2026 plus scaffolds select
candidates; 117/2027/7 are additional confirmation splits, not fresh data.
"""
import sys,json,hashlib,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train import ALLOWED,serialization_features,fit_predict_v6,apply_inactivity_evidence
from literature import lookup_predictions,structure_key
from rdkit.Chem.Scaffolds import MurckoScaffold
OUT=ROOT/'run_architectures';OUT.mkdir(exist_ok=True)

def main():
    tr=pd.read_csv(ROOT/'data/train.csv',usecols=ALLOWED+['SI VACV']);y=tr['SI VACV'].to_numpy(float)
    ref=pd.read_csv(ROOT/'results/validation.csv');np.testing.assert_allclose(ref.target,y)
    lookup=lookup_predictions(tr).prediction.to_numpy();matched=np.isfinite(lookup)
    mols=[Chem.MolFromSmiles(s) for s in tr.SMILES]
    canonical=np.array([structure_key(s) for s in tr.SMILES]);scaffolds=np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m) or 'ACYCLIC' for m in mols])
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024,includeChirality=True)
    fp=np.array([gen.GetFingerprintAsNumPy(m) for m in mols],dtype=float)
    dot=fp@fp.T;norm=fp.sum(1);tanimoto=dot/np.maximum(norm[:,None]+norm[None,:]-dot,1)
    context=tr[['strain','Culture']].fillna('UNKNOWN').astype(str)
    context_kernel=sum((context[c].to_numpy()[:,None]==context[c].to_numpy()[None,:]).astype(float) for c in context)/2
    spec=[('mlp_reg_32',(32,),1.,'reg'),('mlp_reg_64_32',(64,32),1.,'reg'),('mlp_reg_128_64',(128,64),10.,'reg'),('mlp_class_64_32',(64,32),1.,'class')]
    names=[s[0] for s in spec]+['kernel_reg_01','kernel_reg_1','kernel_class_01','kernel_class_1']
    report={'protocols':{},'settings':{'mlp_features':'Fold-local char TF-IDF SVD32 + scaled raw SMILES counts + one-hot strain/Culture; no forbidden columns','mlp_solver':'lbfgs, max_iter=400, random_state=42, relu','architectures':spec,'kernel':'0.4 Morgan Tanimoto + 0.4 fold-local char TF-IDF cosine + 0.2 context equality','blends':[.25,.5,1.],'selection':'Improve both discovery molecule splits 42/2026 and scaffold, then all confirmation splits 117/2027/7','caveat':'Repeated data and searched candidates; confirmation splits reuse molecules and are not independent holdouts.'},'convergence_warnings':{},'source_sha256':{n:hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest() for n in ['train.csv','primary_literature.csv','external.csv']}}
    for seed in [42,2026,None,117,2027,7]:
        label=str(seed) if seed is not None else 'scaffold';groups=canonical if seed is not None else scaffolds
        preds={n:np.zeros(len(y)) for n in names}
        for fold,(ix,va) in enumerate(GroupKFold(5,shuffle=seed is not None,random_state=seed).split(tr,groups=groups)):
            assert not set(canonical[ix])&set(canonical[va])
            _,base=fit_predict_v6(tr.iloc[ix],y[ix],tr.iloc[va])
            act=np.clip(base['active_regressor'].predict(serialization_features(tr.iloc[va].SMILES)),0,5000)
            tf=TfidfVectorizer(analyzer='char',ngram_range=(2,6),lowercase=False,sublinear_tf=True)
            ta=tf.fit_transform(tr.iloc[ix].SMILES);tb=tf.transform(tr.iloc[va].SMILES)
            svd=TruncatedSVD(n_components=32,random_state=42);sa=svd.fit_transform(ta);sb=svd.transform(tb)
            scale=StandardScaler();ra=scale.fit_transform(serialization_features(tr.iloc[ix].SMILES));rb=scale.transform(serialization_features(tr.iloc[va].SMILES))
            enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False);ca=enc.fit_transform(context.iloc[ix]);cb=enc.transform(context.iloc[va])
            fa=np.c_[sa,ra,ca];fb=np.c_[sb,rb,cb]
            for name,layers,alpha,kind in spec:
                cls=MLPRegressor if kind=='reg' else MLPClassifier
                model=cls(hidden_layer_sizes=layers,alpha=alpha,solver='lbfgs',max_iter=400,random_state=42)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    model.fit(fa,np.minimum(y[ix],10000)/10000 if kind=='reg' else (y[ix]==10000).astype(int))
                report['convergence_warnings'][name]=report['convergence_warnings'].get(name,0)+sum(isinstance(w.message,ConvergenceWarning) for w in caught)
                if kind=='reg':p=np.clip(10000*model.predict(fb),0,10000)
                else:
                    prob=model.predict_proba(fb)[:,1];p=10000*prob+(1-prob)*act
                preds[name][va]=apply_inactivity_evidence(tr.iloc[ix],y[ix],tr.iloc[va],p)[0]
            ka=.4*tanimoto[np.ix_(ix,ix)]+.4*(ta@ta.T).toarray()+.2*context_kernel[np.ix_(ix,ix)]
            kb=.4*tanimoto[np.ix_(va,ix)]+.4*(tb@ta.T).toarray()+.2*context_kernel[np.ix_(va,ix)]
            for kind in ['reg','class']:
                for alpha,tag in [(.1,'01'),(1.,'1')]:
                    p=KernelRidge(alpha=alpha,kernel='precomputed').fit(ka,np.minimum(y[ix],10000)/10000 if kind=='reg' else (y[ix]==10000).astype(float)).predict(kb)
                    p=np.clip(p,0,1);p=p*10000 if kind=='reg' else 10000*p+(1-p)*act
                    preds['kernel_'+kind+'_'+tag][va]=apply_inactivity_evidence(tr.iloc[ix],y[ix],tr.iloc[va],p)[0]
            print(label,fold,flush=True)
        saved=pd.DataFrame({'target':y,'baseline':ref[label]});scores={}
        for name,p in preds.items():
            p[matched]=lookup[matched];saved[name]=p
            for weight in [.25,.5,1.]:
                mixed=(1-weight)*ref[label].to_numpy()+weight*p
                scores[name+'_blend'+str(weight)]=float(np.mean((mixed-y)**2))
        scores['baseline']=float(np.mean((ref[label].to_numpy()-y)**2));report['protocols'][label]=scores
        saved.to_csv(OUT/(label+'_oof.csv'),index=False)
        print(label,sorted(scores.items(),key=lambda x:x[1])[:3],flush=True)
    report['discovery_pass']=[n for n in report['protocols']['42'] if n!='baseline' and all(report['protocols'][k][n]<report['protocols'][k]['baseline'] for k in ['42','2026','scaffold'])]
    report['confirmation_pass']=[n for n in report['discovery_pass'] if all(report['protocols'][k][n]<report['protocols'][k]['baseline'] for k in ['117','2027','7'])]
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print('CONFIRMED',report['confirmation_pass'],flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
