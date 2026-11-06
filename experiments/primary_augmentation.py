"""Add independently curated measured AND explicit-inactive compounds to V7.

All original-train structures are excluded from the external pool by neutralized
fragment-parent connectivity, including alternate stereo/mixture records.
Scaffold CV also removes all external validation scaffolds. Lookup is unchanged.
Run: python experiments/primary_augmentation.py
"""
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import train as core
OUT=ROOT/'run_primary_augmentation'

def parent(s):
    m=rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(Chem.MolFromSmiles(s)))
    return Chem.MolToSmiles(m,isomericSmiles=False)

def pool(training):
    p=pd.read_csv(core.DATA/'primary_literature.csv')
    p=p[p.retrieval_status.eq('eligible')].copy()
    p['SI VACV']=np.where(p.outcome.eq('Not active'),10000.,p.SI_from_rounded_means)
    p=p[p['SI VACV'].notna()].copy()
    p['SMILES']=p.smiles;p['Culture']=p.cell
    # Match the spelling of existing allowed assay categories without labels.
    contexts=training[['strain','Culture']].drop_duplicates()
    matches=contexts[contexts.strain.str.contains('copenhagen',case=False,na=False)&contexts.Culture.eq('Vero')]
    if len(matches)==1:p['strain']=matches.iloc[0].strain
    p['canonical']=p.SMILES.map(lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s)))
    p['scaffold']=p.SMILES.map(lambda s:core.MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC')
    original=set(training.SMILES.map(parent))
    p=p[~p.SMILES.map(parent).isin(original)].copy()
    assert not set(p.SMILES.map(parent))&original
    return p.reset_index(drop=True)

def run():
    start=time.perf_counter();OUT.mkdir(exist_ok=True)
    tr=pd.read_csv(core.DATA/'train.csv',usecols=core.ALLOWED+['SI VACV']);y=tr['SI VACV'].to_numpy(float)
    ext=pool(tr); baseline=pd.read_csv(ROOT/'results/validation.csv')
    assert len(tr)==len(baseline);np.testing.assert_allclose(y,baseline.target)
    canonical=tr.SMILES.map(lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s))).to_numpy()
    scaff=tr.SMILES.map(lambda s:core.MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
    configs={f'{scope}_{w}':(scope,w) for scope in ['all','inactive'] for w in [.1,.3,1.]}
    report=dict(external_rows=len(ext),external_molecules=ext.canonical.nunique(),external_inactive_rows=int(ext['SI VACV'].eq(10000).sum()),
      rule='No original training parent connectivity in external pool; no validation scaffolds for scaffold CV. Same fixed V7 lookup.',
      selection_rule='All five molecule splits and scaffold must improve; exploratory repeated CV, no independent holdout.',scores={})
    ext.to_csv(OUT/'external_used.csv',index=False)
    for seed in [42,2026,None]:
        name=str(seed) if seed is not None else 'scaffold';groups=canonical if seed is not None else scaff
        candidates={k:np.zeros(len(tr)) for k in configs}
        for ix,va in GroupKFold(5,shuffle=seed is not None,random_state=seed).split(tr,groups=groups):
            assert not set(canonical[ix])&set(canonical[va])
            for key,(scope,w) in configs.items():
                use=ext if scope=='all' else ext[ext['SI VACV'].eq(10000)]
                a,b,sw,used=core.augment(tr.iloc[ix],y[ix],use,w,set(scaff[va]) if seed is None else None)
                candidates[key][va],_=core.fit_predict(a,b,tr.iloc[va],sample_weight=sw)
        rows={'baseline':float(np.mean((baseline[name]-y)**2))}
        for key,p in candidates.items():rows[key]=float(np.mean((p-y)**2))
        report['scores'][name]=rows
        pd.DataFrame({'target':y,**candidates}).to_csv(OUT/f'{name}_oof.csv',index=False)
        print(name,json.dumps(rows),flush=True)
    report['passing_screen']=[k for k in configs if all(rows[k]<rows['baseline'] for rows in report['scores'].values())]
    report['runtime_seconds']=time.perf_counter()-start
    report['source_sha256']={name:hashlib.sha256((core.DATA/name).read_bytes()).hexdigest() for name in ['train.csv','primary_literature.csv']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
