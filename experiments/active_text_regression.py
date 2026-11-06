"""Ablate the active SI regressor; keep the V9 classifier/retrieval fixed.

The current active regressor clips all training targets to 5000. Compare
fold-local text regressors with and without that cap, without changing labels.
"""
import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import ALLOWED, fit_predict_v6, serialization_features, apply_inactivity_evidence
from literature import structure_key
from experiments.text_gate import v9_lookup
from experiments.camphor2022 import main as reproduce_v9


def main():
    output = ROOT/'run_active_text'
    output.mkdir(exist_ok=True)
    reproduce_v9(output/'baseline_v9')
    tr = pd.read_csv(ROOT/'data/train.csv', usecols=ALLOWED+['SI VACV'])
    y = tr['SI VACV'].to_numpy(float)
    base = pd.read_csv(output/'baseline_v9/validation.csv')
    lookup = v9_lookup(tr).prediction.to_numpy()
    canon = tr.SMILES.map(structure_key).to_numpy()
    scaffold = tr.SMILES.map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
    report = {'status':'local_only_not_submitted', 'scores':{},
              'selection':'Strict improvement on 42/2026/scaffold then 117/2027/7; no public-score tuning.',
              'hypothesis':'Active-only cap5000 and sparse structural features can underestimate large measured SI.',
              'settings':'Replace only active regressor in V6 context branch by active-only TFIDF 2-6 + Ridge alpha .1/1/10, cap5000 or no target cap; nonnegative predictions. Other branches and V9 retrieval frozen.',
              'limitations':['Repeated exploratory validation on previously studied molecules.',
                             'Uncapped models can produce unstable large predictions.']}
    report['source_sha256']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                            ['data/train.csv','results/candidate_camphor2022.csv']}
    predictions = {}
    for split in ['42','2026','scaffold','117','2027','7']:
        seed = None if split == 'scaffold' else int(split)
        pp = {}
        for fold,(ix,va) in enumerate(GroupKFold(5,shuffle=seed is not None,random_state=seed).split(tr, groups=scaffold if seed is None else canon)):
            original, model = fit_predict_v6(tr.iloc[ix], y[ix], tr.iloc[va])
            restored = original.copy()
            covered = np.isfinite(lookup[va])
            restored[covered] = lookup[va][covered]
            np.testing.assert_allclose(restored,base[split].to_numpy()[va],atol=1e-6,rtol=0)
            active = y[ix] != 10000
            tf = TfidfVectorizer(analyzer='char',ngram_range=(2,6),lowercase=False,sublinear_tf=True)
            a = tf.fit_transform(tr.iloc[ix[active]].SMILES)
            b = tf.transform(tr.iloc[va].SMILES)
            # Rebuild the context classifier inputs exactly as in train.py.
            xv = pd.DataFrame(serialization_features(tr.iloc[va].SMILES)).astype(float)
            for col in ['strain','Culture']:
                xv[col] = tr.iloc[va][col].fillna('UNKNOWN').astype(str).to_numpy()
            prob = model['inactive_classifier'].predict_proba(xv)[:,1]
            old_active = np.clip(model['active_regressor'].predict(serialization_features(tr.iloc[va].SMILES)),0,5000)
            old_context = prob*10000+(1-prob)*old_active
            # Reconstruct raw ensemble to avoid undoing evidence correction.
            sm = model['smiles_models']
            text = np.clip(sm['text'].predict(tr.iloc[va].SMILES),0,10000)
            trees = np.clip(sm['trees'].predict(serialization_features(tr.iloc[va].SMILES)),0,10000)
            raw = np.clip((.85*text+.15*trees-5000)*1.5+5000,0,10000)
            for cap in [5000,None]:
                targets = np.minimum(y[ix][active],cap) if cap else y[ix][active]
                for alpha in [.1,1.,10.]:
                    new_active = np.maximum(Ridge(alpha=alpha).fit(a,targets).predict(b),0)
                    new_context = prob*10000+(1-prob)*new_active
                    for weight in [.5,1.]:
                        p = .5*raw+.5*((1-weight)*old_context+weight*new_context)
                        p,_ = apply_inactivity_evidence(tr.iloc[ix],y[ix],tr.iloc[va],p)
                        matched = np.isfinite(lookup[va]);p[matched]=lookup[va][matched]
                        name=f'cap{cap}_alpha{alpha:g}_weight{weight:g}'
                        pp.setdefault(name,np.zeros(len(y)))[va]=p
            print(split,fold,flush=True)
        report['scores'][split] = {'baseline':float(np.mean((base[split]-y)**2)), **{
            k:float(np.mean((p-y)**2)) for k,p in pp.items()}}
        for k,p in pp.items(): predictions[split+':'+k]=p
        print(split, sorted(report['scores'][split].items(),key=lambda p:p[1])[:3],flush=True)
        (output/'metrics.json').write_text(json.dumps(report,indent=2))
    report['passing']=[k for k in report['scores']['42'] if k!='baseline' and all(
        v[k]<v['baseline'] for v in report['scores'].values())]
    np.savez_compressed(output/'predictions.npz',**predictions)
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    print('Passing:',report['passing'],flush=True)


if __name__=='__main__':
    main()
