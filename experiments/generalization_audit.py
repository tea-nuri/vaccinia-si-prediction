"""Explain V9 error concentration and distribution coverage without test labels."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train import ALLOWED
from literature import SPLITS, structure_key, cell_key
from experiments.text_gate import v9_lookup
from experiments.camphor2022 import main as reproduce_v9


def main():
    output=ROOT/'run_gap_audit'; output.mkdir(exist_ok=True)
    reproduce_v9(output/'baseline_v9')
    tr=pd.read_csv(ROOT/'data/train.csv',usecols=ALLOWED+['SI VACV'])
    te=pd.read_csv(ROOT/'data/test.csv',usecols=ALLOWED+['index'])
    y=tr['SI VACV'].to_numpy(float)
    oof=pd.read_csv(output/'baseline_v9/validation.csv')
    audit=v9_lookup(tr);test_audit=v9_lookup(te)
    error=((oof[SPLITS[:-1]].to_numpy()-y[:,None])**2).mean(1)
    matched=audit.prediction.notna().to_numpy()
    canon=tr.SMILES.map(structure_key).to_numpy()
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fpa=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr.SMILES]
    fpb=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in te.SMILES]
    sims=np.array([DataStructs.BulkTanimotoSimilarity(fp,fpa) for fp in fpa])
    test_sims=np.array([DataStructs.BulkTanimotoSimilarity(fp,fpa) for fp in fpb]).max(1)
    cv_sim=np.zeros(len(y))
    for ix,va in GroupKFold(5,shuffle=True,random_state=42).split(tr,groups=canon):
        cv_sim[va]=sims[np.ix_(va,ix)].max(1)
    report={'model':'V9 submitted candidate', 'local_mean_group_mse':float(error.mean()),
            'scaffold_mse':float(np.mean((oof.scaffold-y)**2)),
            'confirmed_public_score':3532699.00740,
            'coverage':{'train_rows':len(tr),'test_rows':len(te),'retrieval_train':int(matched.sum()),
                        'retrieval_test':int(test_audit.prediction.notna().sum())},
            'error_concentration':{'largest_row_share':float(error.max()/error.sum()),
                                   'top10_rows_share':float(np.sort(error)[-10:].sum()/error.sum()),
                                   'matched_mse':float(error[matched].mean()),
                                   'unmatched_mse':float(error[~matched].mean())},
            'similarity':{'fingerprint':'Morgan radius2, 2048 bits, chirality; nearest training molecule',
                          'quantiles':[0,.1,.25,.5,.75,.9,1],
                          'grouped_cv42':np.quantile(cv_sim,[0,.1,.25,.5,.75,.9,1]).tolist(),
                          'test':np.quantile(test_sims,[0,.1,.25,.5,.75,.9,1]).tolist()},
            'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                             ['data/train.csv','data/test.csv','results/candidate_camphor2022.csv']},
            'limitations':['The exact causes of hidden test errors cannot be determined without its labels.',
                           'Scaffold MSE is a stress test, not an unbiased prediction of Kaggle MSE.',
                           'Primary sources were sought after analyzing training errors; CV is exploratory.',
                           'No competition concentration columns, missingness, Title or index are predictors.']}
    details=audit.copy();details['source_row']=np.arange(len(y));details['target']=y
    details['mean_oof']=oof[SPLITS[:-1]].mean(axis=1);details['mean_squared_error']=error
    report['largest_errors']=details.sort_values('mean_squared_error',ascending=False).head(10).fillna('').to_dict(orient='records')
    context=pd.DataFrame({'train_fraction':tr.Culture.map(cell_key).value_counts(normalize=True),
                          'test_fraction':te.Culture.map(cell_key).value_counts(normalize=True)}).fillna(0)
    report['culture_distribution']=context.to_dict(orient='index')
    # Uncertainty illustration for an equally sized public subset of observed rows;
    # not a confidence interval for the real, potentially shifted test set.
    rng=np.random.default_rng(42);public_n=round(.3*len(te))
    draws=np.array([error[rng.choice(len(y),size=public_n,replace=False)].mean() for _ in range(10000)])
    report['observed_row_subset_sensitivity']={'rows_per_draw':public_n,'draws':10000,
        'quantiles':[.025,.5,.975], 'mse_quantiles':np.quantile(draws,[.025,.5,.975]).tolist(),
        'interpretation':'Finite-sample illustration only; does not model unseen labels, new outliers or test shift.'}
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
