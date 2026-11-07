"""Reproduce the exploratory external Morgan kernel candidate.

This is an additional candidate, not a replacement for the verified train.py
baseline. Only SMILES, Culture and strain enter the predictor.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from threadpoolctl import threadpool_limits
import corrected_pipeline as cp
from collect_training_v2 import parent_key

CONFIG={'alpha':.1,'external_weight':.25,'molecular_kernel_weight':.8,
        'context_kernel_weight':.2,'transform':'log','radius':2,'fingerprint_size':2048}


def fingerprint(frame):
    frame=cp.allowed_frame(frame)
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    return np.stack([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(parent_key(s)))
                     for s in frame.SMILES]).astype(float)


def kernel(a,b,ca,cb):
    context=.5*(ca[:,None,0]==cb[None,:,0])+.5*(ca[:,None,1]==cb[None,:,1])
    return .8*cp.tanimoto(a,b)+.2*context


class KernelCandidate:
    def fit(self,frame,y,weights):
        self.fp=fingerprint(frame);self.context=cp.category_values(frame)
        self.weights=np.asarray(weights,float);self.normalized_weights=self.weights/self.weights.sum()
        k=kernel(self.fp,self.fp,self.context,self.context)
        self.column_mean=self.normalized_weights@k
        self.grand_mean=float(self.column_mean@self.normalized_weights)
        z=cp.target_transform(y,'log');self.mean_y=float(self.normalized_weights@z)
        a=k-self.column_mean[None,:]-self.column_mean[:,None]+self.grand_mean
        a.flat[::len(a)+1]+=.1/self.weights
        self.coefficients=cho_solve(cho_factor(a,lower=True,check_finite=False),z-self.mean_y,
                                    check_finite=False)
        return self

    def predict(self,frame):
        b=kernel(fingerprint(frame),self.fp,cp.category_values(frame),self.context)
        b=b-self.column_mean[None,:]-(b@self.normalized_weights)[:,None]+self.grand_mean
        return cp.inverse_target(self.mean_y+b@self.coefficients,'log')


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    train,test=cp.read_inputs()
    source=cp.DATA/'external_training_candidate_v3.csv'
    source_meta=json.loads(source.with_suffix('.validation.json').read_text())
    assert cp.sha(source)==source_meta['output_sha256']
    external=pd.read_csv(source)
    forbidden=set(pd.concat([train.SMILES,test.SMILES]).map(parent_key))
    assert not set(external.SMILES.map(parent_key))&forbidden
    features=pd.concat([train[cp.ALLOWED],external[cp.ALLOWED]],ignore_index=True)
    targets=np.r_[train[cp.TARGET].to_numpy(float),external.model_target.to_numpy(float)]
    weights=np.r_[np.ones(len(train)),np.full(len(external),.25)]
    model=KernelCandidate().fit(features,targets,weights)
    prediction=model.predict(test)
    submission=pd.DataFrame({'index':test['index'],cp.TARGET:prediction})
    submission.to_csv(args.output/'submission.csv',index=False)
    used=features.copy();used[cp.TARGET]=targets;used['sample_weight']=weights
    used['source']=['official_train_v2']*len(train)+external.source_url.tolist()
    used['doi']=[None]*len(train)+external.doi.tolist()
    used.to_csv(args.output/'training_used.csv',index=False)
    joblib.dump(model,args.output/'model.joblib')
    meta={'task_version':cp.TASK_VERSION,'status':'exploratory_not_kaggle_verified',
          'config':CONFIG,'feature_allowlist':cp.ALLOWED,'training_rows':len(features),
          'official_rows':len(train),'external_rows':len(external),'test_parent_overlap':0,
          'test_outcomes_used':False,'competition_concentrations_used':False,
          'source_sha256':{'script':cp.sha(__file__),'pipeline':cp.sha(cp.__file__),
                           'train':cp.TRAIN_SHA256,'external':cp.sha(source),'test':cp.sha(cp.DATA/'test.csv')},
          'submission_sha256':cp.sha(args.output/'submission.csv'),
          'training_used_sha256':cp.sha(args.output/'training_used.csv'),
          'prediction_summary':submission[cp.TARGET].describe().to_dict(),
          'limitations':['Selected after exploratory comparison on reused splits.',
                         'Local gains are modest and do not establish any Public improvement.',
                         'Original-scale MSE is evaluated on the full uncapped target.']}
    cp.save_json(args.output/'export.json',meta)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    # Persist the importable class identity, rather than __main__.KernelCandidate.
    from kernel_candidate import main as run_main
    with threadpool_limits(limits=1):run_main(p.parse_args())
