"""Reproduce hybrid_external_a0.1_sqrt from the frozen MoLFormer benchmark."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import corrected_pipeline as cp
from collect_training_v2 import parent_key

CONFIG={'alpha':.1,'external_weight':.25,'transform':'sqrt','context':'additive',
        'molformer_rbf_weight':.4,'chiral_morgan_weight':.4,'context_weight':.2}


def rbf(a,b):
    squared=np.maximum(0.,((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T)/a.shape[1])
    return np.exp(-squared)


class MolformerCandidate:
    def fit(self,frame,labels,weights,vectors):
        frame=cp.allowed_frame(frame)
        vectors=np.asarray(vectors,float)
        if vectors.shape!=(len(frame),768) or not np.isfinite(vectors).all():raise ValueError('Embedding alignment or values invalid')
        self.scaler=StandardScaler();self.vectors=self.scaler.fit_transform(vectors)
        self.fp=cp.feature_arrays(frame)[2];self.context=cp.category_values(frame)
        self.weights=np.asarray(weights,float);self.normalized_weights=self.weights/self.weights.sum()
        k=self._kernel(self.vectors,self.fp,self.context)
        self.column_mean=self.normalized_weights@k;self.grand_mean=float(self.column_mean@self.normalized_weights)
        z=cp.target_transform(labels,'sqrt');self.mean_y=float(self.normalized_weights@z)
        a=k-self.column_mean[None,:]-self.column_mean[:,None]+self.grand_mean
        a.flat[::len(a)+1]+=.1/self.weights
        self.coefficients=cho_solve(cho_factor(a,lower=True,check_finite=False),z-self.mean_y,check_finite=False)
        return self

    def _kernel(self,vectors,fp,context):
        ck=.5*(context[:,None,0]==self.context[None,:,0])+.5*(context[:,None,1]==self.context[None,:,1])
        return .4*rbf(vectors,self.vectors)+.4*cp.tanimoto(fp,self.fp)+.2*ck

    def predict(self,frame,vectors):
        frame=cp.allowed_frame(frame);vectors=np.asarray(vectors,float)
        if vectors.shape!=(len(frame),768) or not np.isfinite(vectors).all():raise ValueError('Embedding alignment or values invalid')
        k=self._kernel(self.scaler.transform(vectors),cp.feature_arrays(frame)[2],cp.category_values(frame))
        b=k-self.column_mean[None,:]-(k@self.normalized_weights)[:,None]+self.grand_mean
        return cp.inverse_target(self.mean_y+b@self.coefficients,'sqrt')


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    train,test=cp.read_inputs();source=cp.DATA/'external_training_candidate_v3.csv'
    sm=json.loads(source.with_suffix('.validation.json').read_text());assert cp.sha(source)==sm['output_sha256']
    ext=pd.read_csv(source);forbidden=set(pd.concat([train.SMILES,test.SMILES]).map(parent_key))
    assert not set(ext.SMILES.map(parent_key))&forbidden
    features=pd.concat([train[cp.ALLOWED],ext[cp.ALLOWED]],ignore_index=True)
    all_inputs=pd.concat([features,test[cp.ALLOWED]],ignore_index=True)
    meta=json.loads(args.embeddings.with_suffix('.json').read_text())
    assert cp.sha(args.embeddings)==meta['embedding_sha256']
    assert all_inputs.SMILES.tolist()==meta['raw_inputs']
    assert meta['source_sha256']=={'train':cp.TRAIN_SHA256,'external':cp.sha(source),'test':cp.sha(cp.DATA/'test.csv')}
    assert meta['model']=='ibm-research/MoLFormer-XL-both-10pct' and meta['revision']=='361063d0ad524ef77cf39b08469f6be770dc550f'
    vectors=np.load(args.embeddings,allow_pickle=False)['vectors'].astype(float)
    labels=np.r_[train[cp.TARGET].to_numpy(float),ext.model_target.to_numpy(float)]
    weights=np.r_[np.ones(len(train)),np.full(len(ext),.25)]
    model=MolformerCandidate().fit(features,labels,weights,vectors[:len(features)])
    pred=model.predict(test,vectors[len(features):])
    pd.DataFrame({'index':test['index'],cp.TARGET:pred}).to_csv(args.output/'submission.csv',index=False)
    used=features.copy();used[cp.TARGET]=labels;used['sample_weight']=weights
    used['source']=['official_train_v2']*len(train)+ext.source_url.tolist();used['doi']=[None]*len(train)+ext.doi.tolist()
    used.to_csv(args.output/'training_used.csv',index=False);joblib.dump(model,args.output/'model.joblib')
    report={'task_version':cp.TASK_VERSION,'candidate':'hybrid_external_a0.1_sqrt','config':CONFIG,
            'status':'local_candidate_pending_kaggle','public_score':None,
            'feature_allowlist':cp.ALLOWED,'official_rows':len(train),'external_rows':len(ext),'test_rows':len(test),
            'competition_parent_overlap':0,'test_fitted_transforms':False,'test_outcomes_used':False,
            'source_sha256':{'script':cp.sha(__file__),'pipeline':cp.sha(cp.__file__),'train':cp.TRAIN_SHA256,
                             'external':cp.sha(source),'test':cp.sha(cp.DATA/'test.csv'),'embeddings':cp.sha(args.embeddings)},
            'encoder':{k:meta[k] for k in ['model','revision','weights_sha256','extractor_sha256','max_tokens','unknown_token_occurrences','truncation']},
            'submission_sha256':cp.sha(args.output/'submission.csv'),'training_used_sha256':cp.sha(args.output/'training_used.csv'),
            'prediction_quantiles':np.quantile(pred,[0,.5,.9,.95,.99,1]).tolist(),
            'submission_authorization':'User explicitly requested uploading the current locally improved candidate in this chat; one exploratory submission.',
            'limitations':['Candidate chosen after exploratory comparisons; local quality is optimistic.','The source-scale MSE improvement does not guarantee Public improvement.']}
    cp.save_json(args.output/'export.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    from molformer_candidate import main as run_main
    with threadpool_limits(limits=1):run_main(p.parse_args())
