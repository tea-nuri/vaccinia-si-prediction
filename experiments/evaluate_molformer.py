"""Fold-fitted supervised heads for frozen IBM MoLFormer representations."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import corrected_pipeline as cp
from collect_training_v2 import parent_key
from experiments.evaluate_external_transfer import kernel_predict,fingerprints
from experiments.evaluate_corrected_representations import categories


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    train,test=cp.read_inputs();external=pd.read_csv(cp.DATA/'external_training_candidate_v3.csv')
    forbidden=set(pd.concat([train.SMILES,test.SMILES]).map(parent_key))
    assert not set(external.SMILES.map(parent_key))&forbidden
    combined=pd.concat([train[cp.ALLOWED],external[cp.ALLOWED]],ignore_index=True)
    meta=json.loads(args.embeddings.with_suffix('.json').read_text())
    assert cp.sha(args.embeddings)==meta['embedding_sha256']
    assert meta['source_sha256']['train']==cp.TRAIN_SHA256
    assert meta['source_sha256']['external']==cp.sha(cp.DATA/'external_training_candidate_v3.csv')
    assert combined.SMILES.tolist()==meta['raw_inputs']
    assert [Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=False) for s in combined.SMILES]==meta['canonical_inputs']
    x=np.load(args.embeddings,allow_pickle=False)['vectors'].astype(float)
    assert x.shape==(len(combined),768) and np.isfinite(x).all()
    y=train[cp.TARGET].to_numpy(float);ay=np.r_[y,external.model_target.to_numpy(float)]
    _,scaf,fp,_=cp.feature_arrays(combined);chem=cp.tanimoto(fp,fp)
    cats=cp.category_values(combined)
    ctx=.5*(cats[:,None,0]==cats[None,:,0])+.5*(cats[:,None,1]==cats[None,:,1])
    normalized=categories(combined)
    domain=(normalized[:,None,0]==normalized[None,:,0])*(
        .5+.5*(normalized[:,None,1]==normalized[None,:,1]))
    configs={f'{rep}_{pool}_a{alpha}_{transform}':{'rep':rep,'pool':pool,'alpha':alpha,'transform':transform}
             for rep in ['molformer','hybrid'] for pool in ['official','external']
             for alpha in [.1,1.] for transform in ['raw','power075','sqrt','log']}
    splits=cp.SCAFFOLD+cp.MOLECULAR
    report={'task_version':cp.TASK_VERSION,'status':'running','configs':configs,'splits':splits,
            'source_sha256':{'script':cp.sha(__file__),'embedding':cp.sha(args.embeddings),
                             'kernel_code':cp.sha(ROOT/'experiments/evaluate_external_transfer.py'),
                             'train':cp.TRAIN_SHA256,'external':cp.sha(cp.DATA/'external_training_candidate_v3.csv')},
            'encoder':{k:meta[k] for k in ['model','revision','weights_sha256','extractor_sha256','max_tokens','unknown_token_occurrences']},
            'feature_allowlist':cp.ALLOWED,'context_kernel':args.context,
            'test_predictions_generated':False,'test_fitted_transforms':False,
            'competition_parent_overlap':0,'target_clipping':False,'external_weight':.25,
            'selection':'Exploratory comparison; no automatic replacement or submission.',
            'metrics':{},'known_culture_metrics':{},'folds':{},
            'limitations':['Repeated protocols reuse the same 320 labels.',
                           'No Public gain follows directly from these results.']}
    cp.save_json(args.output/'plan.json',report)
    for split in splits:
        start=time.monotonic();ref=pd.read_csv(args.reference/f'oof_{split}.csv')
        np.testing.assert_array_equal(ref.target,y)
        predictions={n:np.full(len(y),np.nan) for n in configs};predictions['rf_log']=ref.rf_log.to_numpy()
        ids=ref.fold.to_numpy(int);known=ref.known_culture.to_numpy(bool);report['folds'][split]=[]
        for fold,ix,va in cp.folds_for(train,split):
            np.testing.assert_array_equal(np.flatnonzero(ids==fold),va)
            ex=np.arange(len(train),len(combined))
            if split in cp.SCAFFOLD:ex=ex[~np.isin(scaf[ex],scaf[va])]
            report['folds'][split].append({'fold':fold,'train_positions':ix.tolist(),
                                          'validation_positions':va.tolist(),'external_positions':(ex-len(train)).tolist()})
            for pool in ['official','external']:
                tr=ix if pool=='official' else np.r_[ix,ex]
                scaler=StandardScaler();a=scaler.fit_transform(x[tr]);b=scaler.transform(x[va])
                # Euclidean distances are normalized by dimension, with no
                # median-distance bandwidth fitted using validation rows.
                aa=np.maximum(0.,((a[:,None,:]-a[None,:,:])**2).mean(2))
                ba=np.maximum(0.,((b[:,None,:]-a[None,:,:])**2).mean(2))
                ka,kb=np.exp(-aa),np.exp(-ba)
                weights=np.r_[np.ones(len(ix)),np.full(len(tr)-len(ix),.25)]
                for rep in ['molformer','hybrid']:
                    if rep=='hybrid':at,bt=.5*ka+.5*chem[np.ix_(tr,tr)],.5*kb+.5*chem[np.ix_(va,tr)]
                    else:at,bt=ka,kb
                    if args.context=='product':
                        at,bt=at*(.1+.9*domain[np.ix_(tr,tr)]),bt*(.1+.9*domain[np.ix_(va,tr)])
                    else:
                        at,bt=.8*at+.2*ctx[np.ix_(tr,tr)],.8*bt+.2*ctx[np.ix_(va,tr)]
                    for alpha in [.1,1.]:
                        for transform in ['raw','power075','sqrt','log']:
                            name=f'{rep}_{pool}_a{alpha}_{transform}'
                            if transform=='power075':
                                predictions[name][va]=kernel_predict(at,bt,ay[tr]**.75,weights,alpha,'raw')**(1/.75)
                            else:
                                predictions[name][va]=kernel_predict(at,bt,ay[tr],weights,alpha,transform)
        assert all(np.isfinite(v).all() for v in predictions.values())
        report['metrics'][split]={n:cp.metrics(y,p) for n,p in predictions.items()}
        report['known_culture_metrics'][split]={n:cp.metrics(y[known],p[known]) for n,p in predictions.items()}
        pd.DataFrame({'target':y,'fold':ids,'known_culture':known,**predictions}).to_csv(args.output/f'oof_{split}.csv',index=False)
        cp.save_json(args.output/'metrics.json',report)
        print(split,round(time.monotonic()-start,2),sorted([(n,round(m['mse'])) for n,m in report['metrics'][split].items()],key=lambda x:x[1])[:4],flush=True)
    for label,protocols in [('scaffold_mean',cp.SCAFFOLD),('molecular2_mean',cp.MOLECULAR[:2]),('molecular8_mean',cp.MOLECULAR)]:
        report[label]={n:{k:float(np.mean([report['metrics'][s][n][k] for s in protocols])) for k in ['mse','mae']} for n in predictions}
    report['status']='completed';cp.save_json(args.output/'metrics.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--context',choices=['additive','product'],default='additive')
    with threadpool_limits(limits=1):main(p.parse_args())
