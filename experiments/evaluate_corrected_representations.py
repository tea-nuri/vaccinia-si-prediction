"""Corrected-task representations and category spelling ablation.

All learned preprocessing and neighbor selection use training folds only. No
test rows, external labels, or upper target caps participate in predictions.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'run_corrected_candidates'
sys.path.insert(0,str(ROOT))
import corrected_pipeline as cp
from collect_training_v2 import parent_key

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
AP = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048,includeChirality=False)


def culture_key(value):
    # Remove annotation words, retaining actual cell-line suffixes (Vero != Vero76).
    s = '' if pd.isna(value) else str(value).casefold()
    s = re.sub(r'\b(cells?|culture|primary)\b',' ',s)
    s = re.sub('[^a-z0-9]','',s)
    return {'humanforeskinfibroblast':'hff','humanforeskinfibroblasts':'hff'}.get(s,s) or 'unknown'


def strain_key(value):
    # Names occur in the provided training input. Unknown strains remain unknown;
    # additional virus names in a source annotation never become a VACV strain.
    s = '' if pd.isna(value) else str(value).casefold()
    for expression,name in [(r'western reserve|\bwr\b','wr'),(r'copenhagen','copenhagen'),
                            (r'ihd[ -]?j','ihdj'),(r'tiantan','tiantan'),(r'livp','livp'),
                            (r'lederle','lederle'),(r'lister|vr[ -]?1549','lister')]:
        if re.search(expression,s):
            return name
    return 'unspecified'


def categories(frame):
    normalized = np.column_stack([frame.Culture.map(culture_key),frame.strain.map(strain_key)])
    return normalized


@lru_cache(None)
def structural(smiles):
    mol = Chem.MolFromSmiles(parent_key(smiles))
    fp = GENERATOR.GetFingerprintAsNumPy(mol).astype(float)
    counts = GENERATOR.GetCountFingerprintAsNumPy(mol).astype(float)
    ap = AP.GetCountFingerprintAsNumPy(mol).astype(float)
    maccs = np.asarray(MACCSkeys.GenMACCSKeys(mol),dtype=float)
    return fp,counts,ap,maccs


def similarity(a,b):
    inter = a@b.T
    den = (a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-inter
    return np.divide(inter,den,out=np.zeros_like(inter),where=den>0)


def neighbor_predict(sim,y,tc,vc,k,transform):
    sameculture = vc[:,0,None]==tc[None,:,0]
    samestrain = vc[:,1,None]==tc[None,:,1]
    knownstrain = (vc[:,1,None]!='unspecified')&(tc[None,:,1]!='unspecified')
    weights = sim**4 * (.1+.9*sameculture) * (.5+.5*(samestrain|~knownstrain))
    nearest = np.argsort(weights,axis=1)[:,-min(k,len(y)):]
    chosen = np.take_along_axis(weights,nearest,axis=1)
    transformed = cp.target_transform(y,transform)
    mean = (chosen*transformed[nearest]).sum(1)/np.maximum(chosen.sum(1),1e-15)
    return cp.inverse_target(mean,transform)


def main(reference, output):
    output.mkdir(parents=True, exist_ok=True)
    frame,_ = cp.read_inputs()
    y = frame[cp.TARGET].to_numpy(float)
    _,_,chiral,desc = cp.feature_arrays(frame)
    structures = [structural(s) for s in frame.SMILES]
    features = {'parentbit':np.stack([r[0] for r in structures]),
                'count':np.stack([r[1] for r in structures]),
                'atompair':np.stack([r[2] for r in structures]),
                'maccs':np.stack([r[3] for r in structures]),'descriptor':np.empty((len(frame),0)),
                'chiral':chiral}
    kernels = {name:similarity(features[name],features[name]) for name in ['parentbit','count','atompair']}
    normalized = categories(frame)
    # Preserved original annotations plus normalized semantic categories.
    extended = np.column_stack([cp.category_values(frame),normalized])
    tree_names = [f'rf_{representation}_{transform}' for representation in features for transform in ['log','raw']]
    nn_names = [f'nn_{representation}_k{k}_{transform}' for representation in kernels for k in [3,10] for transform in ['raw','log']]
    names = tree_names+nn_names
    plan = {'task_version':cp.TASK_VERSION,'models':names,'splits':cp.SCAFFOLD+cp.MOLECULAR,
            'source_sha256':{'script':cp.sha(__file__),'pipeline':cp.sha(cp.__file__),'train':cp.sha(cp.DATA/'train_v2.csv')},
            'test_predictions_generated':False,'feature_allowlist':cp.ALLOWED,
            'external_measurements_used':False,'target_clipping':False,
            'tree_parameters':{'n_estimators':400,'min_samples_leaf':2,'max_features':.5,'random_state':42},
            'normalization':'Keep original annotations; append spelling-normalized cell line and strain parsed solely from allowed input text.',
            'selection_rule':'Same frozen gate: scaffold mean MSE >=10% better, each scaffold <=5% worse, known-Culture scaffold and molecular2 means no worse.',
            'limits':['Parent fingerprints ignore stereochemistry and cannot establish equal activity for stereoisomers.',
                      'Repeated protocols reuse 320 training labels; the minimum across models is optimistic.']}
    cp.save_json(output/'plan.json',plan)
    report = dict(plan,status='running',metrics={},known_culture_metrics={})
    for split in plan['splits']:
        start = time.monotonic()
        ref = pd.read_csv(reference/f'oof_{split}.csv')
        np.testing.assert_array_equal(ref.target,y)
        preds = {n:np.full(len(y),np.nan) for n in names};preds['rf_log'] = ref.rf_log.to_numpy()
        ids = np.full(len(y),-1,int);known=ref.known_culture.to_numpy(bool)
        for fold,ix,va in cp.folds_for(frame,split):
            np.testing.assert_array_equal(np.flatnonzero(ref.fold.to_numpy()==fold),va)
            imp = SimpleImputer(strategy='median',keep_empty_features=True)
            dt,dv = imp.fit_transform(desc[ix]),imp.transform(desc[va])
            encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
            ct,cv = encoder.fit_transform(extended[ix]),encoder.transform(extended[va])
            for representation,v in features.items():
                a = np.column_stack([v[ix],dt,ct]);b=np.column_stack([v[va],dv,cv])
                for transform in ['raw','log']:
                    name = f'rf_{representation}_{transform}'
                    model = RandomForestRegressor(**plan['tree_parameters'],n_jobs=2)
                    model.fit(a,cp.target_transform(y[ix],transform))
                    preds[name][va] = cp.inverse_target(model.predict(b),transform)
            for representation,kernel in kernels.items():
                sim=kernel[np.ix_(va,ix)]
                for k in [3,10]:
                    for transform in ['raw','log']:
                        preds[f'nn_{representation}_k{k}_{transform}'][va] = neighbor_predict(sim,y[ix],normalized[ix],normalized[va],k,transform)
            ids[va]=fold
        assert (ids>=0).all() and all(np.isfinite(p).all() for p in preds.values())
        report['metrics'][split] = {n:cp.metrics(y,p) for n,p in preds.items()}
        report['known_culture_metrics'][split] = {n:cp.metrics(y[known],p[known]) for n,p in preds.items()}
        pd.DataFrame({'target':y,'fold':ids,'known_culture':known,**preds}).to_csv(output/f'oof_{split}.csv',index=False)
        cp.save_json(output/'metrics.json',report)
        best=min(names,key=lambda n:report['metrics'][split][n]['mse'])
        print(split,'seconds',round(time.monotonic()-start,2),'best',best,
              round(report['metrics'][split][best]['mse']),'baseline',round(report['metrics'][split]['rf_log']['mse']),flush=True)
    for label,protocols in [('scaffold_mean',cp.SCAFFOLD),('molecular2_mean',cp.MOLECULAR[:2]),
                            ('molecular5_mean',cp.MOLECULAR[:5]),('molecular_extra3_mean',cp.MOLECULAR[5:])]:
        for field,suffix in [('metrics',''),('known_culture_metrics','_known_culture')]:
            report[label+suffix] = {n:{k:float(np.mean([report[field][s][n][k] for s in protocols])) for k in ['mse','mae']} for n in preds}
    report['material_checks'] = {}
    for name in names:
        report['material_checks'][name] = {
            'scaffold_mean_improves10percent':report['scaffold_mean'][name]['mse']<=.9*report['scaffold_mean']['rf_log']['mse'],
            'each_scaffold_no_more_than5percent_worse':all(report['metrics'][s][name]['mse']<=1.05*report['metrics'][s]['rf_log']['mse'] for s in cp.SCAFFOLD),
            'known_culture_scaffold_mean_no_worse':report['scaffold_mean_known_culture'][name]['mse']<=report['scaffold_mean_known_culture']['rf_log']['mse'],
            'molecular2_mean_no_worse':report['molecular2_mean'][name]['mse']<=report['molecular2_mean']['rf_log']['mse']}
    report['passing']=[n for n,c in report['material_checks'].items() if all(c.values())]
    report['status']='completed';cp.save_json(output/'metrics.json',report)
    print(json.dumps({'scaffold_mean':report['scaffold_mean'],'passing':report['passing']},indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):main(args.reference,args.output)
