"""Learn how reported assay limits relate to competition labels within each fold.

This is a predictive feature transfer with partial stereo compatibility, not an
assertion that an external bound is the exact biological SI of a target molecule.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train import ALLOWED
from literature import cell_key,structure_key
from experiments.measurement_agreement import main as reproduce_baseline


def partial_stereo_compatible(source,target):
    a,b=Chem.MolFromSmiles(source),Chem.MolFromSmiles(target)
    if a is None or b is None:
        raise ValueError('Invalid structure')
    if a.GetNumAtoms()!=b.GetNumAtoms() or a.GetNumBonds()!=b.GetNumBonds():
        return False
    if Chem.MolToSmiles(a,isomericSmiles=False)!=Chem.MolToSmiles(b,isomericSmiles=False):
        return False
    return b.HasSubstructMatch(a,useChirality=True)


def external_features(frame):
    source=pd.read_csv(ROOT/'data/bailey2007_bounds.csv')
    source=source[source.feature_eligible.eq(True)]
    records=[]
    for _,row in frame[ALLOWED].iterrows():
        candidates=[]
        if cell_key(row.Culture)=='bsc40' and 'ihdj' in ''.join(c for c in str(row.strain).lower() if c.isalnum()):
            candidates=[r for r in source.itertuples() if partial_stereo_compatible(r.smiles,row.SMILES)]
        # One eligible source observation is required. Ambiguity never chooses a target-dependent row.
        r=candidates[0] if len(candidates)==1 else None
        records.append(dict(reported_limit=np.nan if r is None else r.reported_cc50_ec50_limit,
                            molecule_id='' if r is None else r.molecule_id,
                            match_type='' if r is None else 'same_connectivity_and_source_specified_stereocentres',
                            exact_stereo=False if r is None else structure_key(r.smiles)==structure_key(row.SMILES)))
    return pd.DataFrame(records)


def calibrated_transfer(training,y,target,baseline,weight=1.):
    a,b=external_features(training),external_features(target)
    valid=a.reported_limit.notna().to_numpy()
    pred=np.asarray(baseline,dtype=float).copy()
    if a.loc[valid,'molecule_id'].nunique()<3:
        return pred,{'applied_rows':0,'reason':'insufficient_fold_support'}
    # Duplicate observations must not dominate the fitted slope.
    counts=a.loc[valid].groupby('molecule_id').molecule_id.transform('size').to_numpy()
    weights=1/counts;x=a.loc[valid,'reported_limit'].to_numpy();labels=np.asarray(y)[valid]
    slope=max(0.,float(np.sum(weights*x*labels)/np.sum(weights*x*x)))
    eligible=b.reported_limit.notna().to_numpy()
    pred[eligible]=(1-weight)*pred[eligible]+weight*slope*b.loc[eligible,'reported_limit'].to_numpy()
    return pred,{'applied_rows':int(eligible.sum()),'slope':slope,
                 'support_rows':int(valid.sum()),'support_molecules':int(a.loc[valid,'molecule_id'].nunique())}


def main():
    output=ROOT/'run_bounded_features';output.mkdir(exist_ok=True)
    baseline_dir=ROOT/'run_measurement_agreement'
    if not all((baseline_dir/f'validation_{s}.csv').exists() for s in ['42','2026','117','2027','7','scaffold','23','101','509']):
        reproduce_baseline()
    tr=pd.read_csv(ROOT/'data/train.csv',usecols=ALLOWED+['SI VACV'])
    te=pd.read_csv(ROOT/'data/test.csv',usecols=ALLOWED+['index']);y=tr['SI VACV'].to_numpy(float)
    canon=tr.SMILES.map(structure_key).to_numpy()
    scaffold=tr.SMILES.map(lambda s:MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
    expected=json.loads((ROOT/'results/measurement_agreement_experiment.json').read_text())
    report={'status':'local_only_not_submitted','scores':{},'fold_calibration':{},
            'features':'Published CC50/EC50 lower limit, same connectivity, agreement of every source-specified stereocentre, BSC40/IHD-J context. Partial stereo is explicit uncertainty.',
            'model':'Nonnegative least-squares slope without intercept learned separately in each training fold; inverse source-molecule count weights; minimum 3 source molecules.',
            'weights':[.5,1.],
            'selection':'A tested weight must improve every original and additional split to be considered; choose lower five-seed mean among passing weights.',
            'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                            ['data/train.csv','data/test.csv','data/bailey2007_bounds.csv','results/candidate_agreement_text.csv']},
            'limitations':['Bounds are features, not exact external labels; predictions target the competition annotation convention.',
                           'Partial stereochemistry does not prove chemical identity; new stereoisomers can behave differently.',
                           'Source investigated after observing training errors; these are exploratory retrieval-assisted scores.',
                           'Repeated split protocols reuse the same dataset; validation molecules are excluded from their own training fold. No public score below one million is claimed.']}
    for split in ['42','2026','117','2027','7','scaffold','23','101','509']:
        baseline=pd.read_csv(baseline_dir/f'validation_{split}.csv')
        np.testing.assert_array_equal(baseline.target,y)
        old=baseline.agreement_and_gate.to_numpy()
        np.testing.assert_allclose(np.mean((old-y)**2),expected['scores'][split]['agreement_and_gate'],rtol=0,atol=1e-5)
        seed=None if split=='scaffold' else int(split)
        predictions={str(w):old.copy() for w in [.5,1.]};folds=[]
        for ix,va in GroupKFold(5,shuffle=seed is not None,random_state=seed).split(tr,groups=scaffold if seed is None else canon):
            assert not set(canon[ix])&set(canon[va])
            for weight in [.5,1.]:
                predictions[str(weight)][va],model=calibrated_transfer(tr.iloc[ix],y[ix],tr.iloc[va],old[va],weight)
            folds.append(model)
        report['scores'][split]={'baseline':float(np.mean((old-y)**2)),**{k:float(np.mean((p-y)**2)) for k,p in predictions.items()}}
        report['fold_calibration'][split]=folds
        pd.DataFrame({'target':y,'baseline':old,**predictions}).to_csv(output/f'validation_{split}.csv',index=False)
        print(split,report['scores'][split],flush=True)
    report['passing']=[str(w) for w in [.5,1.] if all(v[str(w)]<v['baseline'] for v in report['scores'].values())]
    report['selected']=min(report['passing'],key=lambda k:np.mean([report['scores'][s][k] for s in ['42','2026','117','2027','7']])) if report['passing'] else None
    report['mean_original_seeds']={key:float(np.mean([report['scores'][s][key] for s in ['42','2026','117','2027','7']])) for key in ['baseline','0.5','1.0']}
    report['mean_additional_seeds']={key:float(np.mean([report['scores'][s][key] for s in ['23','101','509']])) for key in ['baseline','0.5','1.0']}
    if report['selected']:
        sub=pd.read_csv(ROOT/'results/candidate_agreement_text.csv');np.testing.assert_array_equal(sub['index'],te['index'])
        sub['SI VACV'],report['full_fit']=calibrated_transfer(tr,y,te,sub['SI VACV'].to_numpy(),float(report['selected']))
        sub.to_csv(output/'submission.csv',index=False)
        report['candidate_submission_sha256']=hashlib.sha256((output/'submission.csv').read_bytes()).hexdigest()
    for name,f in [('train',tr),('test',te)]:external_features(f).to_csv(output/f'{name}_features.csv',index=False)
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    print('Selected:',report['selected'],flush=True)


if __name__=='__main__':main()
