"""Reproduce an unsubmitted candidate over frozen V7 predictions and folds."""
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train import ALLOWED
from literature import lookup_predictions,structure_key,SPLITS

def candidate_lookup(target):
    """Only add exact primary matches where the selected V7 has no lookup.

Existing V7 matches remain frozen, including conflicting later/earlier assays.
Only this candidate inspects the additional table. No training labels are read.
"""
    existing=lookup_predictions(target)
    primary=pd.read_csv(ROOT/'data/literature_candidates.csv')
    new=lookup_predictions(target,numerical=pd.read_csv(ROOT/'data/external.csv').iloc[:0],primary=primary,include_2019=True)
    new['existing_v7_prediction']=existing.prediction
    new['eligible_new_match']=new.prediction.notna() & existing.prediction.isna()
    return new

def main(output):
    output=Path(output)
    if output.resolve()==(ROOT/'results').resolve():raise ValueError('Use a candidate output directory')
    output.mkdir(parents=True,exist_ok=True)
    tr=pd.read_csv(ROOT/'data/train.csv',usecols=ALLOWED+['SI VACV'])
    te=pd.read_csv(ROOT/'data/test.csv',usecols=['index']+ALLOWED)
    baseline=pd.read_csv(ROOT/'results/validation.csv');submission=pd.read_csv(ROOT/'results/submission.csv')
    metadata=json.loads((ROOT/'results/metrics.json').read_text())
    for name,digest in metadata['source_sha256'].items():
        assert hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((ROOT/'results/submission.csv').read_bytes()).hexdigest()==metadata['submission_sha256']
    y=tr['SI VACV'].to_numpy();np.testing.assert_array_equal(baseline.target,y)
    np.testing.assert_array_equal(baseline.canonical_smiles,tr.SMILES.map(structure_key))
    np.testing.assert_array_equal(submission['index'],te['index'])
    audit=candidate_lookup(tr);taudit=candidate_lookup(te);mask=audit.eligible_new_match.to_numpy();tmask=taudit.eligible_new_match.to_numpy()
    out=baseline[['target','canonical_smiles']].copy();scores={};old_scores={};unchanged={}
    for split in SPLITS:
        p=baseline[split].to_numpy(copy=True);old_scores[split]=float(np.mean((p-y)**2))
        p[mask]=audit.loc[mask,'prediction'];out[split]=p;scores[split]=float(np.mean((p-y)**2))
        unchanged[split]=float(np.mean((p[~mask]-y[~mask])**2))
    for group in [scores,old_scores,unchanged]:group['mean_group_mse']=float(np.mean([group[k] for k in SPLITS[:-1]]))
    final=submission['SI VACV'].to_numpy(copy=True);final[tmask]=taudit.loc[tmask,'prediction']
    submission['SI VACV']=final;assert np.isfinite(final).all() and (final>=0).all()
    submission.to_csv(output/'submission.csv',index=False);out.to_csv(output/'validation.csv',index=False)
    audit['target']=y;audit.to_csv(output/'train_lookup_audit.csv',index=False)
    taudit.insert(0,'index',te['index']);taudit['final_prediction']=final;taudit.to_csv(output/'test_lookup_audit.csv',index=False)
    report=dict(model='V8 local candidate: V7 plus new unambiguous primary measurements',status='local_only_not_submitted',
        scores=scores,baseline_scores=old_scores,unchanged_rows_scores=unchanged,
        new_train_matches=int(mask.sum()),new_test_matches=int(tmask.sum()),
        overlapping_train_matches_skipped=int((audit.prediction.notna()&~audit.eligible_new_match).sum()),
        overlapping_test_matches_skipped=int((taudit.prediction.notna()&~taudit.eligible_new_match).sum()),
        train_rows=len(tr),test_rows=len(te),features_allowed=ALLOWED,
        selection='Local candidate only; preserve selected public V7. No Kaggle submission authorized.',
        limitations=['Retrieval-assisted CV includes published measurements for validation molecules; not novel-compound generalization.',
                    'The source was sought after inspecting training errors. All six CV splits reuse the same molecules.',
                    'Published SI and ratios of rounded means differ; assay context cannot resolve every discrepancy.',
                    'Existing V7 external matches are retained; additional conflicting studies cannot silently override them.'],
        artifact_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['data/literature_candidates.csv','results/submission.csv','results/validation.csv']},
        candidate_submission_sha256=hashlib.sha256((output/'submission.csv').read_bytes()).hexdigest())
    (output/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'run_candidate_v8');args=parser.parse_args();main(args.output)
