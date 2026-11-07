"""Curate fitted external reporter potency, not a cellular SI or IC50 label.

Only complete, adequately fitted inhibitory curves with an estimate inside the
measured input range are used. Partial/extrapolated estimates remain excluded.
"""
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import corrected_pipeline as cp
from collect_training_v2 import parent_key


def main(cache, output):
    source=cache/'pubchem';out=output;out.mkdir(parents=True,exist_ok=True)
    auxiliary=pd.read_csv(source/'auxiliary_filtered.csv').set_index('CID')
    train,test=cp.read_inputs();excluded=set(pd.concat([train.SMILES,test.SMILES]).map(parent_key))
    assert not set(auxiliary.parent_connectivity)&excluded
    records=[];audit=[];sources=[]
    for aid in [720580,720579]:
        parts=[]
        for path in sorted((source/'positive_full_records').glob(f'{aid}_*.csv')):
            meta=json.loads(path.with_suffix('.source.json').read_text());assert cp.sha(path)==meta['sha256']
            raw=pd.read_csv(path,low_memory=False)
            raw=raw[pd.to_numeric(raw.PUBCHEM_SID,errors='coerce').notna()].copy()
            raw['source_snapshot_sha256']=meta['sha256'];parts.append(raw)
            sources.append({'file':meta['original_path'],'url':meta['url'],'sha256':meta['sha256']})
        frame=pd.concat(parts,ignore_index=True)
        tested=[float(re.search(r'Activity at (\d+(?:\.\d+)?) uM',c)[1]) for c in frame if re.fullmatch(r'Activity at (\d+(?:\.\d+)?) uM',c)]
        low,high=min(tested),max(tested)
        counts={'aid':aid,'source_rows':len(frame),'retained_rows':0,'rejected':{}}
        for _,row in frame.iterrows():
            cid=int(row.PUBCHEM_CID)
            if cid not in auxiliary.index:raise ValueError('Unfiltered source structure')
            reasons=[]
            if row.PUBCHEM_ACTIVITY_OUTCOME!='Active':reasons.append('not_active')
            if not str(row.Curve_Description).startswith('Complete curve'):reasons.append('partial_curve')
            if not np.isfinite(float(row.Fit_R2)) or float(row.Fit_R2)<.8:reasons.append('poor_fit')
            if float(row.Fit_CurveClass)>=0:reasons.append('not_inhibitory_curve')
            potency=float(row.Potency)
            if not np.isfinite(potency) or potency<=0:reasons.append('invalid_potency')
            elif not low<=potency<=high:reasons.append('fit_extrapolates_outside_measured_input_range')
            if reasons:
                for reason in reasons:counts['rejected'][reason]=counts['rejected'].get(reason,0)+1
                continue
            mol=auxiliary.loc[cid]
            records.append({'CID':cid,'SID':int(row.PUBCHEM_SID),'SMILES':mol.SMILES,
                            'parent_connectivity':mol.parent_connectivity,'scaffold':mol.scaffold,
                            'reporter_assay':str(aid),'auxiliary_potency_uM':potency,
                            'fitted_logAC50':float(row.Fit_LogAC50),'fit_R2':float(row.Fit_R2),
                            'curve_description':row.Curve_Description,
                            'curve_class':float(row.Fit_CurveClass),'compound_QC_text':row['Compound QC'],
                            'source_url':f'https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}',
                            'source_snapshot_sha256':row.source_snapshot_sha256,
                            'endpoint_note':'Estimated reporter inhibitory potency, not cellular CC50/IC50/SI.'})
            counts['retained_rows']+=1
        audit.append(counts)
    frame=pd.DataFrame(records).sort_values(['CID','reporter_assay','SID']).reset_index(drop=True)
    frame['sample_weight']=1./frame.groupby(['CID','reporter_assay']).CID.transform('size')
    assert not set(frame.parent_connectivity)&excluded
    frame.to_csv(out/'potency_quantitative.csv',index=False)
    report={'status':'curated_external_auxiliary_reporter_potency','rows':len(frame),
            'unique_compounds':frame.CID.nunique(),'unique_parents':frame.parent_connectivity.nunique(),
            'source_audit':audit,'competition_parent_overlap':0,'SI_labels_created':False,
            'script_sha256':cp.sha(__file__),'output_sha256':cp.sha(out/'potency_quantitative.csv'),
            'sources':sources,'limitations':['Reporter potency is an auxiliary endpoint, not SI.',
                                            'Possible toxicity and reporter interference remain unmeasured confounders.',
                                            'Do not combine these measurements with unrelated cytotoxicity into invented SI labels.']}
    cp.save_json(out/'potency_provenance.json',report)
    print(json.dumps({k:report[k] for k in ['rows','unique_compounds','unique_parents','source_audit']},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.cache,args.output)
