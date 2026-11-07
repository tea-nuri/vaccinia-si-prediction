"""Traceable external CC50 components; inequalities remain inequalities."""
from collections import Counter
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import corrected_pipeline as cp
from collect_training_v2 import parent_key


def cell_name(description):
    for pattern,name in [
        (r'\bVero[ -]?76\b','Vero-76'),(r'\bVero\b','Vero'),
        (r'\bMT[ -]?4\b','MT-4'),(r'\bHEL\b|human embryonic lung','HEL'),
        (r'\bHeLa\b','HeLa'),(r'\bHFF\b|human foreskin fibroblast','Human foreskin fibroblast'),
        (r'\bBSC[ -]?1\b|\bBS-C-1\b','BSC-1'),(r'\bBSC[ -]?40\b|\bBS-C-40\b','BSC-40'),
        (r'\bA549\b','A549')]:
        if re.search(pattern,description,re.I):return name
    return None


def main(cache, output):
    out=output;out.mkdir(parents=True,exist_ok=True)
    train,test=cp.read_inputs();forbidden=set(pd.concat([train.SMILES,test.SMILES]).map(parent_key))
    rows=[];seen=set();counts=Counter();snapshots=[]
    for path in sorted((cache/'chembl').glob('CHEMBL*/activity_*.json')):
        if path.name.endswith('.source.json'):continue
        meta=json.loads(path.with_suffix('.source.json').read_text());assert cp.sha(path)==meta['sha256']
        document=json.loads((path.parent/'document.json').read_text())
        admitted_from_source=False
        for row in json.loads(path.read_text())['activities']:
            aid=row.get('activity_id')
            if row.get('standard_type')!='CC50' or aid in seen:continue
            seen.add(aid);counts['CC50_records']+=1
            smiles=row.get('canonical_smiles')
            if not smiles:counts['missing_structure']+=1;continue
            parent=parent_key(smiles)
            if parent in forbidden:counts['competition_parent_excluded']+=1;continue
            desc=row.get('assay_description') or '';cell=cell_name(desc)
            if cell is None:counts['unresolved_cell']+=1;continue
            if re.search(r'\binfected\b',desc,re.I) and not re.search(r'\b(?:uninfected|non[ -]infected|mock[ -]infected)\b',desc,re.I):
                counts['infected_context_excluded']+=1;continue
            if row.get('data_validity_comment') is not None or row.get('potential_duplicate') not in [0,None]:
                counts['quality_flag_excluded']+=1;continue
            relation=row.get('standard_relation');units=row.get('standard_units')
            if relation not in ['=','>','<','>=','<='] or units not in ['nM','uM']:
                counts['unsupported_relation_or_units']+=1;continue
            try:value=float(row['standard_value'])
            except (ValueError,TypeError):counts['missing_numeric']+=1;continue
            value=value/1000 if units=='nM' else value
            if not np.isfinite(value) or value<=0:counts['nonpositive_or_invalid']+=1;continue
            mol=rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(Chem.MolFromSmiles(smiles)))
            Chem.SanitizeMol(mol);Chem.GetSymmSSSR(mol)
            canonical=Chem.MolToSmiles(mol,isomericSmiles=True)
            durations=re.findall(r'after\s+(\d+(?:\.\d+)?)\s*(hrs?|hours?|days?)',desc,re.I)
            duration_hours=float(durations[0][0])*(24 if durations[0][1].lower().startswith('day') else 1) if len(durations)==1 else np.nan
            rows.append({'SMILES':canonical,'parent_connectivity':parent,
                         'scaffold':MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False) or 'ACYCLIC',
                         'Culture':cell,'duration_hours':duration_hours,'CC50_uM':value,'relation':relation,
                         'activity_id':aid,'assay_id':row['assay_chembl_id'],'doi':document.get('doi'),
                         'assay_description':desc,'source_url':f'https://www.ebi.ac.uk/chembl/api/data/activity/{aid}.json',
                         'snapshot_sha256':meta['sha256'],'source_file':meta['original_path'],'source_assay_type':row.get('assay_type')})
            admitted_from_source=True
        if admitted_from_source:snapshots.append({'path':meta['original_path'],'url':meta['url'],'sha256':meta['sha256']})
    all_rows=pd.DataFrame(rows)
    all_rows.to_csv(out/'toxicity_intervals.csv',index=False)
    exact=all_rows[all_rows.relation.eq('=')].copy()
    keep=[];conflicts=[]
    # Canonical stereoisomers are different measurements. Parent keys are used
    # for competition exclusion, not for incorrectly merging stereoselectivity.
    for (canonical,assay),group in exact.groupby(['SMILES','assay_id']):
        values=group.CC50_uM.to_numpy(float)
        if values.max()-values.min()>max(1e-6,.05*float(values.mean())):
            conflicts.append({'assay':assay,'activity_ids':group.activity_id.tolist(),'reason':'conflicting_exact_values_for_same_stereochemical_structure_and_assay'})
        else:keep.append(group.sort_values('activity_id').index[0])
    exact=exact.loc[keep].sort_values('activity_id').reset_index(drop=True)
    exact['sample_weight']=1./exact.groupby(['SMILES','Culture']).SMILES.transform('size')
    assert not set(exact.parent_connectivity)&forbidden
    assert exact.doi.notna().all() and exact.source_url.notna().all()
    exact.to_csv(out/'toxicity_exact.csv',index=False)
    report={'status':'curated_external_toxicity_auxiliary_pool','counts':dict(counts),
            'interval_rows':len(all_rows),'relation_counts':all_rows.relation.value_counts().to_dict(),
            'exact_rows_before_context_deduplication':int(all_rows.relation.eq('=').sum()),
            'exact_rows_retained':len(exact),'exact_unique_parents':exact.parent_connectivity.nunique(),
            'exact_unique_stereochemical_structures':exact.SMILES.nunique(),
            'source_documents':exact.doi.nunique(),'cell_counts':exact.Culture.value_counts().to_dict(),
            'conflicting_contexts_rejected':conflicts,'competition_parent_overlap':0,
            'SI_labels_created':False,'sample_weighting':'Each stereochemical structure/cell context has total weight 1; independent assay rows retained, identical assay duplicates deduplicated.',
            'exact_sha256':cp.sha(out/'toxicity_exact.csv'),'interval_sha256':cp.sha(out/'toxicity_intervals.csv'),
            'script_sha256':cp.sha(__file__),'sources':snapshots,
            'limitations':['Different durations and assay conditions remain; no cross-assay CC50/IC50 ratio is constructed.',
                          'Only exact CC50 values enter point regression; inequalities remain separate intervals.',
                          'Fold validation scaffolds must be excluded before auxiliary training.']}
    cp.save_json(out/'toxicity_provenance.json',report)
    print(json.dumps({k:report[k] for k in ['interval_rows','relation_counts','exact_rows_retained','exact_unique_parents','exact_unique_stereochemical_structures','source_documents','cell_counts','conflicting_contexts_rejected']},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.cache,args.output)
