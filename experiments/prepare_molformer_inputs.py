"""Prepare only official training and filtered external SMILES for encoding."""
import argparse
from pathlib import Path
import sys
import pandas as pd
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import corrected_pipeline as cp
from collect_training_v2 import parent_key

def main(output, include_test=False):
    train,test=cp.read_inputs();source=cp.DATA/'external_training_candidate_v3.csv'
    external=pd.read_csv(source)
    forbidden=set(pd.concat([train.SMILES,test.SMILES]).map(parent_key))
    assert not set(external.SMILES.map(parent_key))&forbidden
    parts=[train.SMILES,external.SMILES]+([test.SMILES] if include_test else [])
    smiles=pd.concat(parts,ignore_index=True).tolist()
    result={'canonical_inputs':[Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=False) for s in smiles],
            'raw_inputs':smiles,'train_rows':len(train),'external_rows':len(external),'test_rows':len(test) if include_test else 0,
            'source_sha256':{'train':cp.TRAIN_SHA256,'external':cp.sha(source)}}
    if include_test:result['source_sha256']['test']=cp.sha(cp.DATA/'test.csv')
    output.parent.mkdir(parents=True,exist_ok=True);cp.save_json(output,result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-test',action='store_true',help='Append SMILES-only test inputs for frozen inference; no test fit or outcomes.')
    args=p.parse_args();main(args.output,args.include_test)
