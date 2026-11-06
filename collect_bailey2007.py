"""Rebuild bounded external measurements; never turn bounds into exact SI labels."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path

from rdkit import Chem

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'


def build_rows():
    meta=json.loads((DATA/'bailey2007_sources.json').read_text())['activities.json']
    with zipfile.ZipFile(DATA/'bailey2007_snapshot.zip') as archive:
        raw=archive.read('activities.json')
    if hashlib.sha256(raw).hexdigest()!=meta['sha256']:
        raise ValueError('External snapshot integrity check failed')
    response=json.loads(raw)
    records=response['activities']
    if response['page_meta']['total_count']!=len(records) or len(records)!=34:
        raise ValueError('Incomplete source pagination')
    pairs={}
    for r in records:
        if r['document_chembl_id']!='CHEMBL1142975' or r.get('data_validity_comment') not in [None,'','Manually validated']:
            raise ValueError('Unexpected document or invalid measurement')
        assay=r['assay_chembl_id']
        if assay not in ['CHEMBL918571','CHEMBL918574']:
            raise ValueError('Unexpected assay')
        kind='EC50' if assay=='CHEMBL918571' else 'CC50'
        if r['standard_type']!=kind or r['standard_units']!='nM':
            raise ValueError('Endpoint/unit mismatch')
        if kind in pairs.setdefault(r['molecule_chembl_id'],{}):
            raise ValueError('Ambiguous measurement pair')
        pairs[r['molecule_chembl_id']][kind]=r
    rows=[]
    for molecule,pair in sorted(pairs.items()):
        ec,cc=pair['EC50'],pair['CC50']
        smi=Chem.MolToSmiles(Chem.MolFromSmiles(ec['canonical_smiles']),isomericSmiles=True)
        if smi!=Chem.MolToSmiles(Chem.MolFromSmiles(cc['canonical_smiles']),isomericSmiles=True):
            raise ValueError('Paired structures differ')
        ev,cv=float(ec['standard_value'])*.001,float(cc['standard_value'])*.001
        if min(ev,cv)<=0:
            raise ValueError('Nonpositive measurement')
        eligible=ec['standard_relation']=='=' and cc['standard_relation'] in ['>','>=']
        rows.append(dict(molecule_id=molecule,smiles=smi,doi='10.1021/jm061484y',
            cell='BSC40',strain='IHD-J',EC50=ev,EC50_relation=ec['standard_relation'],
            CC50=cv,CC50_relation=cc['standard_relation'],units='uM',
            reported_cc50_ec50_limit=cv/ev if eligible else '',ratio_relation=cc['standard_relation'] if eligible else '',
            feature_eligible=eligible,EC50_activity_id=ec['activity_id'],CC50_activity_id=cc['activity_id'],
            EC50_assay_id=ec['assay_chembl_id'],CC50_assay_id=cc['assay_chembl_id'],
            publication_url='https://pubmed.ncbi.nlm.nih.gov/17335190/',source_url=meta['url'],
            interpretation='Bounded experimental feature, not an exact SI label; source has partially specified stereochemistry.'))
    return rows


def main():
    rows=build_rows()
    with (DATA/'bailey2007_bounds.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print('External records:',len(rows),'eligible features:',sum(r['feature_eligible'] for r in rows))


if __name__=='__main__':
    main()
