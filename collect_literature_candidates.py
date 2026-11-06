"""Audited additional primary measurements; independent of competition files.

Kept separate from the frozen V7 sources until a candidate is selected.
"""
import csv,json,hashlib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors,rdMolDescriptors
ROOT=Path(__file__).resolve().parent
DOI='10.1007/s11172-019-2467-6'
SOURCE='https://www.lookchem.com/FreePDFArticle_5896-17-3_76541533.htm'

def measured(x):
    if x in [None,'NA','NT']:return '', ''
    if isinstance(x,str) and x[0] in '<>':return float(x[1:]),x[0]
    return float(x),'='

def build():
    ar='c2ccccc2O'
    base=lambda ac,aryl=ar:ac+'c1c(C)nc(-'+aryl+')n1O'
    fused=lambda n1,n2:'CC1(C)Cc2'+n1+'c(-'+ar.replace('2','3')+')'+n2+'c2C(=O)C1'
    specs=[
        ('1a',base('CC(=O)'),64.80,14.58,4.39,'C12H12N2O3'),
        ('1b',base('CCOC(=O)'),478.33,4.07,118.53,'C13H14N2O4'),
        ('1c',fused('n','n(O)'),203.,1.29,159.73,'C15H16N2O3'),
        ('1d','O=C(Nc2ccccc2)c1c(-c2ccccc2)nc(-'+ar+')n1O',.201,'NA',None,'C22H17N3O3'),
        ('1e',base('CC(=O)','c2cc(Cl)ccc2O'),1.06,'NA',None,'C12H11ClN2O3'),
        ('1f',base('CCOC(=O)','c2ccc(OCc3ccccc3)cc2O'),1.434,'NA',None,'C20H20N2O5'),
        ('1g','O=C(Nc2ccccc2)c1c(-c2ccccc2)nc(-c2cc(Cl)ccc2O)n1O','<0.16','NA',None,'C22H16ClN3O3'),
        ('1h','O=C(Nc2ccccc2)c1c(-c2ccccc2)nc(-c2cc(Br)ccc2O)n1O','<0.16','NA',None,'C22H16BrN3O3'),
        ('4a','CC(=O)c1c(C)n(C)c(-'+ar+')[n+]1[O-]','>100','NA',None,'C13H14N2O3'),
        ('4b','CCOC(=O)c1c(C)n(C)c(-'+ar+')[n+]1[O-]','>100','NA',None,'C14H16N2O4'),
        ('4c',fused('n(C)','[n+]([O-])'),'>100','NA',None,'C16H18N2O3'),
        ('5a',base('CC(=O)')+'C',56.061,'NA',None,'C13H14N2O3'),
        ('5b',base('CCOC(=O)')+'C','>100',42.926,'>2.32','C14H16N2O4'),
        ('5c',fused('n','n(OC)'),'>100','NA',None,'C16H18N2O3'),
        ('6a','CC(=O)c1c(C)nc(-'+ar+')[nH]1','>100','NA',None,'C12H12N2O2'),
        ('6b','CCOC(=O)c1c(C)nc(-'+ar+')[nH]1',5.231,'NA',None,'C13H14N2O3'),
        ('6c',fused('n','[nH]'),'>100','NA',None,'C15H16N2O2')]
    records=[]
    for compound,smiles,cc,av,si,formula in specs:
        mol=Chem.MolFromSmiles(smiles);assert mol is not None
        actual=rdMolDescriptors.CalcMolFormula(mol);assert actual==formula,(compound,actual,formula)
        mw=Descriptors.MolWt(mol);factor=1000/mw
        ccval,ccrel=measured(cc);avval,avrel=measured(av);sival,sirel=measured(si)
        records.append(dict(doi=DOI,compound=compound,smiles=Chem.MolToSmiles(mol),formula=actual,
            molecular_weight=mw,CC50=ccval*factor if ccval!='' else '',CC50_relation=ccrel,
            IC50=avval*factor if avval!='' else '',antiviral_relation=avrel,SI=sival,SI_relation=sirel,
            concentration_units='uM',original_cytotoxicity=cc,original_antiviral=av,
            original_concentration_units='ug/mL',unit_conversion='uM = (ug/mL)*1000/molecular_weight',
            cytotoxicity_type='TC50',antiviral_type='IC50',cell='Vero',strain='Copenhagen',
            outcome='Not active' if av=='NA' else ('Not tested' if av=='NT' else 'Measured'),
            SI_from_rounded_means=ccval/avval if ccrel==avrel=='=' and avval>0 else '',
            match_scope='exact_stereo',retrieval_status='eligible',source_url=SOURCE,
            publication_url='https://doi.org/'+DOI,source_location='Table 1' if compound.startswith('1') else 'Table 2',
            structure_note='Scheme 1-3 and experimental chemical names; source page tables and schemes visually checked. No label-based structure assignment.',
            quality_note='Published SI and ratio of mean concentrations retained separately; 5b is measured with a bound, never inactive.'))
    return records

def main():
    rows=build();p=ROOT/'data/literature_candidates.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    report=dict(records=len(rows),papers=1,source_url=SOURCE,publication_url='https://doi.org/'+DOI,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                explicit_inactive_records=sum(r['outcome']=='Not active' for r in rows),
                exact_numeric_records=sum(r['SI_from_rounded_means']!='' for r in rows),
                note='Manually transcribed primary tables; no competition files are read. V7 sources remain frozen. Mass concentrations converted with RDKit molecular weight. All bounds preserved.')
    (ROOT/'data/literature_candidates_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
