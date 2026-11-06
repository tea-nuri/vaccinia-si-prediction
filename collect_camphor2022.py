"""Primary VACV Table 2, DOI 10.3390/molecules27154761.

Structures follow experimental systematic names, not competition labels.
Table 3 is Marburg pseudovirus and is deliberately excluded.
"""
import csv
import hashlib
import json
import re
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

ROOT = Path(__file__).resolve().parent
DOI = '10.3390/molecules27154761'
SOURCE = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9331314/'
XML_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9331314/fullTextXML'
XML_SHA256 = '1f015fe7ee344f26f267b6fb5a1b8cc7bc0a08cde7d3abf0c2456830e72efa99'
# Compound, mean CC50, IC50 or explicit outcome, published SI, HRMS ion formula.
# 2b is reported as [M-H]+; all other listed ions are molecular ions.
SPECS = [
    ('2a', 24.0, 6.7, 4.0, 'C17H23N3S'),
    ('2b', 200.0, 'NA', None, 'C14H22N3S'),
    ('2c', 'NT', 'NT', None, 'C13H23N3S'),
    ('2d', 150.0, 97.0, 2.0, 'C12H21N3S'),
    ('2′a', 16.0, 4.3, 4.0, 'C17H23N3S'),
    ('2′b', 35.0, 12.0, 3.0, 'C14H23N3S'),
    ('2′c', 45.0, 14.0, 3.0, 'C13H23N3S'),
    ('2′d', 130.0, 35.0, 5.0, 'C12H21N3S'),
    ('3a', 290.0, 'NA', None, 'C19H23ON3S'),
    ('3b', 99.0, 72.0, None, 'C16H23ON3S'),
    ('3c', 87.0, 24.0, 4.0, 'C15H23ON3S'),
    ('3d', 310.0, 100.0, 3.0, 'C14H21ON3S'),
    ('3′a', 290.0, 75.0, 4.0, 'C19H23ON3S'),
    ('3′b', 160.0, 66.0, 2.0, 'C16H23ON3S'),
    ('3′c', 220.0, 64.0, 4.0, 'C15H23ON3S'),
    ('3′d', 250.0, 'NA', None, 'C14H21ON3S'),
    ('4a', 270.0, 32.5, 8.0, 'C21H27ON3S'),
    ('4b', 'NT', 'NT', None, 'C18H27ON3S'),
    ('4c', 290.0, 'NA', None, 'C17H27ON3S'),
    ('4d', 310.0, 12.7, 24.0, 'C16H25ON3S'),
    ('5a', 229.0, 'NA', None, 'C25H26N3ClS'),
    ('5b', 19.0, 2.5, 8.0, 'C22H26N3ClS'),
    ('5c', 70.0, 12.0, 6.0, 'C21H26N3ClS'),
    ('5d', 144.0, 2.6, 55.0, 'C20H24N3ClS'),
    ('6a', 98.0, 'NA', None, 'C26H29N3S'),
    ('6b', 12.0, 4.0, 3.0, 'C23H29N3S'),
    ('6c', 15.0, 'NA', None, 'C22H29N3S'),
    ('6d', 17.0, 5.1, 3.0, 'C21H27N3S'),
    ('7a', 232.0, 'NA', None, 'C26H29ON3S'),
    ('7b', 200.0, 'NA', None, 'C23H29ON3S'),
    ('7c', 'NT', 'NT', None, 'C22H29ON3S'),
    ('7d', 'NT', 'NT', None, 'C21H27ON3S'),
    ('8a', 122.0, 18.0, 7.0, 'C25H27N3S'),
    ('8b', 17.0, 3.0, 6.0, 'C22H27N3S'),
    ('8c', 17.0, 3.0, 6.0, 'C21H27N3S'),
    ('8d', 14.0, 1.5, 9.0, 'C20H25N3S'),
]


def formula_counts(formula):
    return {atom:int(count or 1) for atom,count in re.findall(r'([A-Z][a-z]?)([0-9]*)',formula)}


def structure(compound):
    series = int(compound[0])
    substituent = {'a':'c4ccccc4','b':'CC=C','c':'CC','d':'C'}[compound[-1]]
    if series == 2:
        tail = '[N:20]NC(=S)N'+substituent
    elif series in [3,4]:
        tail = '[N:20][N:21]=[C:22]3SC'+('(C)(C)' if series==4 else '')+'C(=O)N3'+substituent
    else:
        para = {5:'(Cl)',6:'(C)',7:'(OC)',8:''}[series]
        tail = '[N:20][N:21]=[C:22]3SC=C(c5ccc'+para+'cc5)N3'+substituent
    mol = Chem.MolFromSmiles('CC1(C)[CH:4]2CC[C:1]1(C)[C:2](='+tail+')C2')
    indices = {a.GetAtomMapNum():a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    # Maps must be cleared BEFORE enumerating or assigning CIP priorities.
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    selected = []
    wanted = 'S' if '′' in compound else 'R'
    for candidate in EnumerateStereoisomers(mol, options=StereoEnumerationOptions(maxIsomers=64)):
        Chem.AssignStereochemistry(candidate, force=True, cleanIt=True)
        if any(candidate.GetAtomWithIdx(indices[n]).GetProp('_CIPCode') != wanted for n in [1,4]):
            continue
        if candidate.GetBondBetweenAtoms(indices[2],indices[20]).GetStereo() != Chem.BondStereo.STEREOE:
            continue
        if series != 2:
            expected = Chem.BondStereo.STEREOZ if series in [3,4] else Chem.BondStereo.STEREOE
            if candidate.GetBondBetweenAtoms(indices[21],indices[22]).GetStereo() != expected:
                continue
        selected.append(candidate)
    assert len(selected) == 1, (compound, len(selected))
    return selected[0]


def build():
    records = []
    for compound,cc,ic,si,ion_formula in SPECS:
        mol = structure(compound)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        expected = formula_counts(ion_formula)
        if compound == '2b':
            expected['H'] += 1
        assert formula_counts(formula) == expected, (compound,formula,ion_formula)
        measured = isinstance(ic,(int,float))
        records.append(dict(doi=DOI,compound=compound,smiles=Chem.MolToSmiles(mol),formula=formula,
            HRMS_ion_formula=ion_formula,HRMS_ion='[M-H]+' if compound=='2b' else '[M]+',
            CC50=cc if cc!='NT' else '',CC50_relation='=' if cc!='NT' else '',
            IC50=ic if measured else '',antiviral_relation='=' if measured else '',
            SI=si if si is not None else '',SI_relation='=' if si is not None else '',
            SI_from_rounded_means=cc/ic if measured else '',concentration_units='uM',
            cytotoxicity_type='CC50',antiviral_type='IC50',cell='Vero',strain='Copenhagen',duration_days=4,
            outcome='Measured' if measured else 'Not active' if ic=='NA' else 'Not tested',
            retrieval_status='excluded_not_tested' if ic=='NT' else 'eligible',match_scope='exact_stereo',
            source_url=SOURCE,publication_url='https://doi.org/'+DOI,source_location='Table 2; experimental systematic names; Table 1 substituents',
            structure_note='CIP 1/4 '+('S/S' if '′' in compound else 'R/R')+'; camphor imine E; '+('no second imine' if compound[0]=='2' else 'heterocycle imine '+('Z' if compound[0] in '34' else 'E')),
            quality_note='3b has numeric IC50 despite missing printed SI; it is not inactive. NA and NT are kept distinct. No competition columns read.'))
    return records


def main():
    rows = build()
    path = ROOT/'data/camphor2022.csv'
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    report = dict(records=len(rows),measured=sum(r['outcome']=='Measured' for r in rows),
        inactive=sum(r['outcome']=='Not active' for r in rows),not_tested=sum(r['outcome']=='Not tested' for r in rows),
        source_url=SOURCE,xml_url=XML_URL,reviewed_xml_sha256=XML_SHA256,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        caveat='Exact stereochemical matches only. Existing measurements are not overridden if studies disagree. Table 3 is not VACV.')
    (ROOT/'data/camphor2022_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
