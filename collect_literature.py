"""Independent transcription of primary literature; no competition data used."""
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

rows=[]
def add(doi,compound,smi,cc,ic,si,formula='',endpoint='IC50',cc_type='CC50',note=''):
    m=Chem.MolFromSmiles(smi)
    assert m is not None, (doi,compound,smi)
    actual=rdMolDescriptors.CalcMolFormula(m)
    if formula: assert actual==formula,(compound,actual,formula)
    rows.append(dict(doi=doi,compound=compound,smiles=Chem.MolToSmiles(m),formula=actual,
                     cytotoxicity=cc,antiviral=ic,SI_published=si,antiviral_type=endpoint,
                     cytotoxicity_type=cc_type,cell='Vero',strain='Copenhagen',
                     publication_url='https://doi.org/'+doi,source_location='Table 1',
                     structure_note=note or 'Transcribed from chemical name and scheme; formula checked where published.'))

ad='C12CC3CC(CC(C3)C1)C2'
iso='c1noc(-c2ccccc2)c1'
tol='c1noc(-c2ccc(C)cc2)c1'
nit='c1noc(-c2ccc([N+](=O)[O-])cc2)c1'
thio='c1nsc(Cl)c1Cl'
doi='10.3390/chemistry7020034'
rim=lambda n:'CC('+n+')'+ad
specs=[
('1','O=C(N'+ad+')c1cccnc1',100.67,16.78,6,'C16H20N2O'),
('2','O=C(N'+ad+')c1ccncc1',390.33,3.45,115,'C16H20N2O'),
('3','O=C(N'+ad+')c1cc(-c2ccccc2)nc2ccccc12',350.31,'NA',None,'C26H26N2O'),
('4','O=C(N'+ad+')'+thio,75.50,2.91,26,'C14H16Cl2N2OS'),
('5',rim('N=C'+iso),299.4,75.53,4,'C22H26N2O'),
('6',rim('N=C'+tol),287.35,'NA',None,'C23H28N2O'),
('7',rim('N=Cc1ccc(OC'+tol.replace('1','4')+')cc1'),66.08,40.8,2,'C30H34N2O2'),
('8',rim('N=Cc1cc(OC)c(OC'+thio.replace('1','4')+')cc1'),52.31,'NA',None,'C24H28Cl2N2O2S'),
('9',rim('N=Cc1cc(OC)c(OC(=O)C23CC4CC(CC(C4)C2)C3)cc1'),56.85,'NA',None,'C31H41NO3'),
('10',rim('NC'+iso),15.47,13.91,None,'C22H28N2O'),
('11',rim('NC'+tol),15.14,'NA',None,'C23H30N2O'),
('12',rim('NC'+nit),204.72,9.44,22,'C22H27N3O3'),
('13',rim('NCc1ccc(OC'+tol.replace('1','4')+')cc1'),10.47,'NA',None,'C30H36N2O2'),
('14',rim('N(C(=O)'+thio+')Cc1ccc(OC(=O)'+thio.replace('1','4')+')cc1'),8.87,'NA',None,''),
('15',rim('N(C(=O)'+thio+')Cc1cc(OC)c(OC'+thio.replace('1','4')+')cc1'),21.31,'NA',None,'C28H29Cl4N3O3S2'),
('16',rim('N(C(=O)'+tol+')Cc1cccc(OC(=O)'+tol.replace('1','4')+')c1'),150.37,'NA',None,'C41H41N3O5')]
for compound,smi,cc,ic,si,formula in specs:
    add(doi,compound,smi,cc,ic,si,formula,endpoint='EC50')

# Section 3.2 establishes 12..15 = H,4-F,4-Cl,4-Br. Section 2.1 gives
# 16..18 = 2-NO2,3-NO2,4-NO2. Table row images establish a/b order.
values=[(105.67,4.7,23),(101.33,6.73,15),(347.31,.41,847),(104,2.14,49),
        (346,.13,2662),(104,.18,578),(415,.13,3192),(129.67,.22,589),
        ('>100','NA',None),('>100','NA',None),(7.75,.95,8),(9.5,.99,10),
        (546.43,.03,18214),(492,.04,12300)]
phenyl=['c1ccccc1','c1ccc(F)cc1','c1ccc(Cl)cc1','c1ccc(Br)cc1',
        'c1ccccc1[N+](=O)[O-]','c1cccc([N+](=O)[O-])c1','c1ccc([N+](=O)[O-])cc1']
for j,ar in enumerate(phenyl):
    for k,adam in enumerate([ad,'C1C2CC3CC1CC(C2)C3']):
        add('10.3390/v15010029',str(12+j)+'ab'[k],'O=C(N'+adam+')'+ar,
            *values[j*2+k],cc_type='TC50',note='Scheme 1 and Table 1 structure; positions confirmed in sections 2.1 and 3.2.')

# Stereochemistry is assigned from the published systematic names, with CIP
# checked at numbered scaffold atoms. It is never inferred from a dataset label.
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
def amide(ar, scaffold, cips):
    if scaffold=='bornyl':base='[C:1]12(C)[CH:2](NC(=O)'+ar+')[CH2:3][CH:4](CC1)C2(C)C'
    else:base='[C:1]12(C)[CH:2](NC(=O)'+ar+')[C:3](C)(C)[CH:4](CC1)C2'
    m=Chem.MolFromSmiles(base)
    matches=[]
    for iso in EnumerateStereoisomers(m,options=StereoEnumerationOptions(tryEmbedding=False)):
        # Atom maps do not enter chemical CIP priorities.
        atom_index={a.GetAtomMapNum():a.GetIdx() for a in iso.GetAtoms() if a.GetAtomMapNum()}
        for a in iso.GetAtoms():a.SetAtomMapNum(0)
        Chem.AssignStereochemistry(iso,cleanIt=True,force=True)
        got=tuple(iso.GetAtomWithIdx(atom_index[k]).GetProp('_CIPCode') for k in [1,2,4])
        if got==cips:matches.append(Chem.MolToSmiles(iso))
    assert len(matches)==1,(base,cips,matches)
    return matches[0]

ars=['c3ccccc3','c3ccc(Cl)cc3','c3ccc(Br)cc3','c3ccc(F)cc3',
     'c3ccc([N+](=O)[O-])cc3','c3ccc(C(F)(F)F)cc3','c3ccc(OC)cc3',
     'c3ccc(C)cc3','c3ccncc3','c3cccnc3','c3cnccn3','C3CCCCC3','C3CCC(F)(F)CC3']
sets={
7:('bornyl',('R','R','R'),[(196.4,15.2,13),(1371.7,.6,2326),(312.2,1.4,219),
 (58.1,14.9,3),(626.1,.9,696),(37.5,.6,62),(52.2,7.2,7),(47.9,2.6,18),
 (260.7,17.4,15),(104.5,6.8,15),(170.9,74.1,2),(542.9,4.3,126),('>334.0','NA',None)]),
9:('fenchyl',('R','S','S'),[('>388.5','NA',None),(2158.9,13.6,159),(993.3,5.6,177),
 ('>363.2',31.8,'>11'),(1447.6,5.6,259),(1072.6,3.3,328),(352.5,18.6,19),(1510.7,18.9,79),
 (288,'NA',None),('>387.1','NA',None),(217.5,'NA',None),('>379.6','NA',None),(1155.6,26.4,43)]),
10:('fenchyl',('R','R','S'),[(217.2,'NA',None),(80.2,'NA',None),(303.6,17.8,17),
 ('>363.2','NA',None),(356.2,19.7,18),(50.1,12.1,4),(224.8,20.9,11),('>368.5','NA',None),
 ('>387.1','NA',None),('>387.1','NA',None),(207.8,'NA',None),('>379.6','NA',None),('>334','NA',None)])}
for series,(scaffold,cips,values) in sets.items():
    for j,(ar,v) in enumerate(zip(ars,values)):
        add('10.1002/cmdc.202100771',str(series)+chr(97+j),amide(ar,scaffold,cips),*v,
            note='Accepted manuscript Table 1 and systematic name in experimental section; exact CIP '+str(cips)+' at scaffold positions 1,2,4.')
# 8b/c/e names in the accepted manuscript conflict with its common letter code.
# Retain only unambiguous 8l here; do not silently resolve conflicts using labels.
add('10.1002/cmdc.202100771','8l',amide(ars[11],'bornyl',('R','S','R')),1388.7,1.4,1093,
    note='Unambiguous systematic name and Table 1; excluded 8b/c/e pending source-label resolution.')

# 8e identity is independently corroborated by review compound 73 (ChEMBL
# CHEMBL5410209, activity 25566298: same VACV 0.03, CPXV 1.4, ECTV 0.41).
# The accepted manuscript's experimental headings cyclically mislabel 8b/c/e.
# Keep this correction explicit, with both published SI and ratio retained.
add('10.1002/cmdc.202100771','8e',amide(ars[4],'bornyl',('R','S','R')),1082,.03,31500,
    note='Table 1 common substituent code e=4-NO2. Experimental heading says 4-Cl (inconsistent). Structure and all three antiviral endpoints independently match review compound 73, DOI 10.1021/acs.jmedchem.3c00069, ChEMBL CHEMBL5410209/activity 25566298. Published SI 31500 differs from ratio of rounded means 36066.67; retain both.')

# Primary table and title specify 4-METHYL imidazoles. The ChEMBL record
# represents them as 4-HYDROXY compounds, a different molecular graph.
aryl=['c2ccccc2O','c2ccccc2','c2ccccc2[N+](=O)[O-]',
      'c2cccc([N+](=O)[O-])c2','c2ccc([N+](=O)[O-])cc2','c2ccc(C(F)(F)F)cc2']
esters=[(1825.57,15.53,117),(171.54,'NA',None),(343.30,'NA',None),
        (374.91,'NA',None),(45.70,.45,102),(321.97,.35,919)]
amides=[(196.89,15.02,13),(298.90,'NA',None),(436.26,73.37,6),('>389.11','NA',None),
        ('>314.47','NA',None),('>331.13','NA',None),(164.47,17.64,9),(260.93,'NA',None),
        (200.63,3.02,66),(356.95,10.73,33),(26.10,3.4,8),(201.85,20.58,10)]
for j,ar in enumerate(aryl):
    add('10.1016/j.bmcl.2022.129080','1'+chr(97+j),'CCOC(=O)c1c(C)nc(-'+ar+')n1O',*esters[j],
        note='Table 1, Scheme 1: 1-hydroxy-4-methylimidazole scaffold; ChEMBL 4-OH transcription is not used.')
    for k,amide_group in enumerate(['CCCCN','N(C3CC3)']):
        add('10.1016/j.bmcl.2022.129080','2'+chr(97+2*j+k),amide_group+'C(=O)c1c(C)nc(-'+ar+')n1O',*amides[2*j+k],
            note='Table 1, Scheme 1: 1-hydroxy-4-methylimidazole; R1/R2 taken from Table 1. ChEMBL 4-OH transcription is not used.')



def build():
    """Normalize manually transcribed facts without reading competition files."""
    output=[]
    for record in rows:
        r=record.copy()
        def measured(value):
            if value is None or value=='NA': return '', ''
            if isinstance(value,str) and value.startswith('>'): return float(value[1:]), '>'
            return float(value), '='
        cc,ccrel=measured(r.pop('cytotoxicity'))
        av,avrel=measured(r.pop('antiviral'))
        si,sirel=measured(r.pop('SI_published'))
        r.update(CC50=cc, CC50_relation=ccrel, IC50=av if r['antiviral_type']=='IC50' else '',
                 EC50=av if r['antiviral_type']=='EC50' else '', antiviral_relation=avrel,
                 SI=si, SI_relation=sirel, concentration_units='uM',
                 outcome='Not active' if av=='' else 'Measured',
                 SI_from_rounded_means=cc/av if cc and av and ccrel==avrel=='=' else '')
        # A TC50 is stored in the cytotoxicity column with its original type,
        # as in the existing collector. An EC50 never masquerades as an IC50.
        r['match_scope']='exact_stereo'
        if r['doi']=='10.3390/v15010029':r['match_scope']='achiral_adamantane'
        if r['doi']=='10.3390/chemistry7020034':r['match_scope']='published_stereoisomer_mixture'
        r['retrieval_status']='eligible'
        if r['doi']=='10.1002/cmdc.202100771' and r['compound'].startswith(('9','10')):
            r['retrieval_status']='excluded_pending_stereochemical_source_review'
        r['source_url']=r['publication_url']
        if r['doi']=='10.1002/cmdc.202100771':
            r['source_url']='https://www.researchgate.net/publication/359789665'
        if r['doi']=='10.3390/v15010029':
            r['source_url']='https://pmc.ncbi.nlm.nih.gov/articles/PMC9865139/'
        if r['doi']=='10.1016/j.bmcl.2022.129080':
            r['source_url']='https://pmc.ncbi.nlm.nih.gov/articles/PMC9674569/'
        r['quality_note']=''
        if r['SI'] and r['SI_from_rounded_means'] and abs(r['SI']-r['SI_from_rounded_means'])>max(1.,.05*r['SI']):
            r['quality_note']='Published SI differs from ratio of rounded concentration means; both retained.'
        output.append(r)
    return output


def main():
    import csv, hashlib
    data=Path(__file__).resolve().parent/'data'
    records=build()
    with (data/'primary_literature.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    report={
        'records':len(records),'papers':len({r['doi'] for r in records}),
        'eligible_records':sum(r['retrieval_status']=='eligible' for r in records),
        'explicit_inactive_records':sum(r['outcome']=='Not active' for r in records),
        'method':'Manual primary-table transcription and chemical-name/scheme translation; RDKit validates graph, formula and named CIP. No competition files are read.',
        'source_locations':{doi:sorted({r['source_url'] for r in records if r['doi']==doi}) for doi in sorted({r['doi'] for r in records})},
        'notes':[
            'Primary tables, compound identifiers, endpoint names and bounds are preserved. NA means explicitly not active per each table footnote.',
            'Chemistry 2025 reports stereoisomer mixtures; graph-only matching is confined to this source and achiral adamantane compounds.',
            'Camphor 8e: accepted-manuscript experimental headings conflict with common substituent code. Nitro structure and VACV/CPXV/ECTV endpoints corroborated by CHEMBL5410209 activities 25566298/25566310/25566275 from DOI 10.1021/acs.jmedchem.3c00069.',
            'Fenchyl series 9/10 remains recorded but excluded from retrieval pending resolution of stereochemical assignment and observed source/competition discrepancies.',
            'BMCL 2022.129080: primary structures have 4-methyl groups; the ChEMBL records encode 4-hydroxy groups. Primary graph is used only in this separate curated dataset.',
            'Published SI and ratio of rounded external means can differ. Retrieval experiment explicitly evaluates the latter; never derives SI from competition concentrations.'
        ],
        'sha256':hashlib.sha256((data/'primary_literature.csv').read_bytes()).hexdigest()}
    (data/'primary_literature_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
