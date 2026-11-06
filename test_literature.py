import json
import unittest
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from collect_literature import build
from literature import DATA, lookup_predictions, structure_key, strain_compatible


class LiteratureTests(unittest.TestCase):
    def test_bounds_endpoints_and_explicit_inactivity(self):
        rows = build()
        self.assertEqual(len(rows), 89)
        self.assertEqual(sum(r['retrieval_status']=='eligible' for r in rows),63)
        for r in rows:
            if r['antiviral_type']=='EC50':
                self.assertEqual(r['IC50'],'')
            if r['CC50_relation']=='>':
                self.assertEqual(r['SI_from_rounded_means'],'')
        measured = next(r for r in rows if r['doi']=='10.3390/chemistry7020034' and r['compound']=='10')
        self.assertEqual(measured['SI'],'')
        self.assertEqual(measured['outcome'],'Measured')
        self.assertAlmostEqual(measured['SI_from_rounded_means'],15.47/13.91)

    def test_primary_structures_against_independent_evidence_and_formula(self):
        rows=build()
        lead=next(r for r in rows if r['doi']=='10.1002/cmdc.202100771' and r['compound']=='8e')
        evidence=json.loads((DATA/'primary_structure_evidence.json').read_text())['activities']
        for r in evidence:
            self.assertEqual(lead['smiles'],structure_key(r['canonical_smiles']))
        vacv=next(r for r in evidence if r['target_organism']=='Orthopoxvirus vaccinia')
        self.assertAlmostEqual(lead['IC50'],float(vacv['standard_value'])/1000)
        ester=next(r for r in rows if r['doi']=='10.1016/j.bmcl.2022.129080' and r['compound']=='1a')
        self.assertEqual(ester['formula'],'C13H14N2O4')
        self.assertAlmostEqual(Descriptors.MolWt(Chem.MolFromSmiles(ester['smiles'])),262.26,delta=.1)

    def test_stereoisomers_context_and_forbidden_columns(self):
        primary=pd.read_csv(DATA/'primary_literature.csv')
        rows=primary[(primary.doi=='10.1002/cmdc.202100771') & primary.compound.isin(['7e','8e'])].copy()
        target=pd.DataFrame({'SMILES':rows.smiles.tolist(), 'Culture':['Vero']*2,
                             'strain':['Vaccinia virus Copenhagen']*2})
        numerical=pd.read_csv(DATA/'external.csv').iloc[:0]
        result=lookup_predictions(target,numerical,rows)
        np.testing.assert_allclose(result.prediction,[626.1/.9,1082/.03])
        target['CC50_uM']=[1e20,1e20];target['IC50_uM_VACV']=[0,0]
        np.testing.assert_allclose(lookup_predictions(target,numerical,rows).prediction,result.prediction)
        target['Culture']='HeLa cells'
        self.assertTrue(lookup_predictions(target,numerical,rows).prediction.isna().all())
        target['Culture']='Vero';target['strain']='Vaccinia virus Lister'
        self.assertTrue(lookup_predictions(target,numerical,rows).prediction.isna().all())
        self.assertFalse(strain_compatible('WR','Vaccinia virus Copenhagen'))
        self.assertTrue(strain_compatible('WR','Vaccinia virus Western Reserve'))

    def test_excluded_stereochemical_series_is_not_used(self):
        primary=pd.read_csv(DATA/'primary_literature.csv')
        r=primary[(primary.doi=='10.1002/cmdc.202100771') & (primary.compound=='10h')].iloc[0]
        target=pd.DataFrame([dict(SMILES=r.smiles,Culture='Vero',strain='Copenhagen')])
        self.assertTrue(lookup_predictions(target,pd.read_csv(DATA/'external.csv').iloc[:0],primary).prediction.isna().all())


if __name__=='__main__':unittest.main()
