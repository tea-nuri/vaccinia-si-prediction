import json
import csv
import hashlib
import io
import unittest
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from collect_literature import build
from literature import DATA, lookup_predictions, structure_key, strain_compatible, qualitative_active_flags


class LiteratureTests(unittest.TestCase):
    def test_committed_primary_data_and_hash_reproduce(self):
        rows = build()
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        generated = output.getvalue().encode('utf-8')
        self.assertEqual(generated, (DATA/'primary_literature.csv').read_bytes())
        report = json.loads((DATA/'primary_literature_validation.json').read_text())
        self.assertEqual(hashlib.sha256(generated).hexdigest(), report['sha256'])

    def test_bounds_endpoints_and_explicit_inactivity(self):
        rows = build()
        self.assertEqual(len(rows), 98)
        self.assertEqual(sum(r['retrieval_status']=='eligible' for r in rows),72)
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

    def test_2019_qualitative_evidence_is_opt_in_and_context_specific(self):
        p=pd.read_csv(DATA/'primary_literature.csv')
        p=p[p.doi.eq('10.1007/s11172-019-2467-6')]
        self.assertEqual(set(p.compound),{'1a','1b','1c','5a','5b','5c','6a','6b','6c'})
        self.assertTrue(p[['CC50','IC50','EC50','SI','SI_from_rounded_means']].isna().all().all())
        target=pd.DataFrame({'SMILES':p.smiles.tolist(),'Culture':'Vero','strain':'Vaccinia virus Copenhagen'})
        empty=pd.read_csv(DATA/'external.csv').iloc[:0]
        self.assertTrue(lookup_predictions(target,empty,p).prediction.isna().all())
        values=lookup_predictions(target,empty,p,include_2019=True).prediction.to_numpy()
        inactive=p.outcome.eq('Not active').to_numpy()
        np.testing.assert_array_equal(values[inactive],np.full(5,10000.))
        self.assertTrue(np.isnan(values[~inactive]).all())
        np.testing.assert_array_equal(qualitative_active_flags(target),~inactive)
        # An active qualitative result is not converted into a numerical SI.
        target['CC50_uM']=999999.;target['IC50_uM_VACV']=.000001
        np.testing.assert_array_equal(qualitative_active_flags(target),~inactive)
        target['Culture']='HeLa cells'
        self.assertFalse(qualitative_active_flags(target).any())
        self.assertTrue(lookup_predictions(target,empty,p,include_2019=True).prediction.isna().all())

    def test_excluded_stereochemical_series_is_not_used(self):
        primary=pd.read_csv(DATA/'primary_literature.csv')
        r=primary[(primary.doi=='10.1002/cmdc.202100771') & (primary.compound=='10h')].iloc[0]
        target=pd.DataFrame([dict(SMILES=r.smiles,Culture='Vero',strain='Copenhagen')])
        self.assertTrue(lookup_predictions(target,pd.read_csv(DATA/'external.csv').iloc[:0],primary).prediction.isna().all())


if __name__=='__main__':unittest.main()
