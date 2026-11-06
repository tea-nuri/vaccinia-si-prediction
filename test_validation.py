import copy
import json
import unittest
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from collect_data import measurement, build, build_inactivity, inactive_field
from train import fit_predict, ALLOWED, apply_inactivity_evidence, prepare_external, augment, apply_known_compounds

ROOT = Path(__file__).resolve().parent


class DataValidationTests(unittest.TestCase):
    def test_equivalent_smiles_use_observation_only_in_matching_context(self):
        training = pd.DataFrame([dict(SMILES='CCO',strain='VACV (Copenhagen)',Culture='Vero')])
        target = pd.DataFrame([
            dict(SMILES='OCC',strain='vacv copenhagen',Culture='VERO'),
            dict(SMILES='OCC',strain='VACV WR',Culture='Vero'),
            dict(SMILES='OCC',strain='VACV (Copenhagen)',Culture=None)])
        result, info = apply_known_compounds(training,np.array([79.931]),target,np.array([9000.,9000.,9000.]))
        np.testing.assert_allclose(result,[79.931,9000.,9000.])
        self.assertEqual(info['corrected_positions'],[0])

    def test_conflicting_observations_and_other_stereoisomers_fall_back(self):
        training = pd.DataFrame([
            dict(SMILES='CCO',strain='VACV',Culture='Vero'),
            dict(SMILES='OCC',strain='VACV',Culture='Vero'),
            dict(SMILES='C[C@H](O)C(=O)O',strain='VACV',Culture='Vero')])
        target = pd.DataFrame([
            dict(SMILES='CCO',strain='VACV',Culture='Vero'),
            dict(SMILES='C[C@@H](O)C(=O)O',strain='VACV',Culture='Vero')])
        result, info = apply_known_compounds(training,np.array([80.,10000.,5.]),target,np.array([50.,50.]))
        np.testing.assert_array_equal(result,[50.,50.])
        self.assertEqual(info['conflicting_contexts'],1)

    def test_inactivity_requires_explicit_antiviral_evidence(self):
        self.assertIsNone(inactive_field({'standard_type':'IC50', 'standard_value':None}))
        self.assertIsNone(inactive_field({'standard_type':'CC50', 'standard_text_value':'Not active'}))
        self.assertEqual(inactive_field({'standard_type':'IC50', 'standard_text_value':'Not active'}),
                         'standard_text_value')

    def test_culture_mapping_uses_training_support_and_rejects_disagreement(self):
        train = pd.DataFrame({'SMILES':['C','CC','CCC'], 'Culture':['A']*3})
        target = pd.DataFrame({'SMILES':['O','O','N'], 'Culture':['A','B','A']})
        evidence = {'C','CC','CCC','O'}
        baseline = np.array([7000.,7000.,7000.])
        pred, _ = apply_inactivity_evidence(train, np.array([10000]*3), target, baseline, evidence)
        np.testing.assert_array_equal(pred, [10000,7000,7000])
        pred, _ = apply_inactivity_evidence(train, np.array([200,10000,10000]), target, baseline, evidence)
        np.testing.assert_array_equal(pred, baseline)
        pred, _ = apply_inactivity_evidence(train.iloc[:2], np.array([10000]*2), target, baseline, evidence)
        np.testing.assert_array_equal(pred, baseline)

    def test_inactivity_snapshot_and_multiple_source_guard(self):
        with zipfile.ZipFile(ROOT/'data/vacv_evidence_snapshot.zip') as z:
            payloads = {n: json.loads(z.read(n)) for n in z.namelist()}
        records = [r for n, obj in payloads.items() if n.startswith('activity_') for r in obj['activities']]
        documents = {obj['document_chembl_id']: obj for n,obj in payloads.items() if n.startswith('document_')}
        rows = build_inactivity(records, documents)
        self.assertEqual(len(rows), 1116)
        self.assertTrue(all(r['doi'] and r['outcome']=='Not active' for r in rows))
        sample = next(r for r in records if inactive_field(r))
        changed = copy.deepcopy(sample)
        changed.update(document_chembl_id='ANOTHER_DOCUMENT', activity_comment=None,
                       standard_text_value=None, text_value=None)
        altered = build_inactivity(records+[changed], documents)
        affected = [r for r in altered if r['molecule_id']==sample['molecule_chembl_id']]
        self.assertTrue(affected)
        self.assertTrue(all(not r['single_source'] for r in affected))

    def test_units_and_bounds_remain_distinct(self):
        r = dict(standard_type='IC50', standard_value='1200', standard_units='nM', standard_relation='>')
        value, relation, issue = measurement(r, 'ic')
        self.assertEqual(value, 1.2)
        self.assertEqual(relation, '>')
        self.assertEqual(issue, '')
        r['standard_units'] = 'ug.mL-1'
        self.assertEqual(measurement(r, 'ic')[2], 'unsupported_units')

    def test_selected_submission_is_reproducible(self):
        train = pd.read_csv(ROOT/'data/train.csv', usecols=ALLOWED+['SI VACV'])
        test = pd.read_csv(ROOT/'data/test.csv', usecols=['index']+ALLOWED)
        reference = pd.read_csv(ROOT/'results/submission.csv')
        pred, _ = fit_predict(train, train['SI VACV'].to_numpy(), test)
        np.testing.assert_array_equal(test['index'], reference['index'])
        np.testing.assert_allclose(pred, reference['SI VACV'], rtol=1e-10, atol=1e-6)

    def test_snapshot_is_complete_and_assay_pairs_are_unambiguous(self):
        with zipfile.ZipFile(ROOT/'data/chembl_snapshot.zip') as z:
            payloads = {n: json.loads(z.read(n)) for n in z.namelist()}
        rows = build(payloads)
        self.assertEqual(len(rows), 221)
        self.assertEqual(len({r['doi'] for r in rows}), 9)
        self.assertEqual(sum(r['usable_for_training'] for r in rows), 114)
        self.assertTrue(all(not r['usable_for_training'] for r in rows if r['SI']==''))
        self.assertTrue(all(r['doi'] for r in rows))
        self.assertTrue(all(r['SI_relation']=='=' for r in rows if r['usable_for_training']))
        self.assertTrue(all(not r['usable_for_training'] for r in rows
                            if any(r[c+'_relation']!='=' for c in ['IC50','CC50','SI'])))
        # A duplicate result for the same assay/molecule must fail rather than
        # silently pick one and create a fabricated ratio.
        changed = copy.deepcopy(payloads)
        paper = changed['paper_CHEMBL6092128.json']
        si = next(r for r in paper['activities'] if r['assay_chembl_id']=='CHEMBL6092369')
        paper['activities'].append(si)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            build(changed)

    def test_external_rows_cannot_reintroduce_validation_molecules(self):
        frame = pd.DataFrame([
            dict(smiles=s,SI=si,IC50=1.,CC50=si,IC50_relation='=',CC50_relation='=',SI_relation='=',
                 quality_flags='',cell='Vero',strain='Copenhagen',doi='example')
            for s,si in [('CCO',10.),('CCO.Cl',12.),('CCC',20.),('CCN',10000.)]])
        external = prepare_external(frame,['OCC'])
        self.assertEqual(external.canonical.tolist(),['CCC'])
        original = pd.DataFrame([dict(SMILES='CCO',strain='Copenhagen',Culture='Vero')])
        joined,labels,weights,rows = augment(original,np.array([10.]),external,.3,{'ACYCLIC'})
        self.assertEqual(len(joined),1)
        self.assertTrue(rows.empty)
        self.assertIsNone(weights)

    def test_repeated_external_measurements_share_compound_weight(self):
        external = pd.DataFrame([dict(SMILES='CCC',canonical='CCC',scaffold='ACYCLIC',
            strain='Copenhagen',Culture='Vero',**{'SI VACV':20.})]*2)
        original = pd.DataFrame([dict(SMILES='CCO',strain='Copenhagen',Culture='Vero')])
        _,_,weights,_ = augment(original,np.array([10.]),external,.3)
        np.testing.assert_allclose(weights,[1.,.15,.15])


    def test_curated_tc50_and_missing_si(self):
        with zipfile.ZipFile(ROOT/'data/chembl_snapshot.zip') as z:
            payloads = {n: json.loads(z.read(n)) for n in z.namelist()}
        rows = build(payloads)
        camphor = [r for r in rows if r['document_id'] == 'CHEMBL4257560']
        self.assertTrue(any(r['usable_for_training'] for r in camphor))
        self.assertTrue(all(r['CC50_source_type'] == 'TC50' for r in camphor if r['CC50_activity_id']))
        self.assertTrue(any(r['SI'] == '' for r in camphor))
        r = dict(standard_type='TC50', standard_value='1', standard_units='uM', standard_relation='=')
        self.assertEqual(measurement(r, 'cc')[2], 'wrong_endpoint')
        self.assertEqual(measurement(r, 'cc', 'TC50'), (1., '=', ''))
        # The CV1 source has inconsistent published ratios; do not train on them.
        cv1 = [r for r in rows if r['document_id'] == 'CHEMBL4052722']
        self.assertTrue(all(not r['usable_for_training'] for r in cv1))


if __name__ == '__main__':
    unittest.main()
