import copy
import json
import unittest
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from collect_data import measurement, build, build_inactivity, inactive_field
from train import fit_predict, ALLOWED, apply_inactivity_evidence

ROOT = Path(__file__).resolve().parent


class DataValidationTests(unittest.TestCase):
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
        self.assertEqual(len(rows), 75)
        self.assertEqual(len({r["doi"] for r in rows}), 4)
        self.assertEqual(sum(r["usable_for_training"] for r in rows), 67)
        self.assertTrue(all(not r["usable_for_training"] for r in rows if r["SI"] == ""))
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


if __name__ == '__main__':
    unittest.main()
