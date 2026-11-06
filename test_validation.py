import copy
import json
import unittest
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from collect_data import measurement, build
from train import fit_predict, ALLOWED

ROOT = Path(__file__).resolve().parent


class DataValidationTests(unittest.TestCase):
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
        self.assertEqual(len(rows), 63)
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
