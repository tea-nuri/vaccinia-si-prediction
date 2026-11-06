import copy
import json
import unittest
import zipfile
from pathlib import Path
import pandas as pd
from collect_data import measurement, build
from train_model import prepare_external, canonical

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

    def test_bound_labels_and_overlap_excluded_from_training(self):
        frame = pd.DataFrame([
            dict(smiles='CCO', SI=10, IC50_relation='=', CC50_relation='=', SI_relation='=', quality_flags='', doi='a'),
            dict(smiles='CCN', SI=20, IC50_relation='=', CC50_relation='=', SI_relation='>', quality_flags='', doi='b'),
            dict(smiles='CCC', SI=30, IC50_relation='=', CC50_relation='=', SI_relation='=', quality_flags='', doi='c')])
        result = prepare_external(frame, [canonical('OCC')])
        self.assertEqual(result.canonical.tolist(), ['CCC'])

    def test_snapshot_is_complete_and_assay_pairs_are_unambiguous(self):
        with zipfile.ZipFile(ROOT/'chembl_snapshot.zip') as z:
            payloads = {n: json.loads(z.read(n)) for n in z.namelist()}
        rows = build(payloads)
        self.assertEqual(len(rows), 63)
        self.assertTrue(all(r['doi'] for r in rows))
        self.assertTrue(all(r['SI_relation']=='=' for r in rows if r['usable_for_training']))
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
