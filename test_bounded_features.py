import csv
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from collect_bailey2007 import build_rows, DATA
from experiments.bounded_features import partial_stereo_compatible, external_features, calibrated_transfer
from train import ALLOWED


class BoundedFeaturesTest(unittest.TestCase):
    def test_snapshot_rebuild_preserves_bounds(self):
        rows = build_rows()
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        self.assertEqual(stream.getvalue(), (DATA/'bailey2007_bounds.csv').read_text())
        self.assertEqual(sum(r['feature_eligible'] for r in rows), 16)
        for row in rows:
            self.assertEqual(row['CC50_relation'], '>')
            if row['EC50_relation'] != '=':
                self.assertFalse(row['feature_eligible'])
                self.assertEqual(row['reported_cc50_ec50_limit'], '')

    def test_specified_stereocentres_cannot_be_inverted(self):
        source = 'N[C@@H](C)C(=O)O'
        self.assertTrue(partial_stereo_compatible(source, source))
        self.assertFalse(partial_stereo_compatible(source, 'N[C@H](C)C(=O)O'))
        self.assertFalse(partial_stereo_compatible(source, 'NC(C)C(=O)O'))
        self.assertTrue(partial_stereo_compatible('NC(C)C(=O)O', source))
        self.assertFalse(partial_stereo_compatible(source, 'N[C@@H](CC)C(=O)O'))

    def test_context_and_forbidden_columns(self):
        frame = pd.read_csv(DATA/'train.csv', usecols=ALLOWED)
        expected = external_features(frame)
        self.assertEqual(expected.reported_limit.notna().sum(), 14)
        for name in ['SI VACV', 'CC50_uM', 'IC50_uM_VACV', 'Title', 'index']:
            frame[name] = 'untrusted value'
        pd.testing.assert_frame_equal(external_features(frame), expected)
        frame['Culture'] = 'unmatched cell'
        self.assertTrue(external_features(frame).reported_limit.isna().all())

    def test_calibration_learns_from_training_labels_and_deduplicates(self):
        a = pd.DataFrame({'reported_limit': [10., 10., 20., 30.], 'molecule_id': ['a','a','b','c']})
        b = pd.DataFrame({'reported_limit': [40., np.nan], 'molecule_id': ['d','']})
        with patch('experiments.bounded_features.external_features', side_effect=[a,b]):
            pred, model = calibrated_transfer(None, [30.,30.,40.,60.], None, [1.,7.])
        slope = (10*30 + 20*40 + 30*60)/(10**2 + 20**2 + 30**2)
        np.testing.assert_allclose(pred, [40*slope,7.])
        self.assertAlmostEqual(model['slope'], slope)
        with patch('experiments.bounded_features.external_features', side_effect=[a,b]):
            doubled, _ = calibrated_transfer(None, [60.,60.,80.,120.], None, [1.,7.])
        self.assertAlmostEqual(doubled[0], 2*pred[0])
        with patch('experiments.bounded_features.external_features', side_effect=[a.iloc[:3],b]):
            fallback, model = calibrated_transfer(None, [30.,30.,40.], None, [1.,7.])
        np.testing.assert_array_equal(fallback, [1.,7.])
        self.assertEqual(model['reason'], 'insufficient_fold_support')


if __name__ == '__main__':
    unittest.main()
