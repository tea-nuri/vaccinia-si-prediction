import unittest

import numpy as np
import pandas as pd

import corrected_pipeline as pipeline
from collect_training_v2 import parent_key


class CorrectedTaskTests(unittest.TestCase):
    def test_official_labels_and_alignment(self):
        training, test = pipeline.read_inputs()
        labels = training[pipeline.TARGET].to_numpy(float)
        self.assertEqual(len(training), 320)
        self.assertEqual(int((labels == 0).sum()), 83)
        self.assertEqual(int((labels == 10000).sum()), 0)
        self.assertGreater(labels.max(), 36000)
        self.assertEqual(len(test), 597)

    def test_forbidden_columns_never_change_predictions(self):
        frame = pd.DataFrame({'SMILES': ['C', 'CC', 'CCC', 'CO', 'CCO', 'CCCO'],
                              'strain': ['x'] * 6, 'Culture': ['a'] * 6})
        labels = np.array([0., 2., 50., 0., 20., 500.])
        target = frame.iloc[:2].copy()
        model = pipeline.Regressor('text_log').fit(frame, labels)
        expected = model.predict(target)
        for column in ['CC50_uM', 'IC50_uM_VACV', 'SI VACV', 'Title', 'index']:
            target[column] = [1e50, -1e50]
        np.testing.assert_array_equal(expected, model.predict(target))

    def test_zero_and_large_si_survive_target_transform(self):
        values = np.array([0., .01, 10000., 36066.667])
        for name in ['raw', 'log', 'sqrt']:
            np.testing.assert_allclose(pipeline.inverse_target(pipeline.target_transform(values, name), name), values,
                                       rtol=1e-12, atol=1e-8)
        for values in [np.array([-1.]), np.array([np.nan])]:
            with self.assertRaises(ValueError):
                pipeline.target_transform(values, 'raw')

    def test_all_protocols_exclude_validation_identity(self):
        training, _ = pipeline.read_inputs()
        canonical, scaffolds, _, _ = pipeline.feature_arrays(training)
        for split in pipeline.SCAFFOLD + pipeline.MOLECULAR:
            seen = np.zeros(len(training), int)
            for _, ix, va in pipeline.folds_for(training, split):
                self.assertFalse(set(canonical[ix]) & set(canonical[va]))
                if split in pipeline.SCAFFOLD:
                    self.assertFalse(set(scaffolds[ix]) & set(scaffolds[va]))
                seen[va] += 1
            np.testing.assert_array_equal(seen, 1)

    def test_external_parent_exclusion_handles_salt_and_stereochemistry(self):
        self.assertEqual(parent_key('C[C@H](O)C(=O)[O-].[Na+]'), parent_key('C[C@@H](O)C(=O)O'))
        pool = pd.read_csv(pipeline.DATA / 'external_training_v2.csv')
        training, test = pipeline.read_inputs()
        forbidden = set(training.SMILES.map(parent_key)) | set(test.SMILES.map(parent_key))
        self.assertFalse(set(pool.SMILES.map(parent_key)) & forbidden)
        self.assertTrue(pool.doi.notna().all())
        self.assertTrue(pool.source_url.notna().all())


if __name__ == '__main__':
    unittest.main()
