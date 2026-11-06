import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from train import ALLOWED
from experiments.text_gate import candidate_predict, v9_lookup

ROOT = Path(__file__).resolve().parent


class TextGateIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = pd.read_csv(ROOT/'data/train.csv', usecols=ALLOWED+['SI VACV'])
        cls.target = pd.read_csv(ROOT/'data/test.csv', usecols=ALLOWED+['index'])
        cls.baseline = pd.read_csv(ROOT/'results/candidate_camphor2022.csv')['SI VACV'].to_numpy()
        cls.selected = 'logistic_C100_context0_power2_weight0.25'
        cls.prediction = candidate_predict(cls.training, cls.training['SI VACV'].to_numpy(),
                                          cls.target, cls.baseline, cls.selected)

    def test_forbidden_columns_do_not_change_predictions(self):
        training, target = self.training.copy(), self.target.copy()
        for col in ['CC50_uM', 'IC50_uM_VACV', 'Title']:
            training[col] = 'adversarial unused value'
            target[col] = np.nan
        target['index'] = -1
        target['SI VACV'] = 999999999
        result = candidate_predict(training, training['SI VACV'].to_numpy(),
                                   target, self.baseline, self.selected)
        np.testing.assert_allclose(result, self.prediction, atol=1e-9, rtol=0)

    def test_measurements_preserved_and_large_values_not_clipped(self):
        lookup = v9_lookup(self.target).prediction.to_numpy()
        matched = np.isfinite(lookup)
        self.assertTrue(matched.any())
        np.testing.assert_allclose(self.prediction[matched], lookup[matched], atol=1e-9, rtol=0)
        self.assertTrue(np.isfinite(self.prediction).all())
        self.assertTrue((self.prediction >= 0).all())

    def test_target_row_order_is_equivariant(self):
        order = np.arange(len(self.target))[::-1]
        result = candidate_predict(self.training, self.training['SI VACV'].to_numpy(),
                                   self.target.iloc[order], self.baseline[order], self.selected)
        np.testing.assert_allclose(result, self.prediction[order], atol=1e-8, rtol=0)


if __name__ == '__main__':
    unittest.main()
