import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.scaffold_generalization import (
    CANDIDATES, SCAFFOLD_SPLITS, combine_predictions, known_culture_mask,
    predict_candidates, selected_candidates, selection_checks,
    unseen_scaffold_mask)
from train import ALLOWED


class ScaffoldGeneralizationTest(unittest.TestCase):
    def frame(self, smiles):
        return pd.DataFrame({'SMILES': smiles, 'strain': 'VACV', 'Culture': 'Vero cells'})

    def test_gate_is_training_structure_only_and_preserves_target_order(self):
        training = self.frame(['C1CCCCC1', 'CC'])
        target = self.frame(['OC1CCCCC1', 'c1ccncc1', 'CCC', 'c1ccccc1'])
        expected = np.array([False, True, False, True])
        np.testing.assert_array_equal(unseen_scaffold_mask(training, target), expected)
        for column in ['SI VACV', 'CC50_uM', 'IC50_uM_VACV', 'Title', 'index']:
            training[column] = -123
            target[column] = 'forbidden target value'
        np.testing.assert_array_equal(unseen_scaffold_mask(training, target), expected)
        order = [3, 0, 2, 1]
        np.testing.assert_array_equal(unseen_scaffold_mask(training, target.iloc[order]), expected[order])
        target['Culture'] = ['Vero', 'BSC-40 cells', 'unknown', 'Vero cells']
        np.testing.assert_array_equal(known_culture_mask(training, target), [True, False, False, True])

    def test_blend_keeps_seen_scaffolds_and_precedence_and_forbidden_columns(self):
        training = self.frame(['C1CCCCC1'])
        target = self.frame(['OC1CCCCC1', 'c1ccncc1', 'c1ccccc1', 'C1CCNCC1'])
        for column in ['SI VACV', 'CC50_uM', 'IC50_uM_VACV', 'Title', 'index']:
            training[column] = 'forbidden training value'
            target[column] = 'forbidden target value'

        def check_inputs(a, y, b):
            self.assertEqual(list(a.columns), ALLOWED)
            self.assertEqual(list(b.columns), ALLOWED)

        def neighbor(a, y, b):
            check_inputs(a, y, b)
            return {('text', 7): np.array([1000., 2000., 3000., 4000.])}

        def inactive(a, y, b, pred):
            check_inputs(a, y, b)
            np.testing.assert_allclose(pred, [100., 650., 975., 1300.])
            result = pred.copy(); result[3] = 10000.
            return result, {}

        def bounds(a, y, b, pred):
            check_inputs(a, y, b)
            result = pred.copy(); result[1:3] = [888., 777.]
            return result, {'applied_rows': 2}

        with patch('experiments.scaffold_generalization.neighbors', side_effect=neighbor), \
             patch('experiments.scaffold_generalization.apply_inactivity_evidence', side_effect=inactive), \
             patch('experiments.scaffold_generalization.calibrated_transfer', side_effect=bounds):
            pred, audit = combine_predictions(training, [100.], target,
                                              [100., 200., 300., 400.], .25,
                                              fixed=[np.nan, 123., np.nan, np.nan])
        np.testing.assert_allclose(pred, [100., 123., 777., 10000.])
        self.assertEqual(audit['unseen_scaffold_rows'], 3)

    def test_context_and_combined_heads_are_gated_before_fixed_measurements(self):
        training = self.frame(['C1CCCCC1'])
        target = self.frame(['OC1CCCCC1', 'c1ccncc1', 'c1ccccc1'])
        baseline = np.array([100., 200., 300.])
        nn = np.array([1000., 2000., 3000.])
        head = np.array([10., 20., 30.])
        fixed = np.array([np.nan, np.nan, 123.])
        target['SI VACV'] = [-999., -999., -999.]

        def context(a, y, b, old, selected, lookup_override):
            self.assertEqual(list(a.columns), ALLOWED)
            self.assertEqual(list(b.columns), ALLOWED)
            self.assertEqual(selected, 'logistic_C100_context1_power2_weight0.25')
            return .75*old + .25*head

        identity = lambda a, y, b, pred: (pred, {})
        with patch('experiments.scaffold_generalization.candidate_predict', side_effect=context), \
             patch('experiments.scaffold_generalization.apply_inactivity_evidence', side_effect=identity), \
             patch('experiments.scaffold_generalization.calibrated_transfer', side_effect=identity):
            result = predict_candidates(training, np.array([100.]), target,
                                        baseline, nn, fixed)
        np.testing.assert_allclose(result['context0.25'], [100., 155., 123.])
        np.testing.assert_allclose(result['neighbor0.25_context0.25'], [100., 605., 123.])

    def test_selection_rejects_original_only_gain_and_accepts_guarded_gain(self):
        score = lambda: {'baseline': 100., **{name: 90. for name in CANDIDATES}}
        report = {'scores': {split: score() for split in SCAFFOLD_SPLITS},
                  'known_culture_scores': {split: score() for split in SCAFFOLD_SPLITS},
                  'mean_original_seeds': score(), 'mean_additional_seeds': score(),
                  'mean_scaffold_known_culture_mse': score()}
        # A spectacular score on the already explored split cannot compensate
        # for a worse average on both newly specified scaffold protocols.
        report['scores']['scaffold']['neighbor0.25'] = 1.
        for split in SCAFFOLD_SPLITS[1:]:
            report['scores'][split]['neighbor0.25'] = 105.
        report['known_culture_scores']['scaffold']['neighbor0.5'] = 101.
        report['mean_original_seeds']['neighbor0.25_context0.25'] = 126.
        self.assertEqual(selected_candidates(report), ['context0.25'])
        self.assertFalse(selection_checks(report)['neighbor0.25']['new_scaffold_mean_improves'])


if __name__ == '__main__':
    unittest.main()
