"""Checks that source abstention uses published context, never target labels."""
import unittest

import numpy as np
import pandas as pd

from literature import additional_viruses, source_context_rejections, document_virus_context
from train import fit_predict_v6
from experiments.provenance_guard import combine


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.target = pd.DataFrame({'strain': [
            'Vaccinia virus Copenhagen; Cowpox virus; ECTV',
            'Vaccinia virus Copenhagen; Cowpox virus; ECTV',
            'Vaccinia virus Copenhagen',
            'Vaccinia virus Copenhagen; Cowpox virus; ECTV',
        ]})
        self.audit = pd.DataFrame({
            'prediction': [10., 20., 30., 40.],
            'source': ['10.example/vacv', '10.example/multiple',
                       '10.example/vacv', '10.example/vacv'],
            'match_method': ['matched_ChEMBL_assay_pair_exact_stereo']*3+['exact_stereo'],
        })
        self.coverage = {'10.example/vacv': frozenset(),
                         '10.example/multiple': frozenset({'cowpox','ectromelia'})}

    def test_context_aliases_are_species_not_substrings(self):
        self.assertEqual(additional_viruses('CPXV, ECTV, VARV'),
                         frozenset({'cowpox','ectromelia','variola'}))
        self.assertFalse(additional_viruses('vaccinia orthopoxvirus'))
        self.assertFalse(additional_viruses('notcowpoxword'))

    def test_abstains_only_on_unsupported_numerical_source(self):
        result = source_context_rejections(self.target, self.audit, self.coverage)
        self.assertEqual(result.reject_numerical_lookup.tolist(), [True,False,False,False])
        self.assertEqual(result.reason.tolist(), ['source_context_not_verified','','',''])
        # Another verified candidate source can supply the required context.
        self.audit.loc[0, 'source'] += ';10.example/multiple'
        self.assertFalse(source_context_rejections(
            self.target, self.audit, self.coverage).reject_numerical_lookup.any())

    def test_uses_model_fallback_and_preserves_unaffected_literature(self):
        rejection = source_context_rejections(self.target, self.audit, self.coverage)
        extra = pd.DataFrame({'eligible_new_match': [False]*4,
                              'prediction': [np.nan]*4})
        baseline, candidate = combine(np.array([8000.,7000.,6000.,5000.]),
                                      self.audit, extra, rejection)
        np.testing.assert_array_equal(baseline, [10.,20.,30.,40.])
        np.testing.assert_array_equal(candidate, [8000.,20.,30.,40.])

    def test_forbidden_fields_and_labels_have_no_effect(self):
        expected = source_context_rejections(self.target, self.audit, self.coverage)
        for field in ['CC50_uM','IC50_uM_VACV','SI VACV','Title','index']:
            self.target[field] = [0,1,10000,1e20]
        pd.testing.assert_frame_equal(source_context_rejections(
            self.target, self.audit, self.coverage), expected)
        with self.assertRaisesRegex(ValueError, 'row count'):
            source_context_rejections(self.target.iloc[:1], self.audit, self.coverage)

    def test_invalid_calibration_is_rejected_before_fitting(self):
        for parameter in ['smiles_alpha', 'smiles_slope']:
            for value in [0., -1., np.inf, np.nan]:
                with self.assertRaisesRegex(ValueError, 'positive and finite'):
                    fit_predict_v6(pd.DataFrame(), np.array([]), pd.DataFrame(),
                                   **{parameter:value})

    def test_verified_source_context_has_expected_document_coverage(self):
        sources = document_virus_context()
        self.assertFalse(sources['10.1039/c8md00347e'])
        self.assertTrue({'cowpox','ectromelia'} <= sources['10.1039/d0md00108b'])


if __name__ == '__main__':
    unittest.main()
