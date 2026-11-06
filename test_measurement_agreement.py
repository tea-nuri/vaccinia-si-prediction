import unittest

import numpy as np
import pandas as pd

from train import DATA, ALLOWED
from experiments.measurement_agreement import conflicting_measurements, agreement_lookup


class MeasurementAgreementTest(unittest.TestCase):
    def test_missing_values_and_rounding_are_not_conflicts(self):
        values=[[1,1.04],[1,3],[np.nan,3],[np.nan,np.nan],
                [100,104],[100,106],[10,np.inf],[1,10000]]
        np.testing.assert_array_equal(conflicting_measurements(values),
                                      [False,True,False,False,False,True,False,True])
        np.testing.assert_array_equal(conflicting_measurements(np.array(values)[:,::-1]),
                                      conflicting_measurements(values))

    def test_guard_uses_sources_and_preserves_unambiguous_measurements(self):
        frame=pd.read_csv(DATA/'train.csv',usecols=ALLOWED)
        audit=agreement_lookup(frame)
        self.assertTrue(audit.conflict.any())
        self.assertTrue(audit.loc[audit.conflict,'prediction'].isna().all())
        np.testing.assert_allclose(audit.loc[~audit.conflict,'prediction'],
                                   audit.loc[~audit.conflict,'v9_prediction'],equal_nan=True)
        self.assertTrue(audit.loc[audit.conflict,'observations'].str.contains('10.3390/molecules27154761').all())

    def test_labels_and_forbidden_concentrations_cannot_choose_source(self):
        frame=pd.read_csv(DATA/'train.csv',usecols=ALLOWED)
        expected=agreement_lookup(frame)
        for name in ['SI VACV','CC50_uM','IC50_uM_VACV','Title','index']:
            frame[name]='untrusted value'
        pd.testing.assert_frame_equal(agreement_lookup(frame),expected)


if __name__=='__main__':
    unittest.main()
