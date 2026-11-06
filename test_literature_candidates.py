import csv
import hashlib
import io
import json
import unittest

import numpy as np
import pandas as pd

from collect_literature_candidates import build
from experiments.literature_candidates import candidate_lookup
from literature import DATA, lookup_predictions, structure_key


class CandidateLiteratureTests(unittest.TestCase):
    def test_transcription_reproduces_committed_snapshot(self):
        rows = build()
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        content = stream.getvalue().encode()
        self.assertEqual(content, (DATA/'literature_candidates.csv').read_bytes())
        report = json.loads((DATA/'literature_candidates_validation.json').read_text())
        self.assertEqual(hashlib.sha256(content).hexdigest(), report['sha256'])
        self.assertEqual(len(rows), 17)
        self.assertEqual(sum(r['outcome']=='Not active' for r in rows), 13)

    def test_units_bounds_and_fused_ring_connectivity(self):
        rows = {r['compound']: r for r in build()}
        # Independently written graphs from the fused-ring scheme; a formula
        # assertion alone does not catch accidentally reused SMILES ring digits.
        expected = {
            '1c': 'CC1(C)CC(=O)c2c(nc(-c3ccccc3O)n2O)C1',
            '4c': 'Cn1c2c([n+]([O-])c1-c1ccccc1O)C(=O)CC(C)(C)C2',
        }
        for name, smiles in expected.items():
            self.assertEqual(rows[name]['smiles'], structure_key(smiles))
        r = rows['1c']
        self.assertAlmostEqual(r['IC50'], 1.29*1000/r['molecular_weight'])
        self.assertAlmostEqual(r['CC50']/r['IC50'], 203/1.29)
        self.assertEqual(r['SI'], 159.73)
        self.assertEqual(rows['1g']['CC50_relation'], '<')
        bounded = rows['5b']
        self.assertEqual(bounded['outcome'], 'Measured')
        self.assertEqual(bounded['CC50_relation'], '>')
        self.assertEqual(bounded['SI_relation'], '>')
        self.assertEqual(bounded['SI_from_rounded_means'], '')
        primary = pd.read_csv(DATA/'literature_candidates.csv')
        target = pd.DataFrame([dict(SMILES=bounded['smiles'], Culture='Vero', strain='Copenhagen')])
        result = lookup_predictions(target, pd.read_csv(DATA/'external.csv').iloc[:0], primary, include_2019=True)
        self.assertTrue(result.prediction.isna().all())

    def test_candidate_context_guard_and_existing_measurement_priority(self):
        rows = {r['compound']: r for r in build()}
        target = pd.DataFrame([
            dict(SMILES=rows[name]['smiles'], Culture='Vero', strain='Copenhagen')
            for name in ['1b', '1c', '4c']])
        audit = candidate_lookup(target)
        self.assertFalse(audit.eligible_new_match.iloc[0])
        self.assertTrue(np.isfinite(audit.existing_v7_prediction.iloc[0]))
        np.testing.assert_allclose(audit.prediction.iloc[1:], [203/1.29, 10000])
        self.assertTrue(audit.eligible_new_match.iloc[1:].all())
        target['CC50_uM'] = 1e20
        target['IC50_uM_VACV'] = 0
        pd.testing.assert_frame_equal(candidate_lookup(target), audit)
        target['Culture'] = 'HeLa cells'
        self.assertFalse(candidate_lookup(target).eligible_new_match.any())
        target['Culture'] = 'Vero'
        target['strain'] = 'VACV Lister'
        self.assertFalse(candidate_lookup(target).eligible_new_match.any())


if __name__ == '__main__':
    unittest.main()
