import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import collect_external_candidate_v3 as collector
from collect_training_v2 import parent_key
from corrected_pipeline import DATA, read_inputs


class ExternalCandidateTests(unittest.TestCase):
    def test_real_table_censoring_and_duration_annotations(self):
        # These forms caused the initial inventory to overcount exact pairs.
        self.assertIsNone(collector.printed_point_si('0.06 ± 0.02 (>1667)'))
        self.assertIsNone(collector.printed_point_si('1.0 ± 0.2 (>100)'))
        self.assertIsNone(collector.unambiguous_mean('11 (6 dpt)/40'))
        self.assertIsNone(collector.unambiguous_mean('40 (6 dpt)/>100'))
        self.assertEqual(collector.printed_point_si('0.9 ± 0.1 (45)'), 45.)
        self.assertEqual(collector.unambiguous_mean('6,3'), 6.3)
        for text in ['(≥36)', '(≤3)', '(inf)', '(-1)']:
            self.assertIsNone(collector.printed_point_si(text))

    def test_competition_exclusion_precedes_target_access(self):
        train, _ = read_inputs()
        smiles = train.SMILES.iloc[0]
        manifest = {'paper': {'table_id': 't1', 'names': {'x': {'name': 'ignored', 'pdf_page': 1}}}}
        sources = {'paper/article.xml': b'<article><table-wrap id="t1"><table/></table-wrap></article>',
                   'paper/structures/x.json': json.dumps({'status': 'SUCCESS', 'smiles': smiles}).encode()}
        frame, rejected = collector.collect_paper(manifest, sources, {parent_key(smiles)})
        self.assertTrue(frame.empty)
        self.assertEqual(rejected, [{'compound': 'x', 'reason': 'competition_parent'}])

    def test_offline_source_integrity(self):
        raw = b'verified source snapshot'
        source = {'cache_key': 'paper/source.json', 'url': 'https://example.invalid/no-network',
                  'sha256': hashlib.sha256(raw).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                collector.cached_sources({'sources': [source]}, cache, True)
            target = cache / source['cache_key']
            target.parent.mkdir()
            target.write_bytes(raw)
            self.assertEqual(collector.cached_sources({'sources': [source]}, cache, True)[source['cache_key']], raw)
            target.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                collector.cached_sources({'sources': [source]}, cache, True)

    def test_candidate_pool_and_paper_provenance(self):
        frame = pd.read_csv(DATA / 'external_training_candidate_v3.csv')
        train, test = read_inputs()
        excluded = set(pd.concat([train.SMILES, test.SMILES]).map(parent_key))
        self.assertEqual(len(frame), 60)
        self.assertFalse(set(frame.SMILES.map(parent_key)) & excluded)
        self.assertTrue(np.isfinite(frame.model_target.to_numpy()).all())
        self.assertTrue(frame.doi.notna().all())
        self.assertTrue(frame.source_url.notna().all())
        paper = frame[frame.doi.eq('10.3390/v18010071')]
        self.assertEqual(len(paper), 13)
        np.testing.assert_array_equal(paper.model_target, paper.SI)
        self.assertEqual(int(paper.CC50.isna().sum()), 4)
        self.assertNotIn('2b', paper.source_row.tolist())
        self.assertNotIn('5b', paper.source_row.tolist())
        self.assertTrue(paper.structure_resolver_sha256.str.len().eq(64).all())
        self.assertTrue(paper.structure_supplement_sha256.str.len().eq(64).all())


if __name__ == '__main__':
    unittest.main()
