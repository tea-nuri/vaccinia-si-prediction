import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import collect_high_si as collector
from collect_training_v2 import parent_key
import corrected_pipeline as cp


class HighSIDataTests(unittest.TestCase):
    def test_json_integrity_allows_order_only(self):
        a = {'activities':[{'activity_id':2,'standard_value':'12'},
                           {'activity_id':1,'standard_value':'3'}], 'page_meta':{'next':None}}
        b = json.loads(json.dumps({'page_meta':{'next':None}, 'activities':list(reversed(a['activities']))}))
        self.assertEqual(collector.normalized_json_hash(json.dumps(a)),
                         collector.normalized_json_hash(json.dumps(b)))
        b['activities'][0]['modality'] = None
        self.assertEqual(collector.normalized_json_hash(json.dumps(a)),
                         collector.normalized_json_hash(json.dumps(b)))
        b['activities'][0]['standard_value'] = '3000'
        self.assertNotEqual(collector.normalized_json_hash(json.dumps(a)),
                            collector.normalized_json_hash(json.dumps(b)))

    def test_censoring_is_not_a_point(self):
        for text in ['>1000','<1','≥1000','≤1','not measured']:
            with self.assertRaises(ValueError):
                collector.numeric_mean(text)
        self.assertEqual(collector.numeric_mean('7.9 ± 0.5 nM'), 7.9)

    def test_real_pool_alignment_and_units(self):
        added = pd.read_csv(cp.DATA/'external_training_high_si_v4.added.csv')
        train, test = cp.read_inputs()
        forbidden = set(pd.concat([train.SMILES,test.SMILES]).map(parent_key))
        self.assertFalse(set(added.SMILES.map(parent_key)) & forbidden)
        self.assertEqual(len(added), 7)
        self.assertEqual(added.parent_connectivity.nunique(), 6)
        self.assertEqual(int((added.SI > 1000).sum()), 6)
        nigericin = added[added.compound_name.eq('Nigericin')]
        self.assertEqual(set(nigericin.strain), {'Copenhagen','Western Reserve'})
        np.testing.assert_allclose(sorted(nigericin.IC50), [.0079,.0195])
        self.assertEqual(set(nigericin.SI), {421.,1038.})
        chembl = added[added.assay_id.notna()]
        self.assertEqual(len(chembl), 2)
        self.assertTrue(chembl.strain.isna().all())
        self.assertTrue(chembl[['IC50','CC50','EC50']].isna().all().all())

    def test_exclusion_happens_before_target_parsing(self):
        manifest, raw = collector.sources(True)
        root = ET.fromstring(raw['article.xml'])
        table = next(t for t in root.findall('.//table-wrap') if t.get('id')=='tbl0010')
        for row in table.findall('.//tbody/tr'):
            if ''.join(row[0].itertext()) == 'Monensin':
                for cell in row[1:]:
                    cell.clear()
                    cell.text = 'INVALID TARGET MUST NOT BE READ'
        raw['article.xml'] = ET.tostring(root)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(collector, 'sources', return_value=(manifest,raw)):
                with contextlib.redirect_stdout(io.StringIO()):
                    output = collector.build(True, Path(directory)/'pool.csv')
        self.assertEqual(len(output), 67)


if __name__ == '__main__':
    unittest.main()
