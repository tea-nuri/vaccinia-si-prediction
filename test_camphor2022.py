import csv
import hashlib
import io
import json
import unittest

import pandas as pd
from rdkit import Chem

from collect_camphor2022 import build, structure
from experiments.camphor2022 import audit_matches
from literature import DATA, lookup_predictions, structure_key


class Camphor2022Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {row['compound']:row for row in build()}
        cls.primary = pd.read_csv(DATA/'camphor2022.csv')
        cls.empty = pd.read_csv(DATA/'external.csv').iloc[:0]

    def target(self, names):
        return pd.DataFrame([dict(SMILES=self.rows[n]['smiles'], Culture='Vero', strain='Copenhagen') for n in names])

    def test_snapshot_and_source_outcomes(self):
        rows = list(self.rows.values())
        out = io.StringIO(newline='')
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
        content = out.getvalue().encode()
        self.assertEqual(content, (DATA/'camphor2022.csv').read_bytes())
        report = json.loads((DATA/'camphor2022_validation.json').read_text())
        self.assertEqual(hashlib.sha256(content).hexdigest(), report['sha256'])
        self.assertEqual((report['records'],report['measured'],report['inactive'],report['not_tested']), (36,23,9,4))

    def test_cip_and_double_bond_stereo_survive_smiles_roundtrip(self):
        for name in self.rows:
            mol = Chem.MolFromSmiles(self.rows[name]['smiles'])
            self.assertFalse(any(a.GetAtomMapNum() for a in mol.GetAtoms()))
            expected = 'S' if '′' in name else 'R'
            self.assertEqual([v for _,v in Chem.FindMolChiralCenters(mol)], [expected,expected])
            imines = [b.GetStereo() for b in mol.GetBonds() if b.GetBondType()==Chem.BondType.DOUBLE
                      and {b.GetBeginAtom().GetSymbol(),b.GetEndAtom().GetSymbol()}=={'C','N'}]
            self.assertEqual(imines.count(Chem.BondStereo.STEREOE), 1 if name[0] in '234' else 2)
            self.assertEqual(imines.count(Chem.BondStereo.STEREOZ), 1 if name[0] in '34' else 0)
        self.assertEqual(structure_key(self.rows['3b']['smiles']),
                         structure_key(r'C=CCN1C(=O)CS/C1=N\N=C1/C[C@H]2CC[C@]1(C)C2(C)C'))
        self.assertNotEqual(self.rows['3b']['smiles'], self.rows['3′b']['smiles'])

    def test_missing_si_is_not_inactive_and_nt_is_excluded(self):
        values = lookup_predictions(self.target(['3b','3′d','4b']), self.empty, self.primary).prediction
        self.assertAlmostEqual(values.iloc[0], 99/72)
        self.assertEqual(values.iloc[1], 10000)
        self.assertTrue(pd.isna(values.iloc[2]))
        self.assertEqual(self.rows['2b']['HRMS_ion'], '[M-H]+')
        self.assertEqual(self.rows['2b']['formula'], 'C14H23N3S')

    def test_conflicting_study_is_skipped_and_context_is_required(self):
        target = self.target(['3b','3′c','4c'])
        audit = audit_matches(target)
        self.assertTrue(audit.conflicting_measurement.iloc[0])
        self.assertFalse(audit.eligible_new_match.iloc[0])
        self.assertTrue(audit.eligible_new_match.iloc[1:].all())
        target['CC50_uM']=1e20;target['IC50_uM_VACV']=0
        pd.testing.assert_frame_equal(audit_matches(target), audit)
        target['strain']='Western Reserve'
        self.assertFalse(audit_matches(target).eligible_new_match.any())
        target['strain']='Copenhagen';target['Culture']='HeLa'
        self.assertFalse(audit_matches(target).eligible_new_match.any())


if __name__ == '__main__':
    unittest.main()
