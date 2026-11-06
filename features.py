import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

def features(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    descriptors, fingerprints, canonical, scaffolds = [], [], [], []
    for i, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError(f'Invalid SMILES in row {i}: {value}')
        canonical.append(Chem.MolToSmiles(mol))
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or 'ACYCLIC')
        fp = np.zeros(2048, dtype=np.float64)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), fp)
        fingerprints.append(fp)
        row = []
        for _, function in Descriptors.descList:
            try:
                v = float(function(mol))
                row.append(v if np.isfinite(v) else np.nan)
            except Exception:
                row.append(np.nan)
        descriptors.append(row)
    return np.asarray(fingerprints), np.asarray(descriptors), canonical, scaffolds
