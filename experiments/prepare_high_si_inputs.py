"""Prepare the expanded external pool for the existing frozen encoder."""
import argparse
from pathlib import Path
import sys
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import corrected_pipeline as cp
from collect_training_v2 import parent_key


def main(output):
    train, test = cp.read_inputs()
    source = cp.DATA / 'external_training_high_si_v4.csv'
    external = pd.read_csv(source)
    forbidden = set(pd.concat([train.SMILES, test.SMILES]).map(parent_key))
    assert not set(external.SMILES.map(parent_key)) & forbidden
    smiles = pd.concat([train.SMILES, external.SMILES], ignore_index=True).tolist()
    cp.save_json(output, dict(
        canonical_inputs=[Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in smiles],
        raw_inputs=smiles, train_rows=len(train), external_rows=len(external), test_rows=0,
        source_sha256={'train':cp.TRAIN_SHA256, 'external':cp.sha(source)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
