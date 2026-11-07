"""Reproduce the verified MoLFormer candidate, optionally from a frozen cache.

Without --embeddings, run on CUDA with PyTorch/Transformers installed. With a
verified aligned cache, training and export require only the CPU dependencies.
"""
import argparse
from pathlib import Path
import subprocess
import sys
from threadpoolctl import threadpool_limits
from experiments.prepare_molformer_inputs import main as prepare_inputs
from molformer_candidate import main as export_candidate

ROOT=Path(__file__).resolve().parent

def main(args):
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if args.embeddings is None:
        inputs=args.output/'encoder_inputs.json'
        prepare_inputs(inputs,include_test=True)
        args.embeddings=args.output/'embeddings.npz'
        subprocess.run([sys.executable,str(ROOT/'experiments/extract_molformer.py'),
                        '--input',str(inputs),'--output',str(args.embeddings)],check=True)
    export_candidate(args)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'run_best')
    p.add_argument('--embeddings',type=Path,help='Existing .npz and matching .json from extract_molformer.py')
    with threadpool_limits(limits=1):main(p.parse_args())
