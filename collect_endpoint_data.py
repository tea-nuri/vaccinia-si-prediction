"""Rebuild rejected auxiliary endpoint datasets from hash-verified sources.

These separate endpoints are not SI labels and are not used by train.py.
Default operation is fully offline. --refresh verifies fresh upstream responses
against the saved source hashes and refuses silent source changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
from experiments.curate_toxicity import main as toxicity
from experiments.curate_reporter_potency import main as potency

ROOT=Path(__file__).resolve().parent

def digest(raw):return hashlib.sha256(raw).hexdigest()

def main(args):
    manifest=json.loads((ROOT/'data/endpoint_sources.manifest.json').read_text())
    archive=ROOT/'data/endpoint_sources.zip'
    assert digest(archive.read_bytes())==manifest['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        for source in manifest['files']:
            key=source['key'];relative=Path(key)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid source key')
            raw=z.read(key)
            if args.refresh and source['url']:
                with urllib.request.urlopen(source['url'],timeout=60) as stream:raw=stream.read()
            if digest(raw)!=source['sha256']:raise ValueError('Source changed: '+key)
            dest=args.cache/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    toxicity(args.cache,args.output)
    potency(args.cache,args.output)
    print('Auxiliary endpoints rebuilt; no SI labels or production-model changes.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=ROOT/'cache/endpoint_sources')
    p.add_argument('--output',type=Path,default=ROOT/'data/auxiliary_endpoints')
    p.add_argument('--refresh',action='store_true')
    main(p.parse_args())
