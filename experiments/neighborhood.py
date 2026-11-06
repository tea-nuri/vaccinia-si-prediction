"""Evaluate neighbor regression over V8 on the frozen six validation splits.

Run: OPENBLAS_NUM_THREADS=1 python experiments/neighborhood.py
No network calls. Intermediate predictions stay under run_neighborhood.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train as core
from literature import SPLITS, lookup_predictions, structure_key
from experiments.literature_candidates import candidate_lookup, main as build_v8

OUT = ROOT / 'run_neighborhood'
CONFIGS = [(kind, k, weight) for kind in ('morgan', 'text', 'hybrid')
           for k in (3, 7, 15) for weight in (.25, .5)]


def neighbors(training, y, target):
    """Fit text vocabulary only on this training fold; no target labels used."""
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 6),
                          lowercase=False, sublinear_tf=True)
    x = vec.fit_transform(training.SMILES)
    text = (vec.transform(target.SMILES) @ x.T).toarray()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
                                                    includeChirality=True)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in training.SMILES]
    mol = np.array([DataStructs.BulkTanimotoSimilarity(
        gen.GetFingerprint(Chem.MolFromSmiles(s)), fps) for s in target.SMILES])
    # Context mismatches reduce confidence without excluding all neighbors.
    context = np.ones_like(mol)
    for col in ('strain', 'Culture'):
        normalize = lambda s: s.fillna('UNKNOWN').str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
        a, b = normalize(target[col]).to_numpy(), normalize(training[col]).to_numpy()
        context *= np.where(a[:, None] == b[None, :], 1., .5)
    result = {}
    for kind, sim in [('morgan', mol), ('text', text), ('hybrid', .5*(mol+text))]:
        sim = sim**4 * context
        for k in (3, 7, 15):
            ix = np.argsort(sim, axis=1)[:, -k:]
            w = np.take_along_axis(sim, ix, axis=1)
            result[kind, k] = np.divide((w*np.clip(y[ix], 0, 10000)).sum(1),
                w.sum(1), out=np.full(len(target), np.clip(y, 0, 10000).mean()), where=w.sum(1)>0)
    return result


def combine(base, neighbor, weight, training, y, target):
    p = (1-weight)*base + weight*neighbor
    p = core.apply_inactivity_evidence(training, y, target, p)[0]
    lookup = lookup_predictions(target).prediction.to_numpy(dtype=float, copy=True)
    extra = candidate_lookup(target)
    mask = extra.eligible_new_match.to_numpy()
    lookup[mask] = extra.loc[mask, 'prediction']
    found = np.isfinite(lookup)
    p[found] = lookup[found]
    return p


def main():
    OUT.mkdir(exist_ok=True)
    build_v8(OUT/'v8')  # Checks frozen source hashes and row alignment.
    tr = pd.read_csv(core.DATA/'train.csv', usecols=core.ALLOWED+['SI VACV'])
    y = tr['SI VACV'].to_numpy(float)
    v8 = pd.read_csv(OUT/'v8/validation.csv')
    groups = tr.SMILES.map(structure_key).to_numpy()
    scaffolds = tr.SMILES.map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
    report = {'baseline': 'V8 literature2019 candidate', 'scores': {},
              'selection_rule': 'Strict improvement over V8 on all five molecular splits and scaffold.',
              'limitations': ['Repeatedly explored molecules; this is not independent validation.',
                             'Published measurements remain fixed; evaluates retrieval-assisted prediction.'],
              'status': 'local_only_not_submitted',
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                  for p in ('data/train.csv', 'data/test.csv', 'results/validation.csv',
                            'data/literature_candidates.csv')}}
    names = {c: f'{c[0]}_k{c[1]}_weight{c[2]}' for c in CONFIGS}
    for split in SPLITS:
        seed = None if split == 'scaffold' else int(split)
        predictions = {c: np.zeros(len(y)) for c in CONFIGS}
        for ix, va in GroupKFold(5, shuffle=seed is not None, random_state=seed).split(
                tr, groups=scaffolds if seed is None else groups):
            nn = neighbors(tr.iloc[ix], y[ix], tr.iloc[va])
            for c in CONFIGS:
                predictions[c][va] = combine(v8[split].to_numpy()[va], nn[c[:2]], c[2],
                                             tr.iloc[ix], y[ix], tr.iloc[va])
        scores = {'baseline': float(np.mean((v8[split]-y)**2))}
        scores.update({names[c]: float(np.mean((p-y)**2)) for c, p in predictions.items()})
        report['scores'][split] = scores
        pd.DataFrame({names[c]: p for c, p in predictions.items()}).to_csv(OUT/f'{split}.csv', index=False)
        print(split, 'baseline', scores['baseline'], 'best', min((s,n) for n,s in scores.items()), flush=True)
    passing = [c for c in CONFIGS if all(s[names[c]] < s['baseline'] for s in report['scores'].values())]
    passing.sort(key=lambda c: np.mean([report['scores'][s][names[c]] for s in SPLITS[:-1]]))
    report['passing'] = [names[c] for c in passing]
    report['selected'] = names[passing[0]] if passing else None
    if passing:
        c = passing[0]
        te = pd.read_csv(core.DATA/'test.csv', usecols=core.ALLOWED)
        sub = pd.read_csv(OUT/'v8/submission.csv')
        nn = neighbors(tr, y, te)
        sub['SI VACV'] = combine(sub['SI VACV'].to_numpy(), nn[c[:2]], c[2], tr, y, te)
        assert np.isfinite(sub['SI VACV']).all() and (sub['SI VACV'] >= 0).all()
        sub.to_csv(OUT/'submission.csv', index=False)
        report['candidate_sha256'] = hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest()
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Selected:', report['selected'], flush=True)


if __name__ == '__main__':
    main()
