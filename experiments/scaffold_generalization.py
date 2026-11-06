"""Evaluate neighbor/context-classifier corrections on unseen Murcko scaffolds.

Three scaffold protocols drive selection. Eight canonical-molecule protocols
guard against excessive regressions. All splits reuse the same small dataset;
this is an exploratory stress test, not independent/public-score validation.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import ALLOWED, apply_inactivity_evidence, fit_predict_v6
from literature import cell_key, structure_key
from experiments.bounded_features import calibrated_transfer, main as reproduce_bounded
from experiments.measurement_agreement import SELECTED_GATE, agreement_lookup
from experiments.neighborhood import neighbors
from experiments.text_gate import candidate_predict

OUT = ROOT/'run_scaffold_generalization'
RANDOM_SPLITS = ['42', '2026', '117', '2027', '7', '23', '101', '509']
SCAFFOLD_SPLITS = ['scaffold', 'scaffold_shuffle42', 'scaffold_shuffle2026']
WEIGHTS = [.25, .5]
CONTEXT_GATE = 'logistic_C100_context1_power2_weight0.25'
CANDIDATES = ['neighbor0.25', 'neighbor0.5', 'context0.25', 'neighbor0.25_context0.25']


def scaffold_keys(frame):
    return frame.SMILES.map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(
        smiles=s) or 'ACYCLIC').to_numpy()


def unseen_scaffold_mask(training, target):
    """The applicability gate sees structures in the training fold only."""
    known = set(scaffold_keys(training[ALLOWED]))
    return np.array([key not in known for key in scaffold_keys(target[ALLOWED])])


def known_culture_mask(training, target):
    known = set(training.Culture.map(cell_key)) - {'', 'nan', 'unknown', 'none'}
    return target.Culture.map(cell_key).isin(known).to_numpy()


def combine_predictions(training, y, target, baseline, weight, neighbor=None,
                        fixed=None):
    """Keep measured lookup values and learned evidence corrections intact."""
    training, target = training[ALLOWED], target[ALLOWED]
    baseline = np.asarray(baseline, dtype=float)
    if neighbor is None:
        neighbor = neighbors(training, np.asarray(y), target)['text', 7]
    if fixed is None:
        fixed = agreement_lookup(target).prediction.to_numpy(dtype=float)
    fixed = np.asarray(fixed, dtype=float)
    unseen = unseen_scaffold_mask(training, target)
    prediction = baseline.copy()
    prediction[unseen] = ((1-weight)*baseline[unseen]
                          + weight*np.asarray(neighbor)[unseen])
    prediction, _ = apply_inactivity_evidence(training, np.asarray(y), target, prediction)
    prediction, calibration = calibrated_transfer(training, y, target, prediction)
    # Final precedence guarantees trusted measurements stay fixed even if future
    # external features overlap them. Current source collections do not overlap.
    measured = np.isfinite(fixed)
    prediction[measured] = fixed[measured]
    return prediction, {'unseen_scaffold_rows': int(unseen.sum()),
                        'bounded_features': calibration}


def fit_current_baseline(training, y, target, fixed):
    prediction, _ = fit_predict_v6(training, y, target)
    measured = np.isfinite(fixed)
    prediction[measured] = fixed[measured]
    prediction = candidate_predict(training, y, target, prediction, SELECTED_GATE,
                                   lookup_override=fixed)
    prediction, _ = calibrated_transfer(training, y, target, prediction)
    return prediction


def predict_candidates(training, y, target, baseline, neighbor, fixed):
    predictions = {}
    for weight in WEIGHTS:
        predictions[f'neighbor{weight}'], _ = combine_predictions(
            training, y, target, baseline, weight, neighbor, fixed)
    # candidate_predict returns .75 baseline + .25 fold-trained context head.
    context = candidate_predict(training[ALLOWED], y, target[ALLOWED], baseline,
                                CONTEXT_GATE, lookup_override=fixed)
    unseen = unseen_scaffold_mask(training, target)
    proposed = {
        'context0.25': context,
        'neighbor0.25_context0.25': context + .25*(neighbor-baseline)}
    for name, value in proposed.items():
        prediction = np.where(unseen, value, baseline)
        prediction, _ = apply_inactivity_evidence(training[ALLOWED], y,
                                                  target[ALLOWED], prediction)
        prediction, _ = calibrated_transfer(training[ALLOWED], y,
                                              target[ALLOWED], prediction)
        measured = np.isfinite(fixed)
        prediction[measured] = fixed[measured]
        predictions[name] = prediction
    return predictions


def mse(y, prediction, mask=None):
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(np.mean((np.asarray(y)[mask]-np.asarray(prediction)[mask])**2)) if mask.any() else None


def scores_for(y, predictions, mask=None):
    return {key: mse(y, value, mask) for key, value in predictions.items()}


def selection_checks(report):
    """Expose every frozen guard so rejected tradeoffs remain reviewable."""
    checks = {}
    for name in CANDIDATES:
        original = report['scores']['scaffold']
        new = [report['scores'][key] for key in SCAFFOLD_SPLITS[1:]]
        culture = [report['known_culture_scores'][key] for key in SCAFFOLD_SPLITS]
        checks[name] = {
            'original_scaffold_improves': bool(original[name] < original['baseline']),
            'original_known_culture_improves': bool(culture[0][name] < culture[0]['baseline']),
            'new_scaffold_mean_improves': bool(np.mean([score[name] for score in new]) < np.mean([score['baseline'] for score in new])),
            'new_known_culture_mean_improves': bool(np.mean([score[name] for score in culture[1:]]) < np.mean([score['baseline'] for score in culture[1:]])),
            'each_new_scaffold_within_10_percent': all(score[name] <= 1.10*score['baseline'] for score in new),
            'each_known_culture_scaffold_within_10_percent': all(score[name] is not None and score[name] <= 1.10*score['baseline'] for score in culture),
            'original_canonical_mean_within_25_percent': bool(report['mean_original_seeds'][name] <= 1.25*report['mean_original_seeds']['baseline']),
            'additional_canonical_mean_within_25_percent': bool(report['mean_additional_seeds'][name] <= 1.25*report['mean_additional_seeds']['baseline'])}
    return checks


def selected_candidates(report):
    """Frozen guards: selection must not follow whichever score looks best."""
    passing = [name for name, checks in selection_checks(report).items() if all(checks.values())]
    return sorted(passing, key=lambda name: report['mean_scaffold_known_culture_mse'][name])


def main():
    OUT.mkdir(exist_ok=True)
    if not all((ROOT/f'run_bounded_features/validation_{split}.csv').exists()
               for split in ['scaffold', *RANDOM_SPLITS]):
        reproduce_bounded()
    tr = pd.read_csv(ROOT/'data/train.csv', usecols=ALLOWED+['SI VACV'])
    te = pd.read_csv(ROOT/'data/test.csv', usecols=ALLOWED+['index'])
    y = tr['SI VACV'].to_numpy(float)
    canonical = tr.SMILES.map(structure_key).to_numpy()
    scaffolds = scaffold_keys(tr)
    fixed = agreement_lookup(tr).prediction.to_numpy(dtype=float)
    expected = json.loads((ROOT/'results/bounded_features_experiment.json').read_text())
    for name, digest in expected['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    assert expected['selected'] == '1.0'
    source_names = ['data/train.csv', 'data/test.csv', 'data/external.csv',
                    'data/external_inactivity.csv', 'data/primary_literature.csv',
                    'data/literature_candidates.csv', 'data/camphor2022.csv',
                    'data/bailey2007_bounds.csv', 'results/candidate_bounded_features.csv']
    report = {
        'status': 'local_only_not_submitted', 'scores': {}, 'folds': {},
        'known_culture_scores': {}, 'group_diagnostics': {}, 'candidates': CANDIDATES,
        'model': 'Fold-local text TFIDF 2-6 k=7 neighbors at weights .25/.5; or additional .25 text+context LogisticRegression C100 odds power2; or combined .5 baseline + .25 neighbors + .25 context head. Changes apply only when target Murcko scaffold is absent in training. Existing evidence and trusted measurement precedence retained.',
        'baseline': 'Measurement agreement + selected text gate + fold-trained bounded features, weight 1.0',
        'selection_rule': 'Both whole-set and known-normalized-Culture MSE improve on original scaffold; both metrics improve on average over two new shuffled-scaffold protocols; neither metric in either new protocol worsens >10%; five original canonical seeds and three additional seeds each worsen <=25% on average. Choose lowest mean known-Culture MSE over three scaffold protocols among passing candidates.',
        'selection_context': 'Four candidates and guards frozen before new shuffled-scaffold results. The initial all-scaffold guard was revised after discovering a dominant unseen-Culture family and observing original-split candidate results; this is exploratory, not a preregistered independent study.',
        'limitations': ['All protocols reuse the same 320 rows and external retrieval sources; shuffled scaffold protocols are sensitivity checks, not independent confirmation.',
                        'No assertion that scaffold MSE estimates Kaggle Public Score or that public MSE below one million is reached.',
                        'A dominant held-out family has an unseen Culture; per-family and known-Culture diagnostics are shown because test Culture coverage may differ.',
                        'External bounds and partial stereochemistry retain the uncertainty documented in bounded_features_experiment.json.'],
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}}
    # Rebuild the original scaffold baseline from scratch to verify cached OOF,
    # then perform the two new protocols without reusing old fold predictions.
    for split in SCAFFOLD_SPLITS + RANDOM_SPLITS:
        is_scaffold = split in SCAFFOLD_SPLITS
        seed = None if split == 'scaffold' else int(split.replace('scaffold_shuffle', ''))
        groups = scaffolds if is_scaffold else canonical
        cached = None
        if split == 'scaffold' or not is_scaffold:
            cached_frame = pd.read_csv(ROOT/f'run_bounded_features/validation_{split}.csv')
            np.testing.assert_array_equal(cached_frame.target.to_numpy(), y)
            cached = cached_frame['1.0'].to_numpy(float)
            np.testing.assert_allclose(mse(y, cached), expected['scores'][split]['1.0'], rtol=0, atol=1e-5)
        predictions = {key: np.zeros(len(y)) for key in ['baseline', *CANDIDATES]}
        known_culture = np.zeros(len(y), dtype=bool)
        report['folds'][split] = []
        for fold, (ix, va) in enumerate(GroupKFold(5, shuffle=seed is not None, random_state=seed).split(tr, groups=groups)):
            assert not set(canonical[ix]) & set(canonical[va])
            if is_scaffold:
                assert not set(scaffolds[ix]) & set(scaffolds[va])
                baseline = fit_current_baseline(tr.iloc[ix], y[ix], tr.iloc[va], fixed[va])
                if cached is not None:
                    np.testing.assert_allclose(baseline, cached[va], rtol=0, atol=1e-6)
            else:
                baseline = cached[va]
            predictions['baseline'][va] = baseline
            known_culture[va] = known_culture_mask(tr.iloc[ix], tr.iloc[va])
            nn = neighbors(tr.iloc[ix][ALLOWED], y[ix], tr.iloc[va][ALLOWED])['text', 7]
            for name, prediction in predict_candidates(tr.iloc[ix], y[ix], tr.iloc[va], baseline, nn, fixed[va]).items():
                predictions[name][va] = prediction
            audit = dict(fold=fold, training_rows=len(ix), validation_rows=len(va),
                         known_culture_rows=int(known_culture[va].sum()),
                         unseen_scaffold_rows=int(unseen_scaffold_mask(tr.iloc[ix], tr.iloc[va]).sum()))
            report['folds'][split].append(audit)
            print(split, 'fold', fold, flush=True)
        report['scores'][split] = scores_for(y, predictions)
        report['known_culture_scores'][split] = {'rows': int(known_culture.sum()), **scores_for(y, predictions, known_culture)}
        if is_scaffold:
            entries = []
            for scaffold in sorted(set(scaffolds)):
                mask = scaffolds == scaffold
                entries.append({'scaffold': scaffold, 'rows': int(mask.sum()),
                                'known_culture_rows': int((mask & known_culture).sum()),
                                'mse': scores_for(y, predictions, mask),
                                'sse': {key: float(np.sum((value[mask]-y[mask])**2)) for key, value in predictions.items()}})
            entries.sort(key=lambda entry: entry['sse']['baseline'], reverse=True)
            largest = entries[0]['scaffold']
            report['group_diagnostics'][split] = {
                'groups_sorted_by_baseline_sse': entries,
                'largest_baseline_error_group': largest,
                'outside_largest_group': {'rows': int((scaffolds != largest).sum()),
                                        **scores_for(y, predictions, scaffolds != largest)}}
        pd.DataFrame({'target': y, 'known_culture': known_culture, **predictions}).to_csv(OUT/f'validation_{split}.csv', index=False)
        (OUT/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
        print(split, report['scores'][split], 'known_culture', report['known_culture_scores'][split], flush=True)
    for name, splits in [('mean_original_seeds', RANDOM_SPLITS[:5]),
                         ('mean_additional_seeds', RANDOM_SPLITS[5:]),
                         ('mean_scaffold_mse', SCAFFOLD_SPLITS)]:
        report[name] = {key: float(np.mean([report['scores'][split][key] for split in splits])) for key in predictions}
    report['mean_scaffold_known_culture_mse'] = {
        key: float(np.mean([report['known_culture_scores'][split][key] for split in SCAFFOLD_SPLITS]))
        for key in predictions}
    report['passing'] = selected_candidates(report)
    report['selected'] = report['passing'][0] if report['passing'] else None
    report['selection_checks'] = selection_checks(report)
    report['rejection_reasons'] = {
        name: [check for check, passed in checks.items() if not passed]
        for name, checks in report['selection_checks'].items()}
    report['candidate_artifact_created'] = report['selected'] is not None
    if report['selected'] is not None:
        sub = pd.read_csv(ROOT/'results/candidate_bounded_features.csv')
        np.testing.assert_array_equal(sub['index'], te['index'])
        assert hashlib.sha256((ROOT/'results/candidate_bounded_features.csv').read_bytes()).hexdigest() == expected['candidate_submission_sha256']
        old = sub['SI VACV'].to_numpy().copy()
        test_fixed = agreement_lookup(te).prediction.to_numpy(dtype=float)
        nn = neighbors(tr[ALLOWED], y, te[ALLOWED])['text', 7]
        sub['SI VACV'] = predict_candidates(tr, y, te, old, nn, test_fixed)[report['selected']]
        report['full_fit'] = {'unseen_scaffold_rows': int(unseen_scaffold_mask(tr, te).sum())}
        assert np.isfinite(sub['SI VACV']).all() and sub['SI VACV'].ge(0).all()
        report['full_fit']['changed_rows'] = int((np.abs(sub['SI VACV']-old)>1e-8).sum())
        report['full_fit']['unknown_culture_rows'] = int((~known_culture_mask(tr, te)).sum())
        sub.to_csv(OUT/'submission.csv', index=False)
        report['candidate_submission_sha256'] = hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest()
    elif (OUT/'submission.csv').exists():
        # This filename belongs only to this experiment. A rejected run must not
        # accidentally leave an older run's candidate looking newly selected.
        (OUT/'submission.csv').unlink()
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Selected:', report['selected'], 'scaffold means:', report['mean_scaffold_mse'], flush=True)


if __name__ == '__main__':
    main()
