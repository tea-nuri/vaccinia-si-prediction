"""Compare an opt-in source-provenance guard against the team's frozen candidate.

Run: OPENBLAS_NUM_THREADS=1 python experiments/provenance_guard.py --confirm
No Kaggle operations. Every baseline is retrained inside the same folds.
"""
import argparse
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
from train import ALLOWED, fit_predict_v6
from literature import (SPLITS, apply_literature, document_virus_context,
                        source_context_rejections, structure_key)
from experiments.literature_candidates import candidate_lookup


def prepared_lookup(target):
    # All lookup/provenance decisions use structure, context and fixed sources.
    _, audit = apply_literature(target[ALLOWED], np.zeros(len(target)))
    extra = candidate_lookup(target[ALLOWED])
    rejection = source_context_rejections(target[ALLOWED], audit)
    return audit, extra, rejection


def combine(raw, audit, extra, rejection):
    baseline = np.asarray(raw, dtype=float).copy()
    known = audit.prediction.notna().to_numpy()
    baseline[known] = audit.loc[known, 'prediction']
    new = extra.eligible_new_match.to_numpy()
    baseline[new] = extra.loc[new, 'prediction']
    guarded = baseline.copy()
    rejected = rejection.reject_numerical_lookup.to_numpy()
    # Abstain on retrieval; use the model, never a fabricated activity label.
    guarded[rejected] = np.asarray(raw)[rejected]
    return baseline, guarded


def run(output, confirm=False, calibrated=False):
    output = Path(output)
    if output.resolve() == (ROOT/'results').resolve():
        raise ValueError('Use a candidate directory')
    output.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(ROOT/'data/train.csv', usecols=ALLOWED+['SI VACV'])
    target = pd.read_csv(ROOT/'data/test.csv', usecols=['index']+ALLOWED)
    y = training['SI VACV'].to_numpy(float)
    canonical = training.SMILES.map(structure_key).to_numpy()
    scaffolds = training.SMILES.map(
        lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
    lookup, extra, rejection = prepared_lookup(training)
    matched = rejection.reject_numerical_lookup.to_numpy()
    reference = pd.read_csv(ROOT/'results/validation.csv')
    team_reference = json.loads((ROOT/'results/literature_candidate_experiment.json').read_text())
    np.testing.assert_array_equal(reference.target, y)
    splits = SPLITS + (['13', '73', '811'] if confirm else [])
    scores, changes = {}, []
    model_settings = dict(smiles_alpha=.003, smiles_slope=2., inactive_odds_power=1.5) if calibrated else {}
    oof = pd.DataFrame({'target': y, 'canonical_smiles': canonical})
    for split in splits:
        seed = int(split) if split != 'scaffold' else None
        groups = canonical if seed is not None else scaffolds
        raw = np.full(len(y), np.nan)
        candidate_raw = np.full(len(y), np.nan)
        for ix, va in GroupKFold(5, shuffle=seed is not None, random_state=seed).split(
                training, groups=groups):
            assert not set(canonical[ix]) & set(canonical[va])
            raw[va], _ = fit_predict_v6(training.iloc[ix], y[ix], training.iloc[va])
            if calibrated:
                candidate_raw[va], _ = fit_predict_v6(training.iloc[ix], y[ix], training.iloc[va], **model_settings)
            else:
                candidate_raw[va] = raw[va]
        baseline, pure_guard = combine(raw, lookup, extra, rejection)
        _, pred = combine(candidate_raw, lookup, extra, rejection)
        if split in SPLITS:
            # Independently verify both the selected V7 and teammate baseline.
            v7 = raw.copy()
            known = lookup.prediction.notna().to_numpy()
            v7[known] = lookup.loc[known, 'prediction']
            np.testing.assert_allclose(v7, reference[split], rtol=1e-10, atol=1e-6)
            np.testing.assert_allclose(np.mean((y-baseline)**2),
                                       team_reference['scores'][split], rtol=1e-12)
        scores[split] = dict(baseline_mse=float(np.mean((y-baseline)**2)),
                             candidate_mse=float(np.mean((y-pred)**2)),
                             unchanged_rows_mse=float(np.mean((y[~matched]-pred[~matched])**2)),
                             provenance_only_mse=float(np.mean((y-pure_guard)**2)))
        scores[split]['relative_improvement'] = 1-scores[split]['candidate_mse']/scores[split]['baseline_mse']
        for position in np.flatnonzero(matched):
            changes.append(dict(split=split, source_row=int(position), target=float(y[position]),
                                baseline=float(baseline[position]), candidate=float(pred[position]),
                                source=lookup.iloc[position].source))
        oof['baseline_'+split], oof['candidate_'+split] = baseline, pred
        print(split, scores[split], flush=True)
    tlookup, textra, trejection = prepared_lookup(target)
    raw, _ = fit_predict_v6(training, y, target)
    team_test, _ = combine(raw, tlookup, textra, trejection)
    if calibrated:
        candidate_raw, model = fit_predict_v6(training, y, target, **model_settings)
    else:
        candidate_raw, model = raw, None
    _, pred = combine(candidate_raw, tlookup, textra, trejection)
    if model is not None:
        import joblib
        joblib.dump(model, output/'model.joblib')
    saved_team = pd.read_csv(ROOT/'results/candidate_literature2019.csv')
    np.testing.assert_array_equal(saved_team['index'], target['index'])
    np.testing.assert_allclose(team_test, saved_team['SI VACV'], rtol=1e-10, atol=1e-6)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    submission = pd.DataFrame({'index': target['index'], 'SI VACV': pred})
    submission.to_csv(output/'submission.csv', index=False)
    oof.to_csv(output/'validation.csv', index=False)
    audit = pd.concat([target[['index']].reset_index(drop=True), tlookup,
                       trejection], axis=1)
    audit['teammate_prediction'], audit['candidate_prediction'] = team_test, pred
    audit.to_csv(output/'test_lookup_audit.csv', index=False)
    train_audit = pd.concat([training.reset_index(drop=True), lookup, rejection], axis=1)
    train_audit.to_csv(output/'training_used.csv', index=False)
    mean = lambda key, names: float(np.mean([scores[s][key] for s in names]))
    ordinary = SPLITS[:-1]
    old_mean, new_mean = mean('baseline_mse', ordinary), mean('candidate_mse', ordinary)
    report = dict(model='Source-provenance abstention over teammate literature-2019 candidate',
        status='local_only_not_submitted', model_settings=model_settings, scores=scores, training_changes=changes,
        mean_baseline_mse=old_mean, mean_candidate_mse=new_mean,
        relative_mean_improvement=1-new_mean/old_mean,
        target_30_percent_met=bool(new_mean < .7*old_mean),
        affected_training_rows=int(matched.sum()),
        affected_test_rows=int(trejection.reject_numerical_lookup.sum()),
        source_coverage={k:sorted(v) for k,v in document_virus_context().items()},
        rule='For exact numerical ChEMBL retrieval, abstain if extra virus species named in target strain context are not all documented in at least one candidate source. Use unchanged in-fold V6 model prediction. Keep existing primary-table mappings and teammate additions.',
        selection='Candidate only; never overwrites selected V7 and never submits to Kaggle.',
        limitations=[
            'Rule chosen after inspecting training errors, including the influential row; it is exploratory.',
            'The provenance component gains from one training observation. Calibration also changes model predictions on unmatched molecules.',
            'Calibration alpha .003, slope 2 and odds power 1.5 were fixed from a previous experiment, not tuned on the three additional splits.',
            'Additional seed checks reuse the same molecules and labels; not an independent holdout.',
            'Extra virus names are an imperfect provenance cue. Missing source coverage is not proof the original measurement is wrong.',
            'Literature retrieval includes published validation molecules, so scores are not novel-compound generalization.',
            'Public/private improvement is unknown. No new Kaggle submission is authorized.',
        ],
        artifact_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'data/train.csv', ROOT/'data/test.csv', ROOT/'data/sources.json',
                      ROOT/'data/chembl_snapshot.zip', ROOT/'data/primary_literature.csv',
                      ROOT/'data/literature_candidates.csv', ROOT/'results/candidate_literature2019.csv']},
        candidate_submission_sha256=hashlib.sha256((output/'submission.csv').read_bytes()).hexdigest())
    if confirm:
        held_names = ['13', '73', '811']
        report['confirmation_mean_baseline_mse'] = mean('baseline_mse', held_names)
        report['confirmation_mean_candidate_mse'] = mean('candidate_mse', held_names)
        report['confirmation_relative_improvement'] = 1-report['confirmation_mean_candidate_mse']/report['confirmation_mean_baseline_mse']
    (output/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:report[k] for k in ['mean_baseline_mse','mean_candidate_mse',
        'relative_mean_improvement','affected_training_rows','affected_test_rows']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'run_provenance_guard')
    parser.add_argument('--confirm', action='store_true')
    parser.add_argument('--calibrated', action='store_true', help='Use fixed alpha .003 / slope 2 / odds power 1.5')
    args = parser.parse_args()
    run(args.output, args.confirm, args.calibrated)
