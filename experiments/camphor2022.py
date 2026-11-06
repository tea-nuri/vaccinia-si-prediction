"""Local-only incremental primary-data experiment over the audited V8 candidate."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import ALLOWED
from literature import SPLITS, lookup_predictions
from experiments.literature_candidates import candidate_lookup, main as reproduce_v8


def audit_matches(target):
    previous = candidate_lookup(target)
    existing = previous.existing_v7_prediction.copy()
    existing.loc[previous.eligible_new_match] = previous.loc[previous.eligible_new_match, 'prediction']
    extra = lookup_predictions(target, numerical=pd.read_csv(ROOT/'data/external.csv').iloc[:0],
                               primary=pd.read_csv(ROOT/'data/camphor2022.csv'))
    extra['existing_prediction'] = existing
    extra['eligible_new_match'] = extra.prediction.notna() & existing.isna()
    extra['overlap_skipped'] = extra.prediction.notna() & existing.notna()
    extra['conflicting_measurement'] = extra.overlap_skipped & (
        (extra.prediction-existing).abs() > np.maximum(1., .05*existing.abs()))
    return extra


def main(output):
    output = Path(output)
    if output.resolve() == (ROOT/'results').resolve():
        raise ValueError('Use an experiment directory; selected V7 is frozen')
    output.mkdir(parents=True, exist_ok=True)
    reproduce_v8(output/'baseline_v8')
    baseline_dir = output/'baseline_v8'
    expected = json.loads((ROOT/'results/literature_candidate_experiment.json').read_text())
    assert hashlib.sha256((baseline_dir/'submission.csv').read_bytes()).hexdigest() == expected['candidate_submission_sha256']
    training = pd.read_csv(ROOT/'data/train.csv', usecols=ALLOWED+['SI VACV'])
    test = pd.read_csv(ROOT/'data/test.csv', usecols=ALLOWED+['index'])
    y = training['SI VACV'].to_numpy()
    oof = pd.read_csv(baseline_dir/'validation.csv')
    submission = pd.read_csv(baseline_dir/'submission.csv')
    train_audit, test_audit = audit_matches(training), audit_matches(test)
    matched = train_audit.eligible_new_match.to_numpy()
    test_matched = test_audit.eligible_new_match.to_numpy()
    previous, scores = {}, {}
    for split in SPLITS:
        previous[split] = float(np.mean((oof[split]-y)**2))
        oof.loc[matched, split] = train_audit.loc[matched, 'prediction']
        scores[split] = float(np.mean((oof[split]-y)**2))
    for values in [previous, scores]:
        values['mean_group_mse'] = float(np.mean([values[s] for s in SPLITS[:-1]]))
    submission.loc[test_matched, 'SI VACV'] = test_audit.loc[test_matched, 'prediction']
    assert np.isfinite(submission['SI VACV']).all()
    assert submission['SI VACV'].ge(0).all()
    np.testing.assert_array_equal(submission['index'], test['index'])
    oof.to_csv(output/'validation.csv', index=False)
    submission.to_csv(output/'submission.csv', index=False)
    train_audit['target'] = y
    train_audit.to_csv(output/'train_lookup_audit.csv', index=False)
    test_audit.insert(0, 'index', test['index'])
    test_audit.to_csv(output/'test_lookup_audit.csv', index=False)
    report = dict(model='V9 local: V8 plus exact stereochemical camphor 2022 measurements',
        status='local_only_not_submitted', scores=scores, baseline_v8_scores=previous,
        all_splits_improved=all(scores[s]<previous[s] for s in SPLITS),
        new_train_matches=int(matched.sum()), new_test_matches=int(test_matched.sum()),
        train_overlaps_skipped=int(train_audit.overlap_skipped.sum()),
        test_overlaps_skipped=int(test_audit.overlap_skipped.sum()),
        train_conflicts_skipped=int(train_audit.conflicting_measurement.sum()),
        test_conflicts_skipped=int(test_audit.conflicting_measurement.sum()),
        candidate_submission_sha256=hashlib.sha256((output/'submission.csv').read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest()
                       for n in ['train.csv','test.csv','camphor2022.csv','literature_candidates.csv']},
        limitations=['Six new training molecules with published measurements; all splits reuse the same molecules.',
                    'Retrieval-assisted exploratory validation, not novel-compound generalization or a public-score result.',
                    'No override of old measurements, even when a later study disagrees.',
                    'No stereochemistry stripping, no NT-to-inactive conversion, no competition concentration columns.'])
    (output/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'run_candidate_v9')
    main(parser.parse_args().output)
