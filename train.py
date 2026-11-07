"""Entrypoint for corrected train_v2 regression.

Older helper functions below remain for reproducing historical experiments.
The supported CLI delegates to corrected_pipeline and does not use the former
SI=10000 convention or test-compound literature retrieval.
"""
from pathlib import Path
import argparse
import json
import hashlib
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
ALLOWED = ['SMILES', 'strain', 'Culture']
CHARACTERS = list('cCnNoOpPsSFIBr[]=#@/\\123456789%()+-.')


@lru_cache(maxsize=1)
def load_inactive_smiles():
    evidence = pd.read_csv(DATA / 'external_inactivity.csv')
    # A compound reported in several VACV publications can have contradictory
    # assay outcomes. Such evidence is never used for the correction.
    return frozenset(evidence.loc[evidence.single_source == True, 'smiles'])


def evidence_flags(smiles, inactive_smiles=None):
    if inactive_smiles is None:
        inactive_smiles = load_inactive_smiles()
    return np.array([Chem.MolToSmiles(Chem.MolFromSmiles(s)) in inactive_smiles
                     for s in smiles], dtype=bool)


def apply_inactivity_evidence(training, targets, target, prediction, inactive_smiles=None):
    if inactive_smiles is None:
        inactive_smiles = load_inactive_smiles()
    flags = evidence_flags(training.SMILES, inactive_smiles)
    culture = training.Culture.fillna('UNKNOWN').astype(str).to_numpy()
    supported = []
    for cell in sorted(set(culture[flags])):
        peers = flags & (culture == cell)
        # This mapping is learned ONLY from the current training fold.
        # No external annotation is directly converted into a training target.
        if peers.sum() >= 3 and np.all(targets[peers] == 10000):
            supported.append(cell)
    apply = evidence_flags(target.SMILES, inactive_smiles) & target.Culture.fillna(
        'UNKNOWN').astype(str).isin(supported).to_numpy()
    result = np.asarray(prediction, dtype=float).copy()
    result[apply] = 10000
    return result, {'supported_cultures': supported, 'minimum_support': 3,
                    'inactive_smiles': inactive_smiles,
                    'corrected_rows': int(apply.sum())}


def known_compound_key(row):
    """Exact chemical identity (including stereo) and provided assay context."""
    values = []
    for column in ['strain', 'Culture']:
        value = row[column]
        if pd.isna(value) or not str(value).strip() or str(value).casefold() == 'unknown':
            return None
        normalized = ''.join(c for c in str(value).casefold() if c.isalnum())
        if not normalized or normalized == 'unknown':
            return None
        values.append(normalized)
    mol = Chem.MolFromSmiles(row['SMILES'])
    if mol is None:
        raise ValueError('Invalid SMILES')
    return (Chem.MolToSmiles(mol, isomericSmiles=True), *values)


def apply_known_compounds(training, targets, target, prediction):
    """A distance-zero identity regressor; conflicting observations fall back."""
    groups = {}
    for (_, row), y in zip(training.iterrows(), targets):
        key = known_compound_key(row)
        if key is not None:
            groups.setdefault(key, []).append(float(y))
    known = {key:float(np.mean(values)) for key,values in groups.items()
             if np.allclose(values, values[0], rtol=.01, atol=.001)}
    result = np.asarray(prediction, dtype=float).copy()
    corrected = []
    for position, (_, row) in enumerate(target.iterrows()):
        key = known_compound_key(row)
        if key in known:
            result[position] = known[key]
            corrected.append(position)
    return result, dict(known=known, corrected_positions=corrected,
                        conflicting_contexts=len(groups)-len(known))


def make_model():
    return make_pipeline(
        TfidfVectorizer(analyzer='char', ngram_range=(2, 6),
                        lowercase=False, sublinear_tf=True),
        Ridge(alpha=0.03))


def serialization_features(smiles):
    return np.asarray([[len(s)] + [s.count(c) for c in CHARACTERS]
                       + [s.find('c'), s.find('@'), s.count('C1'), s.count('c1')]
                       for s in smiles], dtype=float)


def _predict_smiles(train_smiles, targets, predict_smiles, sample_weight=None):
    kwargs = {} if sample_weight is None else {'ridge__sample_weight': sample_weight}
    text_model = make_model().fit(train_smiles, np.minimum(targets, 10000), **kwargs)
    tree_model = ExtraTreesRegressor(n_estimators=500, max_features=1.0,
                                    min_samples_leaf=1, n_jobs=4, random_state=42)
    tree_model.fit(serialization_features(train_smiles), np.minimum(targets, 10000), sample_weight=sample_weight)
    text_pred = np.clip(text_model.predict(predict_smiles), 0, 10000)
    tree_pred = np.clip(tree_model.predict(serialization_features(predict_smiles)), 0, 10000)
    blended = .85*text_pred+.15*tree_pred
    prediction = np.clip((blended-5000)*1.5+5000, 0, 10000)
    return prediction, {'text':text_model, 'trees':tree_model,
                        'tree_weight':.15, 'slope':1.5, 'center':5000,
                        'characters':CHARACTERS}


def fit_predict_v6(training, targets, target, sample_weight=None, inactive_odds_power=1.):
    if not np.isfinite(inactive_odds_power) or inactive_odds_power <= 0:
        raise ValueError("inactive_odds_power must be positive and finite")
    # The allowlist excludes concentrations, missingness and identifiers even
    # when a caller passes a full competition dataframe.
    training = training[ALLOWED]
    target = target[ALLOWED]
    smiles_pred, smiles_models = _predict_smiles(
        training.SMILES.to_numpy(), targets, target.SMILES.to_numpy(), sample_weight)
    xt = pd.DataFrame(serialization_features(training.SMILES)).astype(float)
    xv = pd.DataFrame(serialization_features(target.SMILES)).astype(float)
    categorical = []
    for column in ['strain', 'Culture']:
        categorical.append(xt.shape[1])
        xt[column] = training[column].fillna('UNKNOWN').astype(str).to_numpy()
        xv[column] = target[column].fillna('UNKNOWN').astype(str).to_numpy()
    inactive = CatBoostClassifier(iterations=700, depth=5, learning_rate=.035,
                                 l2_leaf_reg=5, verbose=False, thread_count=4,
                                 allow_writing_files=False, random_seed=42)
    inactive.fit(xt, (targets == 10000).astype(int), cat_features=categorical, sample_weight=sample_weight)
    probability = inactive.predict_proba(xv)[:, 1]
    if inactive_odds_power != 1.:
        bounded = np.clip(probability, 1e-12, 1-1e-12)
        log_odds = np.log(bounded)-np.log1p(-bounded)
        probability = 1/(1+np.exp(-np.clip(inactive_odds_power*log_odds, -700, 700)))
    inactive_part = 10000 * probability
    active = targets != 10000
    active_model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                      max_features=.5, random_state=42, n_jobs=4)
    active_model.fit(serialization_features(training.loc[active, 'SMILES']),
                     np.minimum(targets[active], 5000),
                     sample_weight=None if sample_weight is None else sample_weight[active])
    active_pred = np.clip(active_model.predict(serialization_features(target.SMILES)), 0, 5000)
    context_pred = np.clip(inactive_part + (1-inactive_part/10000)*active_pred, 0, 10000)
    prediction, evidence_model = apply_inactivity_evidence(
        training, targets, target, .5*smiles_pred+.5*context_pred)
    return prediction, {
        'smiles_models': smiles_models, 'inactive_classifier': inactive,
        'active_regressor': active_model, 'categorical_positions': categorical,
        'context_weight': .5, 'features_allowed': ALLOWED,
        'inactivity_evidence': evidence_model, 'inactive_odds_power': inactive_odds_power}


def fit_predict(training, targets, target, sample_weight=None, known_compounds=False, inactive_odds_power=1., primary_2019=False):
    from literature import apply_literature
    baseline, model = fit_predict_v6(training, targets, target, sample_weight, inactive_odds_power)
    prediction, audit = apply_literature(target, baseline, include_2019=primary_2019)
    qualitative_active = np.zeros(len(target), dtype=bool)
    if primary_2019:
        from literature import qualitative_active_flags
        qualitative_active = qualitative_active_flags(target) & audit.prediction.isna().to_numpy()
        if qualitative_active.any():
            prediction[qualitative_active] = np.clip(model['active_regressor'].predict(
                serialization_features(target.loc[qualitative_active, 'SMILES'])), 0, 5000)
    model['literature_retrieval'] = {
        'corrected_rows': int(audit.prediction.notna().sum()),
        'primary_2019_enabled': primary_2019,
        'qualitative_active_estimates': int(qualitative_active.sum()),
        'method': 'Fixed external measurements matched by structure and assay context',
        'artifact_sha256': {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                            for name in ['external.csv', 'primary_literature.csv',
                                         'primary_structure_evidence.json']}}
    known_model = None
    if known_compounds:
        prediction, known_model = apply_known_compounds(
            training[ALLOWED], targets, target[ALLOWED], prediction)
    model['known_compounds'] = known_model
    return prediction, model


def prepare_external(frame, original_smiles):
    """Keep exact measured SI and exclude every original-train molecule upfront."""
    canonical = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
    original = {canonical(s) for s in original_smiles}
    uncharger = rdMolStandardize.Uncharger()
    def parent_key(s):
        parent = rdMolStandardize.FragmentParent(Chem.MolFromSmiles(s))
        return Chem.MolToSmiles(uncharger.uncharge(parent))
    original_parents = {parent_key(s) for s in original_smiles}
    exact = frame[[c+'_relation' for c in ['IC50','CC50','SI']]].eq('=').all(axis=1)
    clean = frame.quality_flags.fillna('').eq('')
    numeric = frame[['IC50','CC50','SI']].apply(pd.to_numeric, errors='coerce')
    valid = np.isfinite(numeric).all(axis=1) & numeric.gt(0).all(axis=1)
    # A genuinely measured SI of exactly 10000 must not become an inactivity label.
    rows = frame[exact & clean & valid & numeric.SI.ne(10000)].copy()
    rows['canonical'] = rows.smiles.map(canonical)
    rows = rows[~rows.canonical.isin(original)].copy()
    # Counterions/protonation must not sneak an original validation compound
    # into training under a different canonical SMILES. Stereo is preserved.
    rows = rows[~rows.smiles.map(parent_key).isin(original_parents)].copy()
    rows['scaffold'] = rows.canonical.map(
        lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC')
    rows['SMILES'] = rows.smiles
    rows['Culture'] = rows.cell
    rows['strain'] = rows.strain.where(~rows.strain.str.contains('unspecified', case=False), 'UNKNOWN')
    rows['SI VACV'] = pd.to_numeric(rows.SI)
    # Retain separate experimental contexts; do not average incompatible assays.
    return rows.reset_index(drop=True)


def augment(training, targets, external, weight, forbidden_scaffolds=None):
    rows = external.copy()
    if forbidden_scaffolds is not None:
        rows = rows[~rows.scaffold.isin(forbidden_scaffolds)].copy()
    if not weight or rows.empty:
        return training.reset_index(drop=True), targets, None, rows.iloc[:0]
    # A compound with several measurements receives the same total weight as one
    # compound with a single measurement, while its contexts stay separate.
    counts = rows.groupby('canonical').canonical.transform('size').to_numpy()
    weights = np.r_[np.ones(len(training)), weight/counts]
    joined = pd.concat([training[ALLOWED], rows[ALLOWED]], ignore_index=True)
    return joined, np.r_[targets, rows['SI VACV'].to_numpy(float)], weights, rows


def external_ablation(output):
    """Bounded ablation of the current selected pipeline; never overwrites results."""
    output.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(DATA/'train.csv', usecols=ALLOWED+['SI VACV'])
    target = pd.read_csv(DATA/'test.csv', usecols=['index']+ALLOWED)
    external = prepare_external(pd.read_csv(DATA/'external.csv'), training.SMILES)
    y = training['SI VACV'].to_numpy(float)
    canonical = training.SMILES.map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))).to_numpy()
    scaffolds = training.SMILES.map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or 'ACYCLIC').to_numpy()
    previous = json.loads((ROOT/'results/metrics.json').read_text())
    for name, digest in previous['source_sha256'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest
    baseline = pd.read_csv(ROOT/'results/validation.csv')
    np.testing.assert_allclose(baseline.target, y)
    np.testing.assert_array_equal(baseline.canonical_smiles, canonical)
    report = dict(model='selected V7 with weighted external augmentation', original_train_rows=len(training),
                  exclusion_rule='Exclude every original-train canonical and neutralized fragment-parent structure; also exclude validation scaffolds for scaffold CV.',
                  external_candidate_records=len(external),
                  external_candidate_molecules=int(external.canonical.nunique()),
                  external_sha256=hashlib.sha256((DATA/'external.csv').read_bytes()).hexdigest(),
                  baseline_validation_sha256=hashlib.sha256((ROOT/'results/validation.csv').read_bytes()).hexdigest(),
                  selection_rule='Improve both molecule splits and scaffold split; among passing weights choose lowest mean molecule MSE; otherwise retain weight 0.',
                  caveat='Repeated CV on previously explored molecules; not an independent holdout.',
                  weights={})
    predictions = pd.DataFrame({'source_row':np.arange(len(y)), 'target':y})
    report['weights']['0'] = {name:float(np.mean((baseline[name]-y)**2)) for name in ['42','2026','scaffold']}
    for weight in [.1, .3, 1.]:
        metrics = {}
        for seed in [42, 2026, None]:
            name = str(seed) if seed is not None else 'scaffold'
            groups = canonical if seed is not None else scaffolds
            pred = np.full(len(y), np.nan); counts = []
            for fold,(ix,va) in enumerate(GroupKFold(5,shuffle=seed is not None,random_state=seed).split(training,groups=groups)):
                assert not set(canonical[ix]) & set(canonical[va])
                joined, labels, weights, rows = augment(training.iloc[ix],y[ix],external,weight,
                                                        set(scaffolds[va]) if seed is None else None)
                assert not set(rows.canonical) & set(canonical[va])
                pred[va], _ = fit_predict(joined, labels, training.iloc[va], weights)
                counts.append(len(rows))
                print(f'external weight={weight} split={name} fold={fold} records={len(rows)}',flush=True)
            assert np.isfinite(pred).all()
            metrics[name] = float(np.mean((pred-y)**2))
            metrics[name+'_external_records_per_fold'] = counts
            predictions[f'{weight}_{name}'] = pred
            print(weight,name,metrics[name],flush=True)
        report['weights'][str(weight)] = metrics
    passing = [w for w in [.1,.3,1.] if all(report['weights'][str(w)][p] < report['weights']['0'][p]
                                           for p in ['42','2026','scaffold'])]
    selected = min(passing,key=lambda w:np.mean([report['weights'][str(w)][p] for p in ['42','2026']])) if passing else 0.
    report['recommended_external_weight'] = selected
    joined, labels, weights, rows = augment(training,y,external,selected)
    pred, model = fit_predict(joined, labels, target, weights)
    sample = pd.read_csv(DATA/'sample_submit.csv')
    np.testing.assert_array_equal(sample['index'],target['index'])
    assert np.isfinite(pred).all()
    sample['SI VACV'] = pred
    sample.to_csv(output/'submission.csv',index=False)
    used = joined[ALLOWED].copy()
    used['original_SI'] = labels
    used['source'] = ['competition_train']*len(training)+['external']*len(rows)
    used['doi'] = ['']*len(training)+rows.doi.tolist()
    used['sample_weight'] = np.ones(len(used)) if weights is None else weights
    used.to_csv(output/'training_used.csv',index=False)
    predictions.to_csv(output/'validation.csv',index=False)
    joblib.dump(model,output/'model.joblib')
    report['full_fit_external_records'] = len(rows)
    (output/'external_experiment.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


def legacy_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'run')
    parser.add_argument('--skip-cv', action='store_true')
    parser.add_argument('--primary-2019', action='store_true',
                        help='Include qualitative imidazole measurements from Nikitina 2019')
    parser.add_argument('--inactive-odds-power', type=float, default=1.,
                        help='Experimental inactivity calibration; 1 preserves selected V7')
    parser.add_argument('--known-compounds', action='store_true',
                        help='Use observed SI for exact chemical identity and matching context over V7; disabled by default')
    parser.add_argument('--external-ablation', action='store_true',
                        help='Compare selected V7 with external molecule weights 0.1, 0.3 and 1')
    parser.add_argument('--literature-ablation', action='store_true',
                        help='Rebuild the V6 baseline and evaluate traceable literature retrieval')
    args = parser.parse_args()
    if not np.isfinite(args.inactive_odds_power) or args.inactive_odds_power <= 0:
        parser.error("--inactive-odds-power must be positive and finite")
    if (args.inactive_odds_power != 1. or args.primary_2019) and (args.literature_ablation or args.external_ablation):
        parser.error("Calibration cannot be combined with standalone ablations")
    if (args.inactive_odds_power != 1. or args.primary_2019) and args.output.resolve() == (ROOT/"results").resolve():
        parser.error("Use a candidate directory; selected results require a confirmed Kaggle score")
    if args.literature_ablation:
        if args.external_ablation or args.skip_cv or args.known_compounds:
            parser.error('--literature-ablation must be used without other experiment switches')
        from literature import run_experiment
        run_experiment(args.output)
        return
    if args.external_ablation:
        if args.known_compounds:
            parser.error("--known-compounds cannot be combined with --external-ablation")
        if args.skip_cv:
            parser.error('--external-ablation requires CV; omit --skip-cv')
        external_ablation(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    # Only context explicitly permitted by the organizers joins SMILES.
    train = pd.read_csv(DATA / 'train.csv', usecols=ALLOWED+['SI VACV'])
    test = pd.read_csv(DATA / 'test.csv', usecols=['index']+ALLOWED)
    sample = pd.read_csv(DATA / 'sample_submit.csv')
    x = train.SMILES.to_numpy()
    y = train['SI VACV'].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid training target')
    mols = [Chem.MolFromSmiles(s) for s in x]
    assert all(m is not None for m in mols)
    groups = np.array([Chem.MolToSmiles(m) for m in mols])
    scaffolds = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m)
                          or 'ACYCLIC' for m in mols])
    scores = {}
    oof = pd.DataFrame({'target': y, 'canonical_smiles': groups})
    if not args.skip_cv:
        for seed in [42, 2026, 117, 2027, 7, None]:
            label = str(seed) if seed is not None else 'scaffold'
            splitter = GroupKFold(5, shuffle=seed is not None, random_state=seed)
            p = np.zeros(len(y))
            for ix, va in splitter.split(x, y, groups if seed is not None else scaffolds):
                assert not set(groups[ix]) & set(groups[va])
                # Vocabulary and IDF are fitted strictly inside this training fold.
                p[va], _ = fit_predict(train.iloc[ix], y[ix], train.iloc[va], known_compounds=args.known_compounds, inactive_odds_power=args.inactive_odds_power, primary_2019=args.primary_2019)
            scores[label] = float(np.mean((y-p)**2))
            oof[label] = p
            print(label, scores[label], flush=True)
        scores['mean_group_mse'] = float(np.mean([scores[str(s)] for s in [42, 2026, 117, 2027, 7]]))
        oof.to_csv(args.output / 'validation.csv', index=False)
    pred, model = fit_predict(train, y, test, known_compounds=args.known_compounds, inactive_odds_power=args.inactive_odds_power, primary_2019=args.primary_2019)
    used = train[ALLOWED].copy()
    used['original_SI'] = y
    used['inactive_label'] = (y==10000).astype(int)
    used['v4_training_target'] = np.minimum(y,10000)
    used['active_regression_target'] = np.where(y!=10000, np.minimum(y,5000), np.nan)
    used['external_single_source_inactivity'] = evidence_flags(train.SMILES)
    from literature import lookup_predictions
    train_lookup = lookup_predictions(train, include_2019=args.primary_2019)
    for column in ['prediction', 'source', 'match_method']:
        used['literature_'+column] = train_lookup[column].to_numpy()
    if args.primary_2019:
        from literature import qualitative_active_flags
        used['literature_qualitative_active'] = qualitative_active_flags(train)
    used.to_csv(args.output / 'training_used.csv', index=False)
    test_lookup = lookup_predictions(test, include_2019=args.primary_2019)
    test_lookup.insert(0, 'index', test['index'])
    if args.primary_2019:
        test_lookup['qualitative_active_evidence'] = qualitative_active_flags(test)
    test_lookup['final_prediction'] = pred
    test_lookup.to_csv(args.output / 'test_lookup_audit.csv', index=False)
    assert sample.columns.tolist() == ['index', 'SI VACV']
    assert sample['index'].equals(test['index'])
    assert np.isfinite(pred).all() and len(pred) == len(test)
    sample['SI VACV'] = pred
    sample.to_csv(args.output / 'submission.csv', index=False)
    joblib.dump(model, args.output / 'model.joblib')
    report = {
        'model': ('V7 with experimental known-compound correction' if args.known_compounds
                  else 'V7: V6 plus curated primary literature and matched external assay pairs'),
        'inactive_odds_power': args.inactive_odds_power,
        'candidate_status': 'unsubmitted' if args.inactive_odds_power != 1. or args.primary_2019 else 'selected_configuration',
        'literature_retrieval': model['literature_retrieval'],
        'known_compound_correction': dict(enabled=args.known_compounds,
             corrected_positions=[] if model['known_compounds'] is None else model['known_compounds']['corrected_positions'],
             key='canonical isomeric SMILES + case/punctuation-normalized strain and Culture',
             conflicting_contexts=0 if model['known_compounds'] is None else model['known_compounds']['conflicting_contexts']),
        'inactivity_evidence': {k: v for k, v in model['inactivity_evidence'].items()
                                if k != 'inactive_smiles'},
        'external_evidence_sha256': hashlib.sha256((DATA/'external_inactivity.csv').read_bytes()).hexdigest(),
        'context_weight': .5,
        'inactive_classifier': {'iterations':700,'depth':5,'learning_rate':.035,'l2_leaf_reg':5,'seed':42},
        'tree_weight': .15, 'calibration_slope': 1.5, 'calibration_center': 5000,
        'ngram_range': [2, 6], 'alpha': 0.03,
        'training_target_cap': 10000, 'baseline_prediction_bounds': [0, 10000],
        'prediction_bounds': [0, None],
        'features_allowed': ALLOWED, 'train_rows': len(y),
        'test_rows': len(test), 'scores': scores,
        'source_sha256': {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                          for name in ['train.csv', 'test.csv']},
        'limitations': [
            'CV selection uses the same small set of molecules; it is optimistic.',
            'Raw SMILES serialization carries source conventions. Equivalent rewritten SMILES can receive different predictions.',
            'Train targets are capped, but all validation MSE values use the ORIGINAL uncapped targets.',
            'The baseline target cap can underestimate high SI; exact external retrieval is not capped.',
            'Validation includes retrieval of measurements for published molecules, not compound-disjoint external-data generalization.',
            'Sources and curation rules were selected after inspecting training errors; the large training outlier dominates the local improvement.',
            'Rounded external concentration ratios and coarse assay context can differ from competition annotations.',
            'SI=10000 is an inactivity sentinel per organizer clarification, not measured high selectivity.',
            'External inactivity can differ by assay; a single-source guard and in-fold culture mapping reduce but do not eliminate this risk.',
        ],
    }
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))


def main():
    from corrected_pipeline import main as corrected_main
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        corrected_main()


if __name__ == '__main__':
    main()
