"""Check real artifacts, feature timing, train isolation, predictions and metrics."""

import hashlib
import json
import subprocess
import sys

import joblib
import nbformat
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold

from scripts.model_features import RFMSegmentFeatures, TIER_NAMES
from scripts.predict import predict_file
from scripts.prepare_data import CUTOFF, FEATURES, ROOT, build_features, dump_json


def main():
    checks = []
    dependency_check = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                                      capture_output=True, text=True, check=True)
    print(dependency_check.stdout.strip(), flush=True)
    checks.append('Installed dependencies pass pip check')
    audit = json.loads((ROOT / 'results/data_audit.json').read_text())
    metrics = json.loads((ROOT / 'results/metrics.json').read_text())
    customers = pd.read_csv(ROOT / 'data/processed/customers.csv', index_col='CustomerID')
    transactions = pd.read_csv(ROOT / 'data/processed/transactions.csv', parse_dates=['InvoiceDate'])
    train = customers.loc[customers.split == 'train']
    test = customers.loc[customers.split == 'test']
    model = joblib.load(ROOT / 'models/cltv_model.joblib')
    assert audit['raw_rows'] == 541909
    assert len(transactions) == audit['clean_rows']
    assert transactions.CustomerID.nunique() == audit['clean_customers']
    assert len(customers) == audit['eligible_customers'] == metrics['eligible_customers']
    assert audit['raw_rows'] - sum(audit['removed_rows_sequential'].values()) == len(transactions)
    assert (transactions.Quantity > 0).all() and (transactions.UnitPrice > 0).all()
    assert transactions.CustomerID.notna().all() and not transactions.duplicated().any()
    checks.append('Official UCI row count, sequential cleaning totals and real customer counts reconcile')

    assert train.index.is_unique and test.index.is_unique
    assert set(train.index).isdisjoint(test.index)
    assert len(train) == metrics['train_customers'] and len(test) == metrics['test_customers']
    np.testing.assert_array_equal(sorted(train.index), sorted(model.fit_customer_ids_))
    segmenter = model.pipeline_.named_steps['segments']
    np.testing.assert_array_equal(sorted(train.index), sorted(segmenter.fit_customer_ids_))
    checks.append('Customer-disjoint outer split; final target calibration and segment fitting use training IDs only')

    # Removing every future row must leave every historical feature unchanged.
    historical_only = transactions.loc[transactions.InvoiceDate < CUTOFF]
    rebuilt = build_features(historical_only)
    pd.testing.assert_frame_equal(rebuilt, customers[FEATURES], check_dtype=False,
                                  check_exact=False, rtol=1e-9, atol=1e-8)
    assert 'future_spend' not in segmenter.get_feature_names_out()
    assert 'CustomerID' not in segmenter.get_feature_names_out()
    assert 'split' not in segmenter.get_feature_names_out()
    expected_boundary = train.loc[train.future_spend > 0, 'future_spend'].median()
    np.testing.assert_allclose(model.positive_spend_median_, expected_boundary)
    # Fitted scaler statistics must agree with training history, never all customers.
    expected_mean = np.log1p(train[['recency_days', 'frequency', 'monetary']]).mean().to_numpy()
    np.testing.assert_allclose(segmenter.scaler_.mean_, expected_mean)
    checks.append('All historical features rebuild identically without future transactions; target/ID allow-list enforced')
    checks.append('Final positive-spend boundary and RFM scaler statistics recompute from training customers only')

    fold_results = pd.read_csv(ROOT / 'results/best_cv_folds.csv')
    train_order = pd.read_csv(ROOT / 'data/processed/train_ids.csv').CustomerID
    ordered_train = customers.loc[train_order]
    for fold, (fit_rows, validation_rows) in enumerate(KFold(5, shuffle=True, random_state=42).split(ordered_train)):
        fold_train, fold_valid = ordered_train.iloc[fit_rows], ordered_train.iloc[validation_rows]
        expected = fold_train.loc[fold_train.future_spend > 0, 'future_spend'].median()
        np.testing.assert_allclose(expected, fold_results.loc[fold, 'positive_spend_median_gbp'])
        transformer = clone(RFMSegmentFeatures()).fit(fold_train[FEATURES])
        assert set(transformer.fit_customer_ids_).isdisjoint(fold_valid.index)
        before = transformer.scaler_.mean_.copy()
        assert transformer.transform(fold_valid[FEATURES]).shape[0] == len(fold_valid)
        np.testing.assert_array_equal(before, transformer.scaler_.mean_)
        assert set(fold_train.index).isdisjoint(test.index)
    checks.append('Five fold boundaries recompute correctly; fold-local segmenters exclude validation and test IDs')

    future = transactions.loc[(transactions.InvoiceDate >= CUTOFF) &
                               (transactions.InvoiceDate < pd.Timestamp('2011-12-01'))]
    expected_spending = future.groupby('CustomerID').revenue.sum().reindex(customers.index, fill_value=0)
    np.testing.assert_allclose(expected_spending, customers.future_spend)
    all_tiers = model.tier_labels(customers.future_spend)
    assert all_tiers[customers.future_spend.eq(0)].max() == 0
    assert model.tier_labels([0, expected_boundary, expected_boundary + 0.01]).tolist() == [0, 1, 2]
    checks.append('Future-spending totals and tier boundary edge cases reconcile with real transactions')

    predictions = pd.read_csv(ROOT / 'results/test_predictions.csv').set_index('CustomerID')
    assert set(predictions.index) == set(test.index) and predictions.index.is_unique
    test_aligned = test.loc[predictions.index]
    np.testing.assert_array_equal(predictions.predicted_tier, model.predict(test_aligned[FEATURES]))
    np.testing.assert_array_equal(predictions.actual_tier, model.tier_labels(test_aligned.future_spend))
    np.testing.assert_allclose(predictions.future_spend, test_aligned.future_spend)
    for key, column in [('xgboost_test', 'predicted_tier'), ('majority_baseline_test', 'baseline_tier')]:
        predicted = predictions[column]
        np.testing.assert_allclose(accuracy_score(predictions.actual_tier, predicted), metrics[key]['accuracy'])
        np.testing.assert_allclose(f1_score(predictions.actual_tier, predicted, average='macro'), metrics[key]['macro_f1'])
        np.testing.assert_array_equal(confusion_matrix(predictions.actual_tier, predicted, labels=[0, 1, 2]),
                                      metrics[key]['confusion_matrix'])
    baseline = joblib.load(ROOT / 'models/majority_baseline.joblib')
    np.testing.assert_array_equal(predictions.baseline_tier, baseline.predict(test_aligned[FEATURES]))
    checks.append('Reloaded model/baseline predictions, held-out accuracy, macro F1 and confusion matrices match saved results')

    # Exercise inference from an actual feature-only CSV; no target column is present.
    feature_path = ROOT / 'data/processed/prediction_features.csv'
    customers[FEATURES].to_csv(feature_path)
    fresh = predict_file(feature_path, ROOT / 'results/verified_predictions.csv', ROOT / 'models/cltv_model.joblib')
    original = pd.read_csv(ROOT / 'results/customer_predictions.csv')
    pd.testing.assert_frame_equal(fresh, original, check_dtype=False, check_exact=False, rtol=1e-6)
    assert len(fresh) == len(customers)
    np.testing.assert_allclose(fresh[[f'probability_{i}' for i in range(3)]].sum(axis=1), 1, atol=1e-6)
    checks.append('Feature-only CSV inference reproduces all customer predictions; probabilities sum to one')

    cv = pd.read_csv(ROOT / 'results/cv_results.csv')
    assert len(cv) == 12 and np.isfinite(cv['mean_test_macro_f1']).all()
    assert all(f'split{i}_test_accuracy' in cv for i in range(5))
    assert len(pd.read_csv(ROOT / 'results/segment_profiles.csv')) == 4
    segments = pd.read_csv(ROOT / 'data/processed/customer_segments.csv')
    assert len(segments) == len(customers)
    assert set(segments.segment) == {0, 1, 2, 3}
    checks.append('Twelve completed five-fold XGBoost candidates and four genuine behavioral segments verified')

    notebook_evidence = []
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code = [cell for cell in notebook.cells if cell.cell_type == 'code']
        assert all(cell.execution_count is not None for cell in code)
        assert not any(out.output_type == 'error' for cell in code for out in cell.outputs)
        images = sum('image/png' in out.get('data', {}) for cell in code for out in cell.outputs)
        assert images >= 2
        notebook_evidence.append({'name': path.name, 'code_cells': len(code), 'inline_images': images})
    assert len(notebook_evidence) == 2
    figures = sorted((ROOT / 'results/figures').glob('*.png'))
    assert len(figures) >= 8
    for path in figures:
        with Image.open(path) as image:
            assert image.width >= 500 and image.height >= 300
            assert max(ImageStat.Stat(image.convert('RGB')).stddev) > 5
    checks.append('Both notebooks fully executed with inline PNGs; all saved plots decode and contain nonblank image content')
    verification = {'status': 'PASS', 'checks_passed': len(checks), 'checks': checks,
                    'notebooks': notebook_evidence, 'png_figures': len(figures),
                    'customers': len(customers), 'train_customers': len(train), 'test_customers': len(test),
                    'model_sha256': hashlib.sha256((ROOT / 'models/cltv_model.joblib').read_bytes()).hexdigest()}
    dump_json(ROOT / 'results/verification.json', verification)
    write_report(audit, metrics, verification)
    artifacts = []
    for directory in ['data/processed', 'models', 'notebooks', 'results']:
        for path in sorted((ROOT / directory).rglob('*')):
            if not path.is_file() or path.name in {'execution.log', 'artifact_manifest.csv'}:
                continue
            artifacts.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(artifacts).to_csv(ROOT / 'results/artifact_manifest.csv', index=False)
    print(json.dumps(verification, indent=2), flush=True)


def write_report(audit, metrics, verification):
    xgb, base, cv = metrics['xgboost_test'], metrics['majority_baseline_test'], metrics['cv']
    report = f'''# Execution report

Generated from actual saved predictions and fitted models; verification: **{verification['status']}**.

- Source: official [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online-retail), CC BY 4.0.
- Original workspace: empty; historical dataset, code, target definition and split unavailable.
- Raw: {audit['raw_rows']:,} line items / {audit['raw_customers']:,} identified customers.
- Clean: {audit['clean_rows']:,} positive transaction lines / {audit['clean_customers']:,} customers.
- Eligible before 1 September 2011: {audit['eligible_customers']:,} customers.
- Split: {metrics['train_customers']:,} training / {metrics['test_customers']:,} unseen test customers; zero overlap.
- Model search: 12 XGBoost candidates, five shuffled customer folds, 60 fits plus best-model refit.
- Selected candidate CV accuracy: {cv['accuracy_mean']:.4f} +/- {cv['accuracy_std']:.4f}; macro F1: {cv['macro_f1_mean']:.4f} +/- {cv['macro_f1_std']:.4f}.
- Test XGBoost accuracy: {xgb['accuracy']:.4f}; macro F1: {xgb['macro_f1']:.4f}.
- Test majority baseline accuracy: {base['accuracy']:.4f}; macro F1: {base['macro_f1']:.4f}.
- Full-training positive-spender median boundary: GBP {metrics['positive_spend_median_gbp']:.2f}.
- Verification: {verification['checks_passed']} checks; {verification['png_figures']} decoded nonblank PNGs; two executed notebooks.

| Test tier | Precision | Recall | F1 | Customers |
|---|---:|---:|---:|---:|
'''
    for name in TIER_NAMES:
        row = xgb['classification_report'][name]
        report += f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |\n"
    report += f'''
## Resume claim audit

1. **RFM/PCA/K-Means over 3,000+ customers:** RFM built for {audit['eligible_customers']:,}; PCA/K-Means fit on {metrics['train_customers']:,} training customers and used to assign all {audit['eligible_customers']:,}. The stricter claim of fitting clustering on over 3,000 customers is {'met' if metrics['train_customers'] > 3000 else 'not met'} because test customers are excluded from fitting.
2. **Behavioral cohorts as modeling features:** reproduced; four one-hot segment indicators, two PCA coordinates and four centroid distances enter XGBoost inside each fold.
3. **Transaction, tenure and cadence features:** reproduced; 19 historical inputs include order counts/values, recent activity, first-observed tenure, inter-order gaps and product breadth.
4. **XGBoost + five-fold RandomizedSearchCV:** reproduced. Historical held-out accuracy >85%: **{'reproduced' if metrics['historical_accuracy_above_85_reproduced'] else 'not reproduced'}**. No test-based retuning or dataset filtering was used to chase this claim.

## Interpretation and limitations

The target is 91-day gross positive purchase revenue, not lifetime value, profit or net revenue. No-purchase customers are retained. Refunds/cancellations are excluded rather than netted; positive service/postage lines are retained. Full December is unavailable, so the target ends before 1 December. Tenure is left-censored at the dataset start. New customers first appearing after the cutoff are excluded. Generalization is assessed across unseen customers at one shared cutoff, not across multiple future economic periods. Four clusters are a fixed descriptive choice, not a claim of an optimal segmentation. The 12-candidate search is modest and selected CV scores are optimistic tuning estimates.

Boundaries use zero plus the training-only median among positive future spenders. Each CV fit recalibrates the median from its training fold, then labels validation outcomes with that boundary. Refit recalibrates from all training customers; test outcomes cannot affect it. Macro F1 selects the model. Historical project equivalence cannot be established without its original artifacts.

See `execution.log`, `verification.json`, `data_audit.json`, `metrics.json`, `best_cv_folds.csv`, `cv_results.csv`, `best_parameters.json`, `segment_profiles.csv`, `test_predictions.csv`, `customer_predictions.csv`, and `figures/` for computed evidence.

Reproduce from repository root: `python run_project.py` (Python 3.12 recommended).
'''
    (ROOT / 'results/report.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
