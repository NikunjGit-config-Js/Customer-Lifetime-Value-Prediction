"""Tune on disjoint development customers, then evaluate untouched test customers."""

import json
import platform
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, silhouette_score)
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from scripts.model_features import (FutureTierClassifier, RFMSegmentFeatures, TIER_NAMES,
                                   learned_boundary, tier_accuracy, tier_macro_f1)
from scripts.prepare_data import FEATURES, ROOT, dump_json


def evaluate(y_true, predicted):
    return {'accuracy': float(accuracy_score(y_true, predicted)),
            'macro_f1': float(f1_score(y_true, predicted, labels=[0, 1, 2], average='macro', zero_division=0)),
            'classification_report': classification_report(y_true, predicted, labels=[0, 1, 2],
                                                         target_names=TIER_NAMES, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, predicted, labels=[0, 1, 2]).tolist()}


def main():
    started = time.perf_counter()
    customers = pd.read_csv(ROOT / 'data/processed/customers.csv', index_col='CustomerID')
    # Split on identity only, before looking at future spending. No test stratification.
    train_ids, test_ids = train_test_split(customers.index.to_numpy(), test_size=0.2, random_state=42)
    train = customers.loc[train_ids]
    test = customers.loc[test_ids]
    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train.future_spend, test.future_spend
    assert set(train_ids).isdisjoint(test_ids)
    customers['split'] = np.where(customers.index.isin(train_ids), 'train', 'test')
    customers.to_csv(ROOT / 'data/processed/customers.csv')
    pd.DataFrame({'CustomerID': train_ids}).to_csv(ROOT / 'data/processed/train_ids.csv', index=False)
    pd.DataFrame({'CustomerID': test_ids}).to_csv(ROOT / 'data/processed/test_ids.csv', index=False)

    pipeline = Pipeline([
        ('segments', RFMSegmentFeatures()),
        ('model', XGBClassifier(objective='multi:softprob', eval_metric='mlogloss',
                                tree_method='hist', n_jobs=2, random_state=42)),
    ])
    estimator = FutureTierClassifier(pipeline)
    parameters = {
        'pipeline__model__n_estimators': [100, 200, 300],
        'pipeline__model__max_depth': [2, 3, 4, 5],
        'pipeline__model__learning_rate': [0.03, 0.06, 0.1],
        'pipeline__model__min_child_weight': [3, 8, 15],
        'pipeline__model__subsample': [0.75, 1.0],
        'pipeline__model__colsample_bytree': [0.75, 1.0],
        'pipeline__model__reg_lambda': [1.0, 5.0, 10.0],
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(estimator, parameters, n_iter=12, cv=cv,
                                scoring={'accuracy': tier_accuracy, 'macro_f1': tier_macro_f1,
                                         'spend_boundary': learned_boundary},
                                refit='macro_f1', random_state=42, n_jobs=1, verbose=2,
                                error_score='raise', return_train_score=False)
    print(f'Training {len(train)} customers; holding out {len(test)} customers', flush=True)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    best = search.best_index_
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(ROOT / 'results/cv_results.csv', index=False)
    fold_records = []
    for fold, (fit_rows, validation_rows) in enumerate(cv.split(X_train)):
        fit_spend = y_train.iloc[fit_rows]
        boundary = float(fit_spend[fit_spend > 0].median())
        np.testing.assert_allclose(boundary, cv_results.loc[best, f'split{fold}_test_spend_boundary'])
        fold_records.append({'fold': fold + 1, 'fit_customers': len(fit_rows),
                             'validation_customers': len(validation_rows),
                             'positive_spend_median_gbp': boundary,
                             'accuracy': float(cv_results.loc[best, f'split{fold}_test_accuracy']),
                             'macro_f1': float(cv_results.loc[best, f'split{fold}_test_macro_f1'])})
    pd.DataFrame(fold_records).to_csv(ROOT / 'results/best_cv_folds.csv', index=False)

    train_tiers, test_tiers = model.tier_labels(y_train), model.tier_labels(y_test)
    predicted = model.predict(X_test)
    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, train_tiers)
    baseline_predicted = baseline.predict(X_test)
    joblib.dump(model, ROOT / 'models/cltv_model.joblib')
    joblib.dump(baseline, ROOT / 'models/majority_baseline.joblib')
    reloaded = joblib.load(ROOT / 'models/cltv_model.joblib')
    np.testing.assert_array_equal(predicted, reloaded.predict(X_test))
    np.testing.assert_allclose(model.predict_proba(X_test), reloaded.predict_proba(X_test))
    prediction_table = pd.DataFrame({'CustomerID': test.index, 'future_spend': y_test.to_numpy(),
                                     'actual_tier': test_tiers, 'predicted_tier': predicted,
                                     'baseline_tier': baseline_predicted})
    for tier in range(3):
        prediction_table[f'probability_{tier}'] = model.predict_proba(X_test)[:, tier]
    prediction_table.to_csv(ROOT / 'results/test_predictions.csv', index=False)
    dump_json(ROOT / 'results/best_parameters.json', search.best_params_)
    dump_json(ROOT / 'models/model_metadata.json', {
        'features': FEATURES, 'tier_names': TIER_NAMES,
        'positive_spend_median_gbp': model.positive_spend_median_,
        'tier_rule': '0 if spend=0; 1 if 0<spend<=training_positive_median; 2 otherwise',
        'prediction_cutoff': '2011-09-01', 'target_end_exclusive': '2011-12-01',
        'train_customers': len(train), 'test_customers': len(test), 'random_state': 42})

    segmenter = model.pipeline_.named_steps['segments']
    coordinates = segmenter.coordinates(customers[FEATURES])
    labels = segmenter.segment_labels(customers[FEATURES])
    segments = customers[FEATURES + ['split']].copy()
    segments['segment'] = labels
    segments['pc1'], segments['pc2'] = coordinates[:, 0], coordinates[:, 1]
    segments.to_csv(ROOT / 'data/processed/customer_segments.csv')
    # Profiles use training history only; test customers are assigned without refitting.
    segment_train = segments.loc[train_ids]
    profiles = segment_train.groupby('segment').agg(
        customers=('recency_days', 'size'), recency_days=('recency_days', 'median'),
        orders=('frequency', 'median'), historical_spend_gbp=('monetary', 'median'),
        tenure_days=('tenure_days', 'median'), cadence_days=('mean_gap_days', 'median'))
    overall = segment_train[['recency_days', 'frequency', 'monetary']].median()
    names = []
    for _, row in profiles.iterrows():
        activity = 'Recent' if row.recency_days <= overall.recency_days else 'Lapsed'
        value = 'higher value' if row.historical_spend_gbp >= overall.monetary else 'lower value'
        cadence = 'repeat' if row.orders > 1 else 'single order'
        names.append(f'{activity}, {value}, {cadence}')
    profiles['profile'] = names
    profiles.to_csv(ROOT / 'results/segment_profiles.csv')
    importances = pd.Series(model.pipeline_.named_steps['model'].feature_importances_,
                            index=segmenter.get_feature_names_out(), name='importance').sort_values(ascending=False)
    importances.rename_axis('feature').to_csv(ROOT / 'results/feature_importance.csv')
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(test_tiers, predicted, display_labels=TIER_NAMES,
                                            ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Held-out customers: XGBoost future-spending tiers')
    fig.tight_layout()
    fig.savefig(ROOT / 'results/figures/confusion_matrix.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.head(15).sort_values().plot.barh(ax=ax)
    ax.set(xlabel='XGBoost split gain importance (not causal)', title='Top model features')
    fig.tight_layout()
    fig.savefig(ROOT / 'results/figures/feature_importance.png', dpi=150)
    plt.close(fig)
    silhouette = silhouette_score(segment_train[['pc1', 'pc2']], segment_train.segment,
                                  sample_size=min(2000, len(segment_train)), random_state=42)
    metrics = {
        'seed': 42, 'eligible_customers': len(customers), 'train_customers': len(train),
        'test_customers': len(test), 'customer_overlap': 0,
        'cv': {'folds': 5, 'candidates': 12, 'total_fits': 60, 'refit_metric': 'macro_f1',
               'accuracy_mean': float(cv_results.loc[best, 'mean_test_accuracy']),
               'accuracy_std': float(cv_results.loc[best, 'std_test_accuracy']),
               'macro_f1_mean': float(cv_results.loc[best, 'mean_test_macro_f1']),
               'macro_f1_std': float(cv_results.loc[best, 'std_test_macro_f1']),
               'note': 'Chosen candidate CV scores are tuning estimates; test remains independent.'},
        'xgboost_test': evaluate(test_tiers, predicted),
        'majority_baseline_test': evaluate(test_tiers, baseline_predicted),
        'train_tier_counts': np.bincount(train_tiers, minlength=3).tolist(),
        'test_tier_counts': np.bincount(test_tiers, minlength=3).tolist(),
        'positive_spend_median_gbp': model.positive_spend_median_,
        'segmentation': {'fit_customers': len(train), 'assigned_customers': len(customers),
                         'clusters': 4, 'pca_components': 2,
                         'pca_explained_variance_ratio': segmenter.pca_.explained_variance_ratio_.tolist(),
                         'train_silhouette_pca_space': float(silhouette)},
        'saved_model_reload_prediction_match': True,
        'historical_accuracy_above_85_reproduced': bool(accuracy_score(test_tiers, predicted) > 0.85),
        'training_seconds': time.perf_counter() - started, 'python': platform.python_version(),
    }
    dump_json(ROOT / 'results/metrics.json', metrics)
    pd.DataFrame([{'model': name, 'accuracy': result['accuracy'], 'macro_f1': result['macro_f1']}
                  for name, result in [('XGBoost', metrics['xgboost_test']),
                                       ('Majority baseline', metrics['majority_baseline_test'])]]).to_csv(
        ROOT / 'results/benchmark.csv', index=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
