"""The two sklearn adapters needed for fold-local segments and target thresholds."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from scripts.prepare_data import FEATURES

RFM = ['recency_days', 'frequency', 'monetary']
TIER_NAMES = ['No purchase', 'Regular spender', 'High spender']


class RFMSegmentFeatures(TransformerMixin, BaseEstimator):
    """Append PCA coordinates, cluster one-hot labels and centroid distances."""

    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        self.fit_customer_ids_ = X.index.to_numpy().copy()
        self.imputer_ = SimpleImputer(strategy='median').fit(X[FEATURES])
        self.rfm_imputer_ = SimpleImputer(strategy='median').fit(X[RFM])
        rfm = np.log1p(self.rfm_imputer_.transform(X[RFM]))
        self.scaler_ = StandardScaler().fit(rfm)
        scaled = self.scaler_.transform(rfm)
        self.pca_ = PCA(n_components=2, svd_solver='full').fit(scaled)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=20,
                             random_state=self.random_state).fit(self.pca_.transform(scaled))
        return self

    def coordinates(self, X):
        check_is_fitted(self, 'kmeans_')
        rfm = np.log1p(self.rfm_imputer_.transform(X[RFM]))
        return self.pca_.transform(self.scaler_.transform(rfm))

    def segment_labels(self, X):
        return self.kmeans_.predict(self.coordinates(X))

    def transform(self, X):
        coordinates = self.coordinates(X)
        labels = self.kmeans_.predict(coordinates)
        return np.column_stack([self.imputer_.transform(X[FEATURES]), coordinates,
                                np.eye(self.n_clusters)[labels], self.kmeans_.transform(coordinates)])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES + ['rfm_pc1', 'rfm_pc2'] +
                          [f'segment_{i}' for i in range(self.n_clusters)] +
                          [f'centroid_distance_{i}' for i in range(self.n_clusters)], dtype=object)


class FutureTierClassifier(ClassifierMixin, BaseEstimator):
    """Learn a spending boundary from fit customers, then fit a classification pipeline.

    y remains actual future spending for CV. Each fit learns its own positive-spend
    median; scorers map validation spending using that fit's boundary. This keeps
    validation and test outcomes out of both tier calibration and feature fitting.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        spending = np.asarray(y, dtype=float)
        positive = spending[spending > 0]
        if not len(positive):
            raise ValueError('Training customers have no positive future spending')
        self.positive_spend_median_ = float(np.median(positive))
        self.fit_customer_ids_ = X.index.to_numpy().copy()
        labels = self.tier_labels(spending)
        if set(labels) != {0, 1, 2}:
            raise ValueError('Training data cannot support all three tiers')
        self.pipeline_ = clone(self.pipeline).fit(X, labels)
        self.classes_ = np.arange(3)
        self.n_features_in_ = X.shape[1]
        return self

    def tier_labels(self, spending):
        check_is_fitted(self, 'positive_spend_median_')
        values = np.asarray(spending, dtype=float)
        return np.where(values <= 0, 0, np.where(values <= self.positive_spend_median_, 1, 2))

    def predict(self, X):
        check_is_fitted(self, 'pipeline_')
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, 'pipeline_')
        return self.pipeline_.predict_proba(X)


def tier_accuracy(estimator, X, spending):
    return accuracy_score(estimator.tier_labels(spending), estimator.predict(X))


def tier_macro_f1(estimator, X, spending):
    return f1_score(estimator.tier_labels(spending), estimator.predict(X),
                    labels=[0, 1, 2], average='macro', zero_division=0)


def learned_boundary(estimator, X, spending):
    return estimator.positive_spend_median_
