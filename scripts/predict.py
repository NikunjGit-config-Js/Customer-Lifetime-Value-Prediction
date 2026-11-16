"""Load the fitted pipeline and predict from historical customer features."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.model_features import TIER_NAMES
from scripts.prepare_data import FEATURES, ROOT


def predict_file(input_path, output_path, model_path):
    features = pd.read_csv(input_path, index_col='CustomerID')
    missing = set(FEATURES) - set(features.columns)
    if missing:
        raise ValueError(f'Missing historical feature columns: {sorted(missing)}')
    if not features.index.is_unique or features.index.hasnans:
        raise ValueError('Customer IDs must be present and unique')
    # Select an explicit allow-list: future_spend, split and IDs cannot become features.
    X = features[FEATURES].apply(pd.to_numeric, errors='raise')
    if np.isinf(X.to_numpy()).any() or (X < 0).any().any():
        raise ValueError('Historical features must be nonnegative and finite (NaN is imputed)')
    model = joblib.load(model_path)
    predictions = model.predict(X)
    result = pd.DataFrame({'CustomerID': features.index, 'predicted_tier': predictions,
                           'predicted_label': np.asarray(TIER_NAMES)[predictions]})
    result['segment'] = model.pipeline_.named_steps['segments'].segment_labels(X)
    for tier, probabilities in enumerate(model.predict_proba(X).T):
        result[f'probability_{tier}'] = probabilities
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f'Loaded {model_path.name}; predicted {len(result)} real customers -> {output_path}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/processed/customers.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/customer_predictions.csv')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/cltv_model.joblib')
    args = parser.parse_args()
    predict_file(args.input, args.output, args.model)


if __name__ == '__main__':
    main()
