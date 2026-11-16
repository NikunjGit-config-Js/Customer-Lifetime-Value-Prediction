"""Download official UCI transactions; create one historical row per customer."""

import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'
CUTOFF = pd.Timestamp('2011-09-01')
TARGET_END = pd.Timestamp('2011-12-01')
OBS_START = pd.Timestamp('2010-12-01')
FEATURES = ['recency_days', 'frequency', 'monetary', 'tenure_days', 'active_span_days',
            'average_order_value', 'order_value_std', 'total_units', 'unique_products',
            'active_months', 'mean_gap_days', 'gap_std_days', 'single_order',
            'orders_last_30d', 'spend_last_30d', 'orders_last_90d', 'spend_last_90d',
            'spend_per_observed_day', 'weekend_order_share']


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def download():
    raw_dir = ROOT / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    workbook = raw_dir / 'Online Retail.xlsx'
    archive = raw_dir / 'online_retail.zip'
    if not archive.exists():
        print(f'Downloading {SOURCE}', flush=True)
        request = urllib.request.Request(SOURCE, headers={'User-Agent': 'CLTV research project'})
        temporary = archive.with_suffix('.download')
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open('wb') as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(temporary) as z:
            if 'Online Retail.xlsx' not in z.namelist():
                raise ValueError('Official archive does not contain expected workbook')
        temporary.replace(archive)
    # Extract only the known workbook, avoiding arbitrary archive paths.
    if not workbook.exists():
        with zipfile.ZipFile(archive) as z, z.open('Online Retail.xlsx') as source, workbook.open('wb') as out:
            shutil.copyfileobj(source, out)
    return workbook, archive


def clean_transactions(raw):
    frame = raw.copy()
    counts = {'raw_rows': len(frame), 'raw_customers': int(frame.CustomerID.nunique())}
    rules = [
        ('missing_customer_id', lambda d: d.CustomerID.notna()),
        ('duplicate_rows', lambda d: ~d.duplicated()),
        ('cancelled_invoices', lambda d: ~d.InvoiceNo.astype(str).str.upper().str.startswith('C')),
        ('nonpositive_quantity', lambda d: d.Quantity.gt(0)),
        ('nonpositive_price', lambda d: d.UnitPrice.gt(0)),
        ('invalid_date', lambda d: pd.to_datetime(d.InvoiceDate, errors='coerce').notna()),
    ]
    removed = {}
    for name, valid in rules:
        keep = valid(frame)
        removed[name] = int((~keep).sum())
        frame = frame.loc[keep].copy()
    frame['InvoiceDate'] = pd.to_datetime(frame.InvoiceDate)
    frame['CustomerID'] = frame.CustomerID.astype('int64')
    frame['InvoiceNo'] = frame.InvoiceNo.astype(str)
    frame['StockCode'] = frame.StockCode.astype(str)
    frame['revenue'] = frame.Quantity * frame.UnitPrice
    counts.update({'removed_rows_sequential': removed, 'clean_rows': len(frame),
                   'clean_customers': int(frame.CustomerID.nunique()),
                   'clean_invoices': int(frame.InvoiceNo.nunique()),
                   'first_transaction': str(frame.InvoiceDate.min()),
                   'last_transaction': str(frame.InvoiceDate.max())})
    return frame.sort_values('InvoiceDate'), counts


def build_features(transactions, cutoff=CUTOFF, observation_start=OBS_START):
    history = transactions.loc[(transactions.InvoiceDate >= observation_start) &
                               (transactions.InvoiceDate < cutoff)].copy()
    orders = history.groupby(['CustomerID', 'InvoiceNo']).agg(
        order_date=('InvoiceDate', 'min'), order_value=('revenue', 'sum')).reset_index()
    orders = orders.sort_values(['CustomerID', 'order_date', 'InvoiceNo'])
    orders['gap_days'] = orders.groupby('CustomerID').order_date.diff().dt.total_seconds() / 86400
    orders['weekend'] = orders.order_date.dt.dayofweek.ge(5).astype(int)
    grouped = orders.groupby('CustomerID')
    features = grouped.agg(frequency=('InvoiceNo', 'nunique'), monetary=('order_value', 'sum'),
                           first_order=('order_date', 'min'), last_order=('order_date', 'max'),
                           average_order_value=('order_value', 'mean'),
                           order_value_std=('order_value', 'std'),
                           mean_gap_days=('gap_days', 'mean'), gap_std_days=('gap_days', 'std'),
                           weekend_order_share=('weekend', 'mean'))
    features['recency_days'] = (cutoff - features.last_order).dt.total_seconds() / 86400
    features['tenure_days'] = (cutoff - features.first_order).dt.total_seconds() / 86400
    features['active_span_days'] = (features.last_order - features.first_order).dt.total_seconds() / 86400
    features['total_units'] = history.groupby('CustomerID').Quantity.sum()
    features['unique_products'] = history.groupby('CustomerID').StockCode.nunique()
    history['month'] = history.InvoiceDate.dt.to_period('M')
    features['active_months'] = history.groupby('CustomerID').month.nunique()
    features['single_order'] = features.frequency.eq(1).astype(int)
    for days in [30, 90]:
        recent = orders.loc[orders.order_date >= cutoff - pd.Timedelta(days=days)].groupby('CustomerID')
        features[f'orders_last_{days}d'] = recent.InvoiceNo.nunique().reindex(features.index, fill_value=0)
        features[f'spend_last_{days}d'] = recent.order_value.sum().reindex(features.index, fill_value=0)
    features['spend_per_observed_day'] = features.monetary / features.tenure_days.clip(lower=1)
    # Missing gaps/std for one-order customers stay missing and are imputed within training folds.
    return features[FEATURES].sort_index()


def main():
    for directory in ['data/processed', 'results', 'results/figures', 'models']:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    workbook, archive = download()
    print('Reading genuine UCI workbook', flush=True)
    raw = pd.read_excel(workbook, engine='openpyxl')
    clean, counts = clean_transactions(raw)
    customers = build_features(clean)
    future = clean.loc[(clean.InvoiceDate >= CUTOFF) & (clean.InvoiceDate < TARGET_END)]
    customers['future_spend'] = future.groupby('CustomerID').revenue.sum().reindex(customers.index, fill_value=0)
    assert customers.index.is_unique
    assert np.isfinite(customers.future_spend).all()
    clean.to_csv(ROOT / 'data/processed/transactions.csv', index=False)
    customers.to_csv(ROOT / 'data/processed/customers.csv')
    history = clean.loc[(clean.InvoiceDate >= OBS_START) & (clean.InvoiceDate < CUTOFF)]
    counts.update({'eligible_customers': len(customers), 'observation_rows': len(history),
                   'observation_invoices': int(history.InvoiceNo.nunique()),
                   'target_rows_all_customers': len(future),
                   'target_rows_eligible_customers': int(future.CustomerID.isin(customers.index).sum()),
                   'future_only_customers_excluded': int((~future.CustomerID.isin(customers.index)).groupby(future.CustomerID).any().sum()),
                   'no_future_purchase_customers': int(customers.future_spend.eq(0).sum()),
                   'observation_start_inclusive': str(OBS_START), 'cutoff_exclusive': str(CUTOFF),
                   'target_start_inclusive': str(CUTOFF), 'target_end_exclusive': str(TARGET_END),
                   'source_url': SOURCE, 'source_page': 'https://archive.ics.uci.edu/dataset/352/online-retail',
                   'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                   'workbook_sha256': hashlib.sha256(workbook.read_bytes()).hexdigest(),
                   'original_workspace_data': 'None found; workspace was empty at initial inspection.',
                   'target': '91-day positive purchase revenue in GBP; excludes refunds, not lifetime profit.'})
    dump_json(ROOT / 'results/data_audit.json', counts)
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
