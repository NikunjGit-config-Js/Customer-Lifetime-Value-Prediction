# Data provenance

No original data or code existed in this workspace at initial inspection. This is a transparent reconstruction of the described methodology, not a verified rerun of the historical project.

Source: Daqing Chen (2015), [Online Retail, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online-retail), DOI [10.24432/C5BW33](https://doi.org/10.24432/C5BW33), **CC BY 4.0**. The official archive is downloaded from `https://archive.ics.uci.edu/static/public/352/online+retail.zip`. The workbook contains item-level transactions from a UK online retailer between December 2010 and December 2011; amounts are in GBP.

`raw/` holds the original ZIP and workbook. `processed/transactions.csv` holds actual cleaned positive transaction lines. These larger, regenerable files are excluded from Git. Compact fitted models, customer tables and computed results are retained. Raw data SHA-256 checksums and exact sequential cleaning counts are saved in `results/data_audit.json`.

Cleaning removes missing customer IDs, exact duplicate lines, cancelled invoices, nonpositive quantities/prices and invalid dates, in that order. Missing product descriptions are retained when the transaction is otherwise usable. Positive postage/service charges and extreme genuine purchases are retained. Refunds are excluded rather than subtracted, so revenue is gross positive purchase revenue. Exact duplicate removal is an assumption that repeated identical rows are accidental.

Feature history is `[2010-12-01, 2011-09-01)`. Future spending is `[2011-09-01, 2011-12-01)`: 91 days. Customers need at least one historical purchase. Customers who do not return remain in the target with zero spending. Customers first observed after the cutoff are excluded. Partial December is excluded to avoid an incomplete calendar month.

Processed tables:

- `customers.csv`: one row per historical customer, 19 historical features, future-spending target and outer split.
- `train_ids.csv`, `test_ids.csv`: explicit disjoint customer identities in split order.
- `customer_segments.csv`: historical features, split, frozen segment assignment and PCA coordinates. No future-spending feature.
- `prediction_features.csv`: actual historical features without the target, used to verify saved-model inference.

Future revenue is a finite-horizon CLTV proxy, not lifetime revenue, profit or net revenue. Tenure means time since first observed purchase, not true relationship age. The historical dataset, original target thresholds and original split were not provided, so historical equivalence and the resume accuracy cannot be presumed.
