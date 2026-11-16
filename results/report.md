# Execution report

Generated from actual saved predictions and fitted models; verification: **PASS**.

- Source: official [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online-retail), CC BY 4.0.
- Original workspace: empty; historical dataset, code, target definition and split unavailable.
- Raw: 541,909 line items / 4,372 identified customers.
- Clean: 392,692 positive transaction lines / 4,338 customers.
- Eligible before 1 September 2011: 3,317 customers.
- Split: 2,653 training / 664 unseen test customers; zero overlap.
- Model search: 12 XGBoost candidates, five shuffled customer folds, 60 fits plus best-model refit.
- Selected candidate CV accuracy: 0.5839 +/- 0.0223; macro F1: 0.5422 +/- 0.0164.
- Test XGBoost accuracy: 0.5813; macro F1: 0.5474.
- Test majority baseline accuracy: 0.4292; macro F1: 0.2002.
- Full-training positive-spender median boundary: GBP 602.38.
- Verification: 11 checks; 9 decoded nonblank PNGs; two executed notebooks.

| Test tier | Precision | Recall | F1 | Customers |
|---|---:|---:|---:|---:|
| No purchase | 0.5759 | 0.7719 | 0.6597 | 285 |
| Regular spender | 0.5046 | 0.2850 | 0.3642 | 193 |
| High spender | 0.6416 | 0.5968 | 0.6184 | 186 |

## Resume claim audit

1. **RFM/PCA/K-Means over 3,000+ customers:** RFM built for 3,317; PCA/K-Means fit on 2,653 training customers and used to assign all 3,317. The stricter claim of fitting clustering on over 3,000 customers is not met because test customers are excluded from fitting.
2. **Behavioral cohorts as modeling features:** reproduced; four one-hot segment indicators, two PCA coordinates and four centroid distances enter XGBoost inside each fold.
3. **Transaction, tenure and cadence features:** reproduced; 19 historical inputs include order counts/values, recent activity, first-observed tenure, inter-order gaps and product breadth.
4. **XGBoost + five-fold RandomizedSearchCV:** reproduced. Historical held-out accuracy >85%: **not reproduced**. No test-based retuning or dataset filtering was used to chase this claim.

## Interpretation and limitations

The target is 91-day gross positive purchase revenue, not lifetime value, profit or net revenue. No-purchase customers are retained. Refunds/cancellations are excluded rather than netted; positive service/postage lines are retained. Full December is unavailable, so the target ends before 1 December. Tenure is left-censored at the dataset start. New customers first appearing after the cutoff are excluded. Generalization is assessed across unseen customers at one shared cutoff, not across multiple future economic periods. Four clusters are a fixed descriptive choice, not a claim of an optimal segmentation. The 12-candidate search is modest and selected CV scores are optimistic tuning estimates.

Boundaries use zero plus the training-only median among positive future spenders. Each CV fit recalibrates the median from its training fold, then labels validation outcomes with that boundary. Refit recalibrates from all training customers; test outcomes cannot affect it. Macro F1 selects the model. Historical project equivalence cannot be established without its original artifacts.

See `execution.log`, `verification.json`, `data_audit.json`, `metrics.json`, `best_cv_folds.csv`, `cv_results.csv`, `best_parameters.json`, `segment_profiles.csv`, `test_predictions.csv`, `customer_predictions.csv`, and `figures/` for computed evidence.

Reproduce from repository root: `python run_project.py` (Python 3.12 recommended).
