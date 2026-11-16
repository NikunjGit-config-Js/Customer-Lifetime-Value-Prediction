# Project instructions

- Work autonomously inside this CLTV workspace only. Use a project-local `.venv`.
- Inspect for original data first; none was present at initial inspection. Download genuine UCI Online Retail from the official repository as the documented replacement.
- Use readable Python scripts for preparation, training and prediction; notebooks for executed EDA, RFM and segmentation. Avoid unnecessary abstractions.
- Run `python run_project.py` after meaningful pipeline changes. Debug failures, execute both notebooks, inspect plots, reload the saved model and verify real outputs. Never claim execution without evidence.
- Never fabricate data, metrics, charts or model files. Save actual benchmark results under `results/`.
- Define future-spending tiers as a finite-horizon CLTV proxy, never proven lifetime revenue. Keep future transactions out of features.
- Use customer-disjoint outer train/test sets. Fit imputation, scaling, PCA and K-Means within each training/CV fold. Learn spending tier boundaries only from the corresponding training customers.
- Use XGBoost with five-fold RandomizedSearchCV and compare with a majority baseline. Report accuracy, macro F1, per-class precision/recall and confusion matrices honestly.
- Investigate all four resume claims: 3,000+ RFM/PCA/K-Means customers; segment features; transaction/tenure/cadence features; XGBoost five-fold tuning and historical >85% accuracy. Never force the accuracy claim.
- Save processed data, predictions, fitted models, hyperparameters, CV results and visualizations. Verify counts, splits and saved-model predictions.
- Keep README concise and interview-ready; document historical-data uncertainty, methodology, real results and limitations.
- Do not publish, push, or contact anyone without authorization. Do not modify other project directories.
