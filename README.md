# CTG – Feature Selection for Fetal State Classification (NSP)

<!-- TODO: course name, semester, team members -->
Course: Foundation of Data Science  
Team: G8 (Mahela Büchi, Alina Eichenberger, Samira Eugster, Lara Staub)

## Research question

Which CTG measurement features (biomarkers) are the most important for predicting the
expert fetal state classification NSP (1 = Normal, 2 = Suspect, 3 = Pathologic)?

Data: UCI Cardiotocography data set (2126 CTG recordings, 21 measurement features).

## Requirements

- Python 3.10 or newer (tested with Python 3.13)
- Packages: see `requirements.txt`

```bash
pip install -r requirements.txt
```

## How to run

All paths are relative, so `main.py` **must be started from the project folder**:

```bash
cd data-science
python main.py
```

`main.py` runs the whole project in the correct order. No other script has to be started.
Runtime: about 25–45 minutes (depends on the computer).
All results (CSV, JSON, PNG) are written to `results_feature_selection/`.

The input file `CTG.csv` must be in the project folder.

## Pipeline (what `main.py` does)

| Step | Script | What happens |
|---|---|---|
| 0 | `data_cleaning_v1.py`, `split.py` | Cleaning (`CTG.csv` → `CTG_cleaned_V1.csv`), stratified 80/20 train/test split (`train.csv`, `test.csv`) |
| 1 | `fs0_correlation.py` – `fs4_consensus.py` | Feature selection steps on the whole training set (plots for the report) |
| 2 | `baseline.py` | Baseline: random forest with all 21 features, untuned, outer 5-fold CV |
| 3 | `main.py` → `run_nested_cv()` | Nested CV: baseline vs. selected features, stability of the selection, final feature set |
| 4 | `model_comparison.py`, `models.py` | LogReg, kNN, SVM, random forest tuned in a nested CV (macro F1, ROC/AUC) → final model |

### Feature selection (steps fs0–fs5)

| Step | Method | Family |
|---|---|---|
| fs0 | Spearman correlation (context only, nothing is removed) | – |
| fs1 | Mutual information | Filter |
| fs2 | Permutation importance of the random forest (Gini for comparison only) | Embedded |
| fs3 | Recursive feature elimination (RFE), RFECV reported | Wrapper |
| fs4 | Consensus ranking (mean rank of fs1–fs3), Kendall's W | – |
| fs5 | Top-k curve, k chosen with the one-standard-error rule | – |

### Excluded columns (before any method runs)

`CLASS`, `A`–`E`, `AD`, `DE`, `LD`, `FS`, `SUSP` (expert labels assigned together with NSP
→ data leakage), `b`, `e` (segment start/end time), `LBE` (duplicate of `LB`).

## Validation design

- **Train/test split:** 80/20, stratified by NSP, `random_state = 42`.
  The test set (`test.csv`) is **not used** by `main.py`; it is reserved for the final evaluation.
- **Outer CV:** 5-fold stratified (`OUTER_CV`), used for every evaluation.
- **Inner CV:** 5-fold stratified (`INNER_CV`), only inside the training part of each outer fold
  (feature selection and hyperparameter tuning) → no information from the outer test fold leaks
  into the selection or the tuning.
- **Class imbalance:** `class_weight="balanced"` (random forest, LogReg, SVM); SMOTE for kNN,
  applied only to the training data of each fold.
- **Metric:** macro F1 (all three classes weighted equally); additionally balanced accuracy and
  macro ROC AUC (one-vs-rest).

## Project structure

```
data-science/
├── main.py                    # entry point: runs everything
├── config.py                  # shared settings: features, CV splits, model, paths
├── data_cleaning_v1.py        # cleaning: CTG.csv -> CTG_cleaned_V1.csv
├── split.py                   # stratified 80/20 split -> train.csv, test.csv
├── baseline.py                # baseline model (all 21 features)
├── fs0_correlation.py         # step 0: correlation analysis
├── fs1_mutual_information.py  # step 1: filter
├── fs2_rf_importance.py       # step 2: embedded
├── fs3_rfe.py                 # step 3: wrapper
├── fs4_consensus.py           # step 4: consensus ranking
├── fs5_topk_curve.py          # step 5: top-k curve + choice of k
├── models.py                  # candidate models + hyperparameter grids
├── model_comparison.py        # nested CV model comparison + final model
├── requirements.txt
├── CTG.csv                    # raw data (input)
└── results_feature_selection/ # all outputs (created automatically)
```

## Main outputs (`results_feature_selection/`)

| File | Content |
|---|---|
| `baseline_summary.json`, `baseline_confusion_matrix.png` | Baseline performance |
| `fs0_correlation.png` … `fs4_consensus.png` | Plots of the feature selection steps |
| `main_nested_cv_folds.csv`, `main_nested_cv_performance.png` | Baseline vs. selected features per outer fold |
| `main_nested_cv_stability.csv`, `main_nested_cv_stability.png` | How often each feature was selected |
| `final_features.json`, `final_topk_curve.png` | Final feature set (whole training set) |
| `model_comparison_summary.csv`, `model_comparison.png`, `model_comparison_roc_*.png` | Model comparison (macro F1, ROC/AUC) |
| `final_model.json` | Best model + hyperparameters for the final test |

## Reproducibility

All random steps use `SEED = 42` (inner CV: 43), so repeated runs give the same results.
Each script can also be run on its own for testing (e.g. `python fs1_mutual_information.py`),
but for the full results only `main.py` is needed.