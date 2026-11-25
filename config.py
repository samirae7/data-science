"""Shared settings for the feature selection (steps fs0–fs5 and fs_main).

The feature selection functions only ever use the data they are given.
-> Inside the nested CV (fs_main.py), they are called with the OUTER-TRAINING part only.
-> Run standalone (python fsX_....py), they use the whole training set from split.py.
"""
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

SEED = 42

# ---------------------------------------------------------------------------
# Features: the 21 CTG measurement features. The leakage columns (CLASS, A–SUSP,
# b, e, LBE) were already removed during data cleaning.
# exclude_features() is a safety check: it fails if a feature is missing and drops
# anything else (e.g. the target NSP passed in by mistake).
# ---------------------------------------------------------------------------
FEATURES = ["LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV",
            "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean",
            "Median", "Variance", "Tendency"]


def exclude_features(X):
    """Keep only the 21 measurement features (in the fixed order of FEATURES)."""
    missing = [f for f in FEATURES if f not in X.columns]
    if missing:
        raise ValueError(f"Missing measurement features: {missing}")
    return X[FEATURES]


SCORING = "f1_macro"          # macro F1 because of class imbalance
TOP_N = 5                     # "top 5" used for the consensus overview

# Outer CV of the nested CV (fs_main.py): estimates how good / how stable the selection is.
OUTER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Inner CV used INSIDE the feature selection (MI, permutation importance, RFECV, top-k curve).
# Inside the nested CV this splits the outer-training part only.
# Different seed than OUTER_CV, so the inner and outer splits are not aligned by chance.
INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)

RESULTS = Path("results_feature_selection")
RESULTS.mkdir(exist_ok=True)


def make_model():
    """Model used for RF importance, RFE and the top-k curve (same as the baseline).
    The scaler has no effect on a random forest; it is kept only for consistency
    with the baseline pipeline.
    If the class-imbalance handling changes (e.g. SMOTE), change it HERE only,
    e.g. with an imblearn Pipeline so resampling happens only on training data."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=SEED, n_jobs=-1)),
    ])


def load_train():
    """Training data for standalone runs (test set is never loaded here)."""
    from split import X_train, y_train
    return exclude_features(X_train), y_train