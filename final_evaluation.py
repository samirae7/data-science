"""Final evaluation on the TEST SET (used exactly once, at the very end).

Compared:
  - Baseline:     make_model() (StandardScaler -> RF, untuned), all 21 features
  - Final model:  model + hyperparameters from final_model.json (model_comparison.py),
                  only the final features from final_features.json (main.py)
Both are trained on the WHOLE training set and evaluated once on the test set.

Metrics (3 classes: Normal, Suspect, Pathologic):
  - accuracy, balanced accuracy, macro F1
  - precision, recall, F1 per class
  - ROC AUC and PR AUC (average precision), one-vs-rest, per class and macro
  - train vs. test (overfitting check) and CV estimate vs. test (generalisation check)
  - paired bootstrap (2000 resamples of the test set): 95 % CI of the macro F1 difference
    final - baseline. If the CI contains 0, the difference is not significant.
Plots: confusion matrices, ROC curves, precision-recall curves, metric bar chart.

Called by main.py as the last step. Standalone: python final_evaluation.py
(needs the outputs of main.py and model_comparison.py).
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, confusion_matrix,
                             roc_auc_score, roc_curve, average_precision_score,
                             precision_recall_curve)
from sklearn.preprocessing import label_binarize

from config import FEATURES, RESULTS, make_model, load_train, exclude_features
from models import MODELS
from baseline import CLASS_NAMES

CLASSES = [1, 2, 3]                                          # NSP: Normal, Suspect, Pathologic
COLORS = {"Baseline": "#7F7F7F", "Final model": "#55A868"}


def load_test():
    """Test set from split.py. Only loaded here, at the very end."""
    from split import X_test, y_test
    return exclude_features(X_test), y_test


def build_final_model():
    """Final model from final_model.json (model type, features, tuned hyperparameters)."""
    with open(RESULTS / "final_model.json") as f:
        cfg = json.load(f)
    model = MODELS[cfg["model"]][0]()                             # fresh pipeline from models.py
    model.set_params(**{f"model__{k}": v for k, v in cfg["best_params"].items()})
    return cfg, model


def evaluate(model, X, y):
    """All metrics for one model on one data set."""
    pred = model.predict(X)
    proba = model.predict_proba(X)                                # columns = classes 1, 2, 3
    y_bin = label_binarize(y, classes=CLASSES)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    m = {"accuracy": accuracy_score(y, pred),
         "balanced_accuracy": balanced_accuracy_score(y, pred),
         "f1_macro": f1_score(y, pred, average="macro"),
         "roc_auc_macro": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
         "pr_auc_macro": average_precision_score(y_bin, proba, average="macro")}
    for i, c in enumerate(CLASS_NAMES):
        m[f"precision_{c}"] = prec[i]
        m[f"recall_{c}"] = rec[i]
        m[f"f1_{c}"] = f1[i]
        m[f"roc_auc_{c}"] = roc_auc_score(y_bin[:, i], proba[:, i])
        m[f"pr_auc_{c}"] = average_precision_score(y_bin[:, i], proba[:, i])
    return m, pred, proba


def cv_estimates(final_name):
    """Macro F1 estimated by the nested CV (training set) for both models, if available."""
    est = {}
    p = RESULTS / "main_nested_cv_folds.csv"
    if p.exists():
        est["Baseline"] = pd.read_csv(p)["f1_baseline"].mean()
    p = RESULTS / "model_comparison_summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        row = s[(s.model == final_name) & (s.features == "selected")]
        if len(row):
            est["Final model"] = row["f1_macro_mean"].iloc[0]
    return est


def bootstrap_diff(y_true, pred_a, pred_b, n=2000, seed=42):
    """Paired bootstrap: 95 % CI of macro F1(a) - macro F1(b) on the same resampled test cases."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        diffs.append(f1_score(y_true[i], pred_a[i], average="macro")
                     - f1_score(y_true[i], pred_b[i], average="macro"))
    return np.percentile(diffs, [2.5, 97.5])


def run():
    """Train both models on the whole training set, evaluate once on the test set."""
    X_train, y_train = load_train()
    X_test, y_test = load_test()
    cfg, final_model = build_final_model()
    feats = cfg["features"]
    print(f"Training set: {X_train.shape}, test set: {X_test.shape}")
    print(f"Final model: {cfg['model']} {cfg['best_params']}, {len(feats)} features: {feats}\n")

    models = {"Baseline": (make_model(), FEATURES),
              "Final model": (final_model, feats)}
    rows, test_out = [], {}
    for name, (model, cols) in models.items():
        model.fit(X_train[cols], y_train)                     # whole training set
        for set_name, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
            m, pred, proba = evaluate(model, X[cols], y)
            rows.append({"model": name, "data": set_name, "k": len(cols), **m})
            if set_name == "Test":
                test_out[name] = (pred, proba)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "final_evaluation_metrics.csv", index=False)

    # ---------------- Console output ----------------
    main_cols = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_macro", "pr_auc_macro"]
    print("=" * 80)
    print("FINAL EVALUATION (train = whole training set, test = held-out test set)")
    print("=" * 80)
    print(res[["model", "data", "k"] + main_cols].round(3).to_string(index=False))

    test = res[res.data == "Test"].set_index("model")
    print("\nPer class on the TEST set:")
    per_class = pd.DataFrame({
        (name, metric): [test.loc[name, f"{metric}_{c}"] for c in CLASS_NAMES]
        for name in models for metric in ["precision", "recall", "f1", "roc_auc", "pr_auc"]},
        index=CLASS_NAMES)
    print(per_class.round(3).to_string())
    per_class.round(4).to_csv(RESULTS / "final_evaluation_per_class.csv")

    # Is the difference between the two models larger than chance on 411 test cases?
    lo, hi = bootstrap_diff(y_test, test_out["Final model"][0], test_out["Baseline"][0])
    diff = test.loc["Final model", "f1_macro"] - test.loc["Baseline", "f1_macro"]
    print(f"\nMacro F1 difference (final - baseline): {diff:+.3f}   "
          f"95 % bootstrap CI [{lo:+.3f}, {hi:+.3f}]")
    with open(RESULTS / "final_bootstrap_diff.json", "w") as f:
        json.dump({"f1_macro_final_minus_baseline": round(diff, 4),
                   "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
                   "n_bootstrap": 2000}, f, indent=2)

    est = cv_estimates(cfg["model"])
    if est:
        print("\nGeneralisation check: macro F1 estimated by nested CV vs. measured on the test set")
        for name, v in est.items():
            print(f"  {name:12s} CV estimate: {v:.3f}   test: {test.loc[name, 'f1_macro']:.3f}   "
                  f"difference: {test.loc[name, 'f1_macro'] - v:+.3f}")

    # ---------------- Plots ----------------
    y_bin = label_binarize(y_test, classes=CLASSES)

    # 1. Confusion matrices (test set): counts + percentage of the true class (= recall)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, models):
        cm = confusion_matrix(y_test, test_out[name][0], labels=CLASSES)
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([[f"{cm[i, j]}\n({pct[i, j]:.0f}%)" for j in range(3)] for i in range(3)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f"{name} ({len(models[name][1])} features)\n"
                     f"macro F1 = {test.loc[name, 'f1_macro']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices on the test set (% of the actual class)")
    plt.tight_layout(); plt.savefig(RESULTS / "final_confusion_matrices.png", dpi=150); plt.close()

    # 2. ROC curves and 3. precision-recall curves: one subplot per class (one-vs-rest)
    for kind in ["roc", "pr"]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True)
        for c, (ax, cls) in enumerate(zip(axes, CLASS_NAMES)):
            for name in models:
                p = test_out[name][1][:, c]
                if kind == "roc":
                    x, yv, _ = roc_curve(y_bin[:, c], p)
                    score = roc_auc_score(y_bin[:, c], p)
                else:
                    yv, x, _ = precision_recall_curve(y_bin[:, c], p)
                    score = average_precision_score(y_bin[:, c], p)
                ax.plot(x, yv, lw=1.8, color=COLORS[name], label=f"{name} (AUC = {score:.3f})")
            if kind == "roc":
                ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="Chance (AUC = 0.5)")
                ax.set_xlabel("False positive rate (1 - specificity)")
            else:
                prev = y_bin[:, c].mean()                  # chance level = share of the class
                ax.axhline(prev, color="k", ls=":", lw=0.8, label=f"Chance (= {prev:.2f})")
                ax.set_xlabel("Recall (sensitivity)")
            ax.set_title(f"{cls} vs. rest")
            ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=9)
            ax.set_aspect("equal")
        axes[0].set_ylabel("True positive rate (sensitivity)" if kind == "roc" else "Precision")
        title = "ROC curves" if kind == "roc" else "Precision-recall curves"
        fig.suptitle(f"{title} on the test set (one-vs-rest)")
        plt.tight_layout()
        plt.savefig(RESULTS / f"final_{kind}_curves.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 4. Bar chart: main metrics, train vs. test, both models
    labels = ["Accuracy", "Balanced acc.", "Macro F1", "ROC AUC", "PR AUC"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, set_name, hatch) in enumerate([("Baseline", "Train", "//"), ("Baseline", "Test", ""),
                                                 ("Final model", "Train", "//"), ("Final model", "Test", "")]):
        v = res[(res.model == name) & (res.data == set_name)][main_cols].iloc[0].values
        ax.bar(x + (i - 1.5) * 0.2, v, 0.2, color=COLORS[name], hatch=hatch, edgecolor="white",
               alpha=0.55 if set_name == "Train" else 1.0, label=f"{name} – {set_name}")
    ax.set_xticks(x, labels); ax.set_ylim(0.6, 1.02); ax.set_ylabel("Score")
    ax.set_title("Train vs. test (both models trained on the whole training set)")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    plt.tight_layout(); plt.savefig(RESULTS / "final_metrics.png", dpi=150); plt.close()

    print(f"\nSaved: final_evaluation_metrics.csv, final_evaluation_per_class.csv, "
          f"final_bootstrap_diff.json, final_confusion_matrices.png, final_roc_curves.png, final_pr_curves.png, "
          f"final_metrics.png (in '{RESULTS}')")


if __name__ == "__main__":
    run()