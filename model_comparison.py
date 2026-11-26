import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from config import FEATURES, OUTER_CV, INNER_CV, SCORING, RESULTS, load_train
from models import MODELS
from baseline import CLASS_NAMES

warnings.filterwarnings("ignore", category=UserWarning)   # e.g. convergence messages
CLASSES = [1, 2, 3]                                        # NSP: Normal, Suspect, Pathologic
SETS = ["all 21", "selected"]


def load_fold_features():
    """Selected features per outer fold, as saved by main.py."""
    path = RESULTS / "main_nested_cv_folds.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -> run main.py first.")
    folds = pd.read_csv(path)
    return {int(r.outer_fold): r.selected.split(", ") for r in folds.itertuples()}


def tune(name, X, y):
    """Inner loop: grid search with INNER_CV. Returns the fitted search."""
    build, grid = MODELS[name]
    return GridSearchCV(build(), grid, cv=INNER_CV, scoring=SCORING, n_jobs=-1).fit(X, y)


def plot_roc(y_true, oof_proba, set_name, path):
    """One subplot per class (one-vs-rest), one curve per model, AUC in the legend."""
    y_bin = label_binarize(y_true, classes=CLASSES)          # n x 3 matrix of 0/1
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True)
    for c, (ax, cls_name) in enumerate(zip(axes, CLASS_NAMES)):
        for name in MODELS:
            p = oof_proba[(name, set_name)][:, c]
            fpr, tpr, _ = roc_curve(y_bin[:, c], p)
            ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC = {roc_auc_score(y_bin[:, c], p):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance (AUC = 0.5)")
        ax.set_title(f"{cls_name} vs. rest")
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_aspect("equal")
    axes[0].set_ylabel("True positive rate (sensitivity)")
    fig.suptitle(f"ROC curves, one-vs-rest ({set_name} features, out-of-fold predictions, nested CV)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    X, y = load_train()
    fold_features = load_fold_features()
    n_outer = OUTER_CV.get_n_splits()
    rows = []
    # out-of-fold probabilities: one row per training sample, filled fold by fold
    oof_proba = {(m, s): np.zeros((len(y), len(CLASSES))) for m in MODELS for s in SETS}

    # ---------------- Nested CV ----------------
    for fold, (tr, te) in enumerate(OUTER_CV.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        feature_sets = {"all 21": FEATURES, "selected": fold_features[fold]}
        print(f"===== Outer fold {fold}/{n_outer} (selected: k = {len(fold_features[fold])})")

        for name in MODELS:
            for set_name, feats in feature_sets.items():
                search = tune(name, X_tr[feats], y_tr)                        # inner loop
                model = search.best_estimator_
                pred = model.predict(X_te[feats])                             # outer test
                proba = model.predict_proba(X_te[feats])                      # columns = 1, 2, 3
                oof_proba[(name, set_name)][te] = proba

                f1 = f1_score(y_te, pred, average="macro")
                auc = roc_auc_score(y_te, proba, multi_class="ovr", average="macro")
                aucs = roc_auc_score(label_binarize(y_te, classes=CLASSES), proba, average=None)
                rows.append({"outer_fold": fold, "model": name, "features": set_name,
                             "k": len(feats), "f1_macro": f1, "auc_macro": auc,
                             **{f"auc_{c.lower()}": a for c, a in zip(CLASS_NAMES, aucs)},
                             "balanced_accuracy": balanced_accuracy_score(y_te, pred),
                             "inner_best_f1": search.best_score_,
                             "best_params": json.dumps({p.replace("model__", ""): v for p, v
                                                        in search.best_params_.items()})})
                print(f"  {name:16s} {set_name:8s}  macro F1 = {f1:.3f}  macro AUC = {auc:.3f}")

    # ---------------- Results ----------------
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "model_comparison_folds.csv", index=False)

    metrics = ["f1_macro", "auc_macro", "auc_normal", "auc_suspect", "auc_pathologic"]
    summary = res.groupby(["model", "features"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS / "model_comparison_summary.csv", index=False)

    wide = summary.pivot(index="model", columns="features")
    order = wide[("f1_macro_mean", "selected")].sort_values(ascending=False).index

    print("\n" + "=" * 80)
    print("MODEL COMPARISON (outer test folds, mean ± SD over 5 folds)")
    print("=" * 80)
    print(f"{'':16s} {'macro F1':^27s} {'macro AUC (OvR)':^27s}")
    print(f"{'model':16s} {'all 21':>13s} {'selected':>13s} {'all 21':>13s} {'selected':>13s}")
    for m in order:
        cells = [f"{wide.loc[m, (f'{met}_mean', s)]:.3f}±{wide.loc[m, (f'{met}_std', s)]:.3f}"
                 for met in ["f1_macro", "auc_macro"] for s in SETS]
        print(f"{m:16s} " + " ".join(f"{c:>13s}" for c in cells))
    print("\nAUC per class (selected features):")
    print(wide.loc[order, [(f"auc_{c.lower()}_mean", "selected") for c in CLASS_NAMES]]
          .set_axis(CLASS_NAMES, axis=1).round(3).to_string())
    print("\nChosen hyperparameters per outer fold (selected features):")
    print(res[res.features == "selected"].pivot(index="outer_fold", columns="model",
                                               values="best_params").to_string())

    # ---------------- Final model: best model, tuned on the whole training set ----------------
    best_model = order[0]
    with open(RESULTS / "final_features.json") as f:
        final_feats = json.load(f)["features"]
    final_search = tune(best_model, X[final_feats], y)
    final = {"model": best_model, "features": final_feats,
             "best_params": {p.replace("model__", ""): v
                             for p, v in final_search.best_params_.items()},
             "inner_cv_f1": round(final_search.best_score_, 3)}
    with open(RESULTS / "final_model.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nBest model (macro F1, selected features): {best_model}")
    print(f"Final hyperparameters (whole training set, final features): {final['best_params']}")
    print(f"-> saved to {RESULTS / 'final_model.json'} (for the final evaluation on the test set)")

    # ---------------- Plots ----------------
    # 1. ROC curves (one figure per feature set)
    plot_roc(y, oof_proba, "selected", RESULTS / "model_comparison_roc_selected.png")
    plot_roc(y, oof_proba, "all 21", RESULTS / "model_comparison_roc_all21.png")

    # 2. Macro F1 and macro AUC side by side
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, met, label, ylim in [(axes[0], "f1_macro", "Macro F1", (0.6, 1.0)),
                                 (axes[1], "auc_macro", "Macro AUC (one-vs-rest)", (0.85, 1.0))]:
        for offset, s, lab, color in [(-0.2, "all 21", "All 21 features", "#7F7F7F"),
                                      (0.2, "selected", "Selected features", "#55A868")]:
            ax.bar(x + offset, wide.loc[order, (f"{met}_mean", s)], 0.4,
                   yerr=wide.loc[order, (f"{met}_std", s)], capsize=3, label=lab, color=color)
        ax.set_xticks(x, order)
        ax.set_ylim(*ylim); ax.set_ylabel(f"{label} on outer test folds (mean ± SD)")
        ax.set_title(label)
        ax.legend(loc="lower left")
    fig.suptitle("Model comparison (nested CV, tuned in the inner loop)")
    plt.tight_layout()
    plt.savefig(RESULTS / "model_comparison.png", dpi=150)
    plt.close()