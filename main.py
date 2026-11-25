import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report

from config import FEATURES, OUTER_CV, INNER_CV, TOP_N, RESULTS, make_model, load_train, \
    exclude_features
from baseline import fit_baseline, CLASS_NAMES
from fs1_mutual_information import mutual_information
from fs2_rf_importance import rf_importance
from fs3_rfe import rfe_ranking
from fs4_consensus import consensus_ranking, kendall_w
from fs5_topk_curve import topk_curve, choose_k, plot_topk_curve


def select_features(X, y, cv=INNER_CV, verbose=False):
    """Complete feature selection (steps 1–5) on the data passed in.
    Returns (selected_features, details)."""
    X = exclude_features(X)                     # safety check: only the 21 features
    log = print if verbose else (lambda *a, **k: None)
    mi = mutual_information(X, y, cv);          log("  1. mutual information done")
    rf, _ = rf_importance(X, y, cv);            log("  2. RF importance done")
    rfe, _ = rfe_ranking(X, y, cv);             log("  3. RFE done")
    cons = consensus_ranking(mi, rf, rfe);      log("  4. consensus done")
    order = list(cons.index)
    curve = topk_curve(X, y, order, cv)
    k = choose_k(curve);                        log(f"  5. top-k curve done -> k = {k}")
    details = {"mi": mi, "rf": rf, "rfe": rfe, "consensus": cons, "curve": curve, "k": k,
               "kendall_w": kendall_w(cons)}
    return order[:k], details


if __name__ == "__main__":
    X, y = load_train()
    n_outer = OUTER_CV.get_n_splits()
    print(f"Training set: {X.shape}\n")

    rows, selected_per_fold, ranks_per_fold = [], [], []
    oof = {"true": [], "baseline": [], "subset": []}           # out-of-fold predictions

    # ---------------- Nested CV ----------------
    for fold, (tr, te) in enumerate(OUTER_CV.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        print(f"===== Outer fold {fold}/{n_outer}: train {len(X_tr)} rows, test {len(X_te)} rows")

        # 1. Baseline: all 21 features, untuned
        pred_base = fit_baseline(X_tr, y_tr).predict(X_te[FEATURES])

        # 2. Feature selection on the outer-training part only
        selected, details = select_features(X_tr, y_tr, verbose=True)

        # 3. Reduced model: same untuned model, only the selected features
        pred_sub = make_model().fit(X_tr[selected], y_tr).predict(X_te[selected])

        f1_base = f1_score(y_te, pred_base, average="macro")
        f1_sub = f1_score(y_te, pred_sub, average="macro")
        print(f"  Macro F1  baseline (21): {f1_base:.3f} | selected ({len(selected)}): {f1_sub:.3f}\n")

        rows.append({"outer_fold": fold, "k": len(selected),
                     "f1_baseline": f1_base, "f1_subset": f1_sub, "diff": f1_sub - f1_base,
                     "bal_acc_baseline": balanced_accuracy_score(y_te, pred_base),
                     "bal_acc_subset": balanced_accuracy_score(y_te, pred_sub),
                     "kendall_w": details["kendall_w"], "selected": ", ".join(selected)})
        selected_per_fold.append(pd.Series(1, index=selected, name=f"fold_{fold}"))
        ranks_per_fold.append(details["consensus"]["consensus_rank"].rename(f"fold_{fold}"))
        oof["true"] += list(y_te); oof["baseline"] += list(pred_base); oof["subset"] += list(pred_sub)

    # ---------------- Results of the nested CV ----------------
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "main_nested_cv_folds.csv", index=False)

    sel = pd.concat(selected_per_fold, axis=1).reindex(FEATURES).fillna(0).astype(int)
    ranks = pd.concat(ranks_per_fold, axis=1)
    stability = pd.DataFrame({"selected_in_folds": sel.sum(axis=1),
                              f"top{TOP_N}_in_folds": (ranks <= TOP_N).sum(axis=1),
                              "mean_consensus_rank": ranks.mean(axis=1),
                              "std_consensus_rank": ranks.std(axis=1)}) \
                  .sort_values(["mean_consensus_rank", "selected_in_folds"], ascending=[True, False])
    stability.index.name = "feature"
    stability.to_csv(RESULTS / "main_nested_cv_stability.csv")
    ranks.to_csv(RESULTS / "main_consensus_rank_per_fold.csv")

    print("=" * 72)
    print("NESTED CV RESULTS (outer test folds)")
    print("=" * 72)
    print(res[["outer_fold", "k", "f1_baseline", "f1_subset", "diff", "kendall_w"]]
          .round(3).to_string(index=False))
    for name, col in [("Baseline (21 features)", "f1_baseline"), ("Selected features", "f1_subset")]:
        print(f"{name:24s} Macro F1: {res[col].mean():.3f} ± {res[col].std():.3f}")
    print(f"{'Difference':24s}         : {res['diff'].mean():+.3f} ± {res['diff'].std():.3f}")
    print(f"\nSelection stability over the {n_outer} outer folds:")
    print(stability.round(2).to_string())
    for name in ["baseline", "subset"]:
        print(f"\nClassification report – {name} (all outer test folds combined):")
        print(classification_report(oof["true"], oof[name], target_names=CLASS_NAMES, digits=3))

    # ---------------- Final selection on the whole training set ----------------
    print("=" * 72)
    print("FINAL SELECTION (whole training set) -> features for the final model")
    print("=" * 72)
    final_feats, final = select_features(X, y, verbose=True)
    final["consensus"].to_csv(RESULTS / "final_consensus.csv")
    final["curve"].to_csv(RESULTS / "final_topk_curve.csv", index=False)
    with open(RESULTS / "final_features.json", "w") as f:
        json.dump({"k": final["k"], "features": final_feats,
                   "kendall_w": round(final["kendall_w"], 3)}, f, indent=2)
    print(f"Final: k = {final['k']} -> {final_feats}")
    print(f"Kendall's W (agreement of the 3 methods): {final['kendall_w']:.2f}")

    # ---------------- Plots ----------------
    plot_topk_curve(final["curve"], final["k"], RESULTS / "final_topk_curve.png",
                    title="Top-k consensus curve (final selection, whole training set)")

    s = stability.iloc[::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(s.index, s.selected_in_folds, color="#4C72B0")
    plt.xlabel(f"Selected in how many of the {n_outer} outer folds"); plt.xticks(range(n_outer + 1))
    plt.title("Feature selection stability (nested CV)\nsorted by mean consensus rank")
    plt.tight_layout(); plt.savefig(RESULTS / "main_nested_cv_stability.png", dpi=150); plt.close()

    x = range(len(res))
    plt.figure(figsize=(7, 4.5))
    plt.bar([i - 0.2 for i in x], res.f1_baseline, 0.4, label="Baseline (21 features)", color="#7F7F7F")
    plt.bar([i + 0.2 for i in x], res.f1_subset, 0.4, label="Selected features", color="#55A868")
    for i, k in enumerate(res.k):
        plt.text(i + 0.2, res.f1_subset[i] + 0.005, f"k={k}", ha="center", fontsize=8)
    plt.xticks(list(x), [f"Fold {i}" for i in res.outer_fold])
    plt.ylim(0.7, 1.0); plt.ylabel("Macro F1 on outer test fold")
    plt.title("Nested CV: baseline vs. selected features (both untuned)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(RESULTS / "main_nested_cv_performance.png", dpi=150); plt.close()
