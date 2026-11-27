"""Step 5: top-k curve and choice of k (one-standard-error rule).
For k = 1..21, the model is evaluated with only the top-k features of the consensus
order (step 4). k = smallest number of features whose macro F1 is within one standard
error of the best k (Breiman et al. 1984; Hastie et al., Elements of Statistical Learning)."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from config import SCORING, INNER_CV, RESULTS, make_model, load_train


def topk_curve(X, y, order, cv=INNER_CV):
    """Inner-CV macro F1 for the top-k features of the consensus order, k = 1..n."""
    rows = []
    for k in range(1, len(order) + 1):
        s = cross_val_score(make_model(), X[order[:k]], y, cv=cv, scoring=SCORING)
        rows.append({"k": k, "added_feature": order[k - 1], "f1_mean": s.mean(),
                     "f1_std": s.std(), "f1_se": s.std() / np.sqrt(len(s))})
    return pd.DataFrame(rows)


def choose_k(curve):
    """One-standard-error rule: smallest k whose mean F1 is within 1 SE of the best k."""
    best = curve.loc[curve["f1_mean"].idxmax()]
    return int(curve.loc[curve["f1_mean"] >= best["f1_mean"] - best["f1_se"], "k"].min())


def plot_topk_curve(curve, k, path, title="Top-k consensus curve (training set)"):
    best = curve.loc[curve.f1_mean.idxmax()]
    plt.figure(figsize=(9, 5))
    plt.errorbar(curve.k, curve.f1_mean, yerr=curve.f1_std, marker="o", capsize=3,
                 label="Top-k (consensus), error bars = SD")
    plt.axhline(best.f1_mean - best.f1_se, color="red", ls=":", label="Best - 1 SE")
    plt.axvline(k, color="green", ls=":", label=f"selected: k = {k}")
    plt.xticks(curve.k, [f"{i}\n{f}" for i, f in zip(curve.k, curve.added_feature)], fontsize=7)
    plt.xlabel("Number of features (k) and the feature added at each step")
    plt.ylabel("Macro F1 (inner 5-fold CV)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run():
    """Step 5: top-k curve + choice of k. Needs the CSV of step 4."""
    X, y = load_train()
    cons = pd.read_csv(RESULTS / "fs4_consensus.csv", index_col="feature")   # step 4
    order = list(cons.index)
    curve = topk_curve(X, y, order)
    k = choose_k(curve)
    curve.to_csv(RESULTS / "fs5_topk_curve.csv", index=False)
    with open(RESULTS / "fs5_selected_features.json", "w") as f:
        json.dump({"k": k, "features": order[:k]}, f, indent=2)

    print(curve.drop(columns="f1_se").round(3).to_string(index=False))
    print(f"\nSelected (one-SE rule): k = {k} -> {order[:k]}")
    plot_topk_curve(curve, k, RESULTS / "fs5_topk_curve.png")


if __name__ == "__main__":
    run()