"""Step 2 – b. Embedded: RF importance in the inner CV.
   (i)  Gini importance (built into the random forest) -> additional info only
   (ii) Permutation importance on the validation fold -> enters the consensus (step 4)"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from config import INNER_CV, SEED, SCORING, RESULTS, make_model, load_train


def rf_importance(X, y, cv=INNER_CV, seed=SEED, n_repeats=10, verbose=False):
    """Gini + permutation importance of the baseline RF, averaged over cv folds."""
    fold_f1, gini, perm = [], [], []
    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        model = make_model().fit(X.iloc[tr], y.iloc[tr])
        f1 = f1_score(y.iloc[va], model.predict(X.iloc[va]), average="macro")
        fold_f1.append(f1)
        gini.append(model.named_steps["rf"].feature_importances_)
        r = permutation_importance(model, X.iloc[va], y.iloc[va], scoring=SCORING,
                                   n_repeats=n_repeats, random_state=seed)
        perm.append(r.importances_mean)
        if verbose:
            print(f"Fold {fold} macro F1: {f1:.3f}")

    gini = pd.DataFrame(gini, columns=X.columns)
    perm = pd.DataFrame(perm, columns=X.columns)
    ranking = pd.DataFrame({
        "gini_mean": gini.mean(), "gini_std": gini.std(),
        "perm_mean": perm.mean(), "perm_std": perm.std(),
        "perm_folds_positive": (perm > 0).sum(),   # stability: folds with importance > 0
    })
    ranking["rank_gini"] = ranking["gini_mean"].rank(ascending=False, method="min").astype(int)
    ranking["rank_perm"] = ranking["perm_mean"].rank(ascending=False, method="min").astype(int)
    ranking = ranking.sort_values("rank_perm")
    ranking.index.name = "feature"
    return ranking, np.array(fold_f1)


def run():
    """Step 2: Gini + permutation importance + plot (whole training set)."""
    X, y = load_train()
    ranking, fold_f1 = rf_importance(X, y, verbose=True)
    print(f"Average CV macro F1 (baseline): {fold_f1.mean():.3f} ± {fold_f1.std():.3f}\n")
    ranking.to_csv(RESULTS / "fs2_rf_importance.csv")              # -> step 4
    print("RF importance (Gini and permutation, mean over 5 folds):")
    print(ranking.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, col, title, color in [(axes[0], "gini", "Gini importance (comparison only)", "#DD8452"),
                                  (axes[1], "perm", "Permutation importance (drop in macro F1)", "#55A868")]:
        r = ranking.sort_values(f"{col}_mean")
        ax.barh(r.index, r[f"{col}_mean"], xerr=r[f"{col}_std"], color=color)
        ax.set_title(title)
    fig.suptitle("b. Embedded: random forest importance (5-fold CV, training set)")
    plt.tight_layout()
    plt.savefig(RESULTS / "fs2_rf_importance.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run()