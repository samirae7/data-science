"""Step 3 – c. Wrapper: recursive feature elimination with the baseline random forest.
   (i)  RFE down to 1 feature -> complete elimination order (rank 1 = removed last)
        -> this ranking goes into the consensus (step 4)
   (ii) RFECV -> optimal number of features via inner cross-validation (reported)"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, RFECV
from config import SCORING, INNER_CV, RESULTS, make_model, load_train

GETTER = "named_steps.rf.feature_importances_"   # importance from the RF inside the pipeline


def rfe_ranking(X, y, cv=INNER_CV, with_rfecv=False):
    rfe = RFE(make_model(), n_features_to_select=1, step=1, importance_getter=GETTER).fit(X, y)
    ranking = pd.DataFrame({"rank_rfe": rfe.ranking_}, index=X.columns)
    ranking.index.name = "feature"
    rfecv = None
    if with_rfecv:
        rfecv = RFECV(make_model(), step=1, cv=cv, scoring=SCORING, min_features_to_select=1,
                      importance_getter=GETTER, n_jobs=-1).fit(X, y)
        ranking["rfecv_selected"] = rfecv.support_
    return ranking.sort_values("rank_rfe"), rfecv

def one_se_k(mean, std, n_folds):
    """Smallest k whose mean score is within 1 standard error of the best k."""
    best = int(np.argmax(mean))
    se = std[best] / np.sqrt(n_folds)
    return int(np.argmax(mean >= mean[best] - se)) + 1       # +1: index 0 = 1 feature


def run():
    """Step 3: RFE ranking + RFECV curve + plot (whole training set)."""
    X, y = load_train()
    ranking, rfecv = rfe_ranking(X, y, with_rfecv=True)
    ranking.to_csv(RESULTS / "fs3_rfe.csv")                         # -> step 4
    print("RFE ranking (1 = removed last = most important):")
    print(ranking.to_string())

    m = rfecv.cv_results_["mean_test_score"]
    s = rfecv.cv_results_["std_test_score"]
    k_1se = one_se_k(m, s, INNER_CV.get_n_splits())
    print(f"\nRFECV maximum: {rfecv.n_features_} features (macro F1 = {m.max():.3f})")
    print(f"RFECV one-SE rule: {k_1se} features")
    print(f"RFECV selected (maximum): {list(ranking.index[ranking.rfecv_selected])}")

    curve = pd.DataFrame({"n_features": range(1, len(X.columns) + 1),
                          "mean_test_score": m, "std_test_score": s})
    curve.to_csv(RESULTS / "fs3_rfecv_curve.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.errorbar(curve.n_features, curve.mean_test_score, yerr=curve.std_test_score,
                 marker="o", capsize=3)
    plt.axvline(rfecv.n_features_, color="green", ls=":", label=f"maximum: {rfecv.n_features_}")
    plt.axvline(k_1se, color="red", ls="--", label=f"one-SE rule: {k_1se}")
    plt.xlabel("Number of features"); plt.ylabel("Macro F1 (inner 5-fold CV, error bars = SD)")
    plt.xticks(curve.n_features); plt.title("c. Wrapper: RFECV (baseline random forest)")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "fs3_rfecv.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run()