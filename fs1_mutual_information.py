"""Step 1 – a. Filter: mutual information between each feature and NSP.
Computed on the training part of each inner CV fold -> mean + stability."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from config import INNER_CV, SEED, RESULTS, load_train


def mutual_information(X, y, cv=INNER_CV, seed=SEED):
    """MI of each feature with NSP, averaged over the training parts of cv."""
    mi = [mutual_info_classif(X.iloc[tr], y.iloc[tr], random_state=seed)
          for tr, _ in cv.split(X, y)]
    mi = pd.DataFrame(mi, columns=X.columns)
    ranking = (pd.DataFrame({"mi_mean": mi.mean(), "mi_std": mi.std()})
               .sort_values("mi_mean", ascending=False))
    ranking["rank_mi"] = range(1, len(ranking) + 1)
    ranking.index.name = "feature"
    return ranking


if __name__ == "__main__":
    X, y = load_train()
    ranking = mutual_information(X, y)
    ranking.to_csv(RESULTS / "fs1_mutual_information.csv")        # -> step 4
    print("Mutual information with NSP (mean over 5 folds):")
    print(ranking.round(4).to_string())

    r = ranking.iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(r.index, r.mi_mean, xerr=r.mi_std, color="#4C72B0")
    plt.xlabel("Mutual information with NSP (nats)")
    plt.title("a. Filter: mutual information (training set)")
    plt.tight_layout()
    plt.savefig(RESULTS / "fs1_mutual_information.png", dpi=150)
    plt.close()