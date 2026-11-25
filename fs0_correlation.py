"""Step 0 (preliminary): correlation analysis -> which features are redundant?"""
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS, load_train
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def correlation_analysis(X, threshold=0.8):
    """Spearman correlation matrix + list of strongly correlated pairs."""
    corr = X.corr(method="spearman")
    feats = list(X.columns)
    pairs = pd.DataFrame(
        [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:]
         if abs(corr.loc[a, b]) > threshold],
        columns=["feature_1", "feature_2", "rho"])
    pairs = pairs.reindex(pairs["rho"].abs().sort_values(ascending=False).index)
    return corr, pairs


if __name__ == "__main__":
    X, y = load_train()
    corr, pairs = correlation_analysis(X)
    corr.round(3).to_csv(RESULTS / "fs0_correlation.csv")
    pairs.to_csv(RESULTS / "fs0_correlated_pairs.csv", index=False)
    print("Strongly correlated pairs (|rho| > 0.8):")
    print(pairs.round(2).to_string(index=False))
    print("\nNote: no feature is removed based on this analysis (interpretation only).")

    # Hierarchical clustering with distance 1 - |rho|: strongly positive AND strongly
    # negative correlated features (both redundant) end up next to each other.
    dist = 1 - corr.abs()                                   # |rho| hoch -> Abstand klein
    link = linkage(squareform(dist.values, checks=False), method="average")
    g = sns.clustermap(corr, row_linkage=link, col_linkage=link,
                       cmap="coolwarm", vmin=-1, vmax=1,
                       annot=True, fmt=".1f", annot_kws={"size": 7},
                       figsize=(12, 12),
                       cbar_kws={"label": "Spearman rho"})
    g.fig.suptitle("Correlation of CTG features (training set)", y=1.02)
    g.savefig(RESULTS / "fs0_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)