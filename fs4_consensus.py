"""Step 4: consensus ranking across the three method families.
Rank per family: filter = MI, embedded = permutation importance ONLY, wrapper = RFE.
Gini importance is reported for comparison but NOT used in the consensus, because RFE
already relies on the Gini importance internally (this would double-count it)."""
import pandas as pd
import matplotlib.pyplot as plt
from config import TOP_N, RESULTS

METHOD_RANKS = ["rank_mi", "rank_perm", "rank_rfe"]


def consensus_ranking(mi, rf, rfe, top_n=TOP_N):
    """Mean rank over the three families. Returns a DataFrame sorted by consensus_rank."""
    c = pd.DataFrame({
        "rank_mi": mi["rank_mi"],
        "rank_gini": rf["rank_gini"],
        "rank_perm": rf["rank_perm"],
        "rank_rfe": rfe["rank_rfe"],
    })
    if c.isna().any().any():   # a feature missing in one method would silently get fewer votes
        raise ValueError(f"Feature sets differ between methods:\n{c[c.isna().any(axis=1)]}")

    c["consensus_score"] = c[METHOD_RANKS].mean(axis=1)
    c["top_in_methods"] = (c[METHOD_RANKS] <= top_n).sum(axis=1)
    # ties: more top-n placements first; equal scores get the same rank (1, 2, 2, 4, ...)
    c = c.sort_values(["consensus_score", "top_in_methods"], ascending=[True, False])
    c["consensus_rank"] = c["consensus_score"].rank(method="min").astype(int)
    c.index.name = "feature"
    return c


def kendall_w(c, cols=METHOD_RANKS):
    """Kendall's W: agreement of the methods' rankings (0 = none, 1 = identical)."""
    R = c[cols].to_numpy(dtype=float)
    n, m = R.shape                                  # n features, m methods
    S = ((R.sum(axis=1) - m * (n + 1) / 2) ** 2).sum()
    return 12 * S / (m ** 2 * (n ** 3 - n))


if __name__ == "__main__":
    mi = pd.read_csv(RESULTS / "fs1_mutual_information.csv", index_col="feature")   # step 1
    rf = pd.read_csv(RESULTS / "fs2_rf_importance.csv", index_col="feature")        # step 2
    rfe = pd.read_csv(RESULTS / "fs3_rfe.csv", index_col="feature")                 # step 3
    c = consensus_ranking(mi, rf, rfe)
    c.to_csv(RESULTS / "fs4_consensus.csv")                                          # -> step 5
    print(f"Consensus ranking (top_in_methods = in how many of the 3 families in the top {TOP_N}):")
    print(c.round(2).to_string())
    print(f"\nIn the top {TOP_N} of all 3 families: {list(c.index[c.top_in_methods == 3])}")
    print(f"Kendall's W (agreement of the 3 methods): {kendall_w(c):.2f}")
    print("Pairwise Spearman correlation of the rankings:")
    print(c[METHOD_RANKS].corr(method="spearman").round(2).to_string())

    show = c[["rank_mi", "rank_perm", "rank_rfe", "consensus_rank", "rank_gini"]]
    plt.figure(figsize=(7, 9))
    plt.imshow(show, cmap="viridis_r", aspect="auto")
    plt.xticks(range(show.shape[1]), ["MI\n(filter)", "Permutation\n(embedded)",
                                      "RFE\n(wrapper)", "Consensus", "Gini\n(comparison only)"])
    plt.yticks(range(len(show)), show.index)
    for i in range(show.shape[0]):
        for j in range(show.shape[1]):
            v = show.iloc[i, j]
            plt.text(j, i, int(v), ha="center", va="center",
                     color="white" if v > 10 else "black", fontsize=8)
    plt.colorbar(label="Rank (1 = most important)")
    plt.title("Feature rank per method")
    plt.tight_layout()
    plt.savefig(RESULTS / "fs4_consensus.png", dpi=150)
    plt.close()