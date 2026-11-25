import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from config import OUTER_CV, RESULTS, FEATURES, make_model, load_train

CLASS_NAMES = ["Normal", "Suspect", "Pathologic"]      # NSP = 1, 2, 3


def fit_baseline(X, y):
    """Untuned baseline model trained on all 21 features of the data passed in."""
    return make_model().fit(X[FEATURES], y)


def evaluate_baseline(X, y, cv=OUTER_CV):
    """Per-fold scores + out-of-fold predictions of the baseline model."""
    scores = cross_validate(make_model(), X[FEATURES], y, cv=cv,
                            scoring=["f1_macro", "balanced_accuracy", "accuracy"])
    folds = pd.DataFrame({
        "fold": range(1, cv.get_n_splits() + 1),
        "f1_macro": scores["test_f1_macro"],
        "balanced_accuracy": scores["test_balanced_accuracy"],
        "accuracy": scores["test_accuracy"],
    })
    # every sample is predicted exactly once, by the model that did NOT train on it
    y_pred = cross_val_predict(make_model(), X[FEATURES], y, cv=cv)
    return folds, y_pred


if __name__ == "__main__":
    X, y = load_train()
    print(f"Training set: {len(y)} samples, class counts: {y.value_counts().sort_index().to_dict()}")

    folds, y_pred = evaluate_baseline(X, y)
    folds.to_csv(RESULTS / "baseline_outer_folds.csv", index=False)
    print("\nBaseline (all 21 features, untuned), outer 5-fold CV:")
    print(folds.round(3).to_string(index=False))
    summary = {m: {"mean": round(folds[m].mean(), 3), "std": round(folds[m].std(), 3)}
               for m in ["f1_macro", "balanced_accuracy", "accuracy"]}
    for m, v in summary.items():
        print(f"{m:>18}: {v['mean']:.3f} ± {v['std']:.3f}")
    with open(RESULTS / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPer class (out-of-fold predictions):")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES, digits=3))

    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues", values_format="d")
    plt.title("Baseline confusion matrix (outer CV, training set)")
    plt.tight_layout()
    plt.savefig(RESULTS / "baseline_confusion_matrix.png", dpi=150)
    plt.close()