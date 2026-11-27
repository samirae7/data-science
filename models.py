from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from config import SEED


def pipe(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


# name -> (function that builds a fresh pipeline, hyperparameter grid)
# "model__C" = parameter C of the pipeline step "model"
MODELS = {
    "LogReg": (
        lambda: pipe(LogisticRegression(class_weight="balanced", max_iter=5000)),
        {"model__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "kNN": (
        lambda: ImbPipeline([("scaler", StandardScaler()),
                             ("smote", SMOTE(random_state=SEED)),
                             ("model", KNeighborsClassifier())]),
        {"model__n_neighbors": [3, 5, 11, 21], "model__weights": ["uniform", "distance"]},
    ),

    "SVM": (
        lambda: pipe(SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED)),
        {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.1]},
    ),
    "RandomForest": (
        # n_jobs=1: the grid search itself already runs in parallel
        lambda: pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=SEED, n_jobs=1)),
        {"model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 3, 5]},
    ),
}