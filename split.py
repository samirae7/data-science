import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
CSV_PATH = "CTG_cleaned_V1_gesendet.csv"

# 1. Daten laden
df = pd.read_csv(CSV_PATH)

# 2. Features und Zielvariable trennen
FEATURES = ["LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV",
            "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean",
            "Median", "Variance", "Tendency"]
X = df[FEATURES]
y = df["NSP"].astype(int)

# 3. Stratifizierter Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 4. Kontrolle und Speichern
print(len(X_train), len(X_test))
print(y_train.value_counts(normalize=True).round(3))
print(y_test.value_counts(normalize=True).round(3))

train = X_train.assign(NSP=y_train)
test = X_test.assign(NSP=y_test)
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)