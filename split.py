import sys
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, make_scorer,
                             recall_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "CTG_cleaned_V1.csv"
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "ergebnisse")
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
TARGET = "NSP"                       # 1 = Normal, 2 = Suspect, 3 = Pathologic
CLASS_NAMES = {1: "Normal", 2: "Suspect", 3: "Pathologic"}
PATHO = 3
 
# Feature-Pool = die 21 automatisch erhobenen Messmerkmale.
# NICHT enthalten (Begruendung):
#   CLASS, A, B, C, D, E, AD, DE, LD, FS, SUSP : morphologisches Muster = bereits
#       eine Experten-Einordnung der Aufnahme (Leakage; allein Macro-F1 ~0.97)
#   LBE  : Baseline vom Experten, identisch mit LB (100 % gleiche Werte)
#   b, e : Start-/Endzeitpunkt der Aufnahme, keine Messung am Foetus
#   has_na : konstant
MEASUREMENT_FEATURES = ["LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV",
                        "ALTV", "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros",
                        "Mode", "Mean", "Median", "Variance", "Tendency"]
 
SEED = 42                            # einmal festlegen, nicht mehr aendern
TEST_SIZE = 0.2
N_SPLITS, N_REPEATS = 5, 5
 
# Subset-Groessen, in dieser Reihenfolge getestet (gross -> klein)
K_GRID = [15, 10, 8, 5, 3]
 
# Hauptentscheidung: nur Macro-F1. Toleranz vor dem Lauf festgelegt.
DELTA_F1 = 0.02
# Sicherheitsmetrik (nur beschreibend, steuert die Entscheidung NICHT):
# Pathologic hat nur ~34 Faelle pro Fold, der Recall schwankt stark; eine enge
# Toleranz waere mit dieser Datenmenge fast nie nachweisbar.
DELTA_RECALL_PATH = 0.05
ALPHA = 0.05                         # einseitig
N_BOOT = 2000
 
# =============================================================================
# 1. Daten laden und pruefen
# =============================================================================
print(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | "
      f"pandas {pd.__version__} | numpy {np.__version__}")
print(f"Seed={SEED}, Test={TEST_SIZE}, CV={N_SPLITS}x{N_REPEATS}, K_GRID={K_GRID}, "
      f"delta_F1={DELTA_F1}, alpha={ALPHA}\n")
 
df = pd.read_csv(CSV_PATH)
print(f"Daten: {df.shape[0]} Zeilen, {df.shape[1]} Spalten | "
      f"fehlende Werte: {int(df.isna().sum().sum())} | Duplikate: {int(df.duplicated().sum())}")
missing = [c for c in MEASUREMENT_FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise ValueError(f"Spalten fehlen in der CSV: {missing}")
 
X = df[MEASUREMENT_FEATURES].astype(float)
y = df[TARGET].astype(int)
print("Klassen gesamt:", y.map(CLASS_NAMES).value_counts().to_dict())
print(f"Feature-Pool: {X.shape[1]} Messmerkmale")
 
# =============================================================================
# 2. Stratifizierter Split
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)} | "
      f"Pathologic im Test: {int((y_test == PATHO).sum())}")
print("Anteile Train:", y_train.map(CLASS_NAMES).value_counts(normalize=True).round(3).to_dict())
print("Anteile Test: ", y_test.map(CLASS_NAMES).value_counts(normalize=True).round(3).to_dict())