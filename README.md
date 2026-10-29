# data-science
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE

# 1. DATA ACQUISITION & CLEANING
xls = pd.ExcelFile('CTG.xls')
df = pd.read_excel(xls, sheet_name='Raw Data')

# Drop empty rows and irrelevant/leaky metadata
df = df.dropna(how='all')
columns_to_drop = ['FileName', 'Date', 'SegFile', 'b', 'e', 'CLASS', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP']
df_clean = df.drop(columns=columns_to_drop)

# 2. SEPARATE FEATURES AND TARGET
X = df_clean.drop('NSP', axis=1)
y = df_clean['NSP']

# Train/Test Split (Stratified due to imbalance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. BASELINE MODEL (All Features)
baseline_model = RandomForestClassifier(class_weight='balanced', random_state=42)
baseline_model.fit(X_train_scaled, y_train)
y_pred_base = baseline_model.predict(X_test_scaled)
print("Baseline Performance:\n", classification_report(y_test, y_pred_base))

# 4. FEATURE SELECTION (Testing HA)
# Example using RFE to select the top 5 features
selector = RFE(baseline_model, n_features_to_select=5, step=1)
selector = selector.fit(X_train_scaled, y_train)
selected_features = X.columns[selector.support_]
print("Selected Subset:", selected_features.tolist())

# 5. SUBSET MODEL
X_train_sub = scaler.fit_transform(X_train[selected_features])
X_test_sub = scaler.transform(X_test[selected_features])

subset_model = RandomForestClassifier(class_weight='balanced', random_state=42)
subset_model.fit(X_train_sub, y_train)
y_pred_sub = subset_model.predict(X_test_sub)
print("Subset Model Performance:\n", classification_report(y_test, y_pred_sub))
