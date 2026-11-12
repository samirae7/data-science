import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE

from split import X_train, y_train, X_test, y_test

seed=42

#train_CV = pd.read_csv("train.csv")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#fold_f1_scores = []

# Manually loop through the folds
for train_index, test_index in cv.split(X_train, y_train):
    
    # 1. Split the fold data
    X_fold_train, X_fold_test = X_train.iloc[train_index], X_train.iloc[test_index]
    y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]
    
    # 2. Scale the data safely inside the loop
    scaler = StandardScaler()
    X_fold_train_scaled = scaler.fit_transform(X_fold_train)
    X_fold_test_scaled = scaler.transform(X_fold_test)
    
    # 3. Train and Predict
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    model.fit(X_fold_train_scaled, y_fold_train)
    y_pred = model.predict(X_fold_test_scaled)



