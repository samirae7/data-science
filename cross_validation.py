import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_selection import RFE

from split import X_train, y_train, X_test, y_test

SEED =42

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state= SEED)

# Create a list to store the scores for each fold
fold_f1_scores = []

# Manually loop through the folds
fold_number = 1
for train_index, val_index in cv.split(X_train, y_train):
    
    # 1. Split the fold data 
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # 2. Scale the data safely inside the loop
    scaler = StandardScaler()
    X_fold_train_scaled = scaler.fit_transform(X_fold_train)
    X_fold_val_scaled = scaler.transform(X_fold_val)
    
    # 3. Train and Predict
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED)
    model.fit(X_fold_train_scaled, y_fold_train)
    y_pred = model.predict(X_fold_val_scaled)
    
    # 4. Evaluate and Store the Score
    # Using average='macro' because of class imbalance
    fold_f1 = f1_score(y_fold_val, y_pred, average='macro')
    fold_f1_scores.append(fold_f1)
    
    print(f"Fold {fold_number} Macro F1-Score: {fold_f1:.3f}")
    fold_number += 1

# Print the final averaged result across all 5 folds
print(f"\nAverage Cross-Validation Macro F1-Score: {np.mean(fold_f1_scores):.3f}")




