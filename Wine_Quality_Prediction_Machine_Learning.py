import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Suppress warnings
warnings.filterwarnings("ignore")

# =====================================================================
# 1. LOAD DATASET
# =====================================================================
# Ensure 'winequality.csv' is placed in your active working directory
df = pd.read_csv("winequality.csv")
print("--- Dataset Head ---")
print(df.head())

# =====================================================================
# 2. DATA CLEANING & PREPROCESSING
# =====================================================================
# Handle missing values by replacing them with the column mean
missing_cols = df.columns[df.isnull().any()]
for col in missing_cols:
    df[col] = df[col].fillna(df[col].mean())

# Convert categorical 'type' column (white/red) into numerical binary flags
df = pd.get_dummies(df, columns=["type"], drop_first=True)

# Define target variable: Convert quality scores into binary classes
# Good quality wine (1) if score is 6 or higher, otherwise bad quality (0)
df["best_quality"] = df["quality"].apply(lambda x: 1 if x >= 6 else 0)

# =====================================================================
# 3. FEATURE SELECTION & TRAIN-TEST SPLIT
# =====================================================================
# Drop original quality strings/scores to isolate features
X = df.drop(["quality", "best_quality"], axis=1)
y = df["best_quality"]

# Split data into 80% training and 20% validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=40
)

# Scale features uniformly between 0 and 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# =====================================================================
# 4. MODEL TRAINING & EVALUATION
# =====================================================================
models = [LogisticRegression(), SVC(probability=True), XGBClassifier()]

for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)

    print(f"\n==================== {model_name} ====================")
    print(
        f"Training Accuracy : {metrics.roc_auc_score(y_train, model.predict_proba(X_train)[:,1]):.4f}"
    )
    print(
        f"Validation Accuracy: {metrics.roc_auc_score(y_val, model.predict_proba(X_val)[:,1]):.4f}"
    )
