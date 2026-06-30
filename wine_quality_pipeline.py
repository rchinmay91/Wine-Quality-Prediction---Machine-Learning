import joblib
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Suppress warnings
warnings.filterwarnings("ignore")

# =====================================================================
# 1. LOAD DATASET
# =====================================================================
df = pd.read_csv("winequality.csv")

# =====================================================================
# 2. DATA CLEANING & PREPROCESSING
# =====================================================================
# Handle missing values by replacing them with the column mean
missing_cols = df.columns[df.isnull().any()]
for col in missing_cols:
    df[col] = df[col].fillna(df[col].mean())

# Convert categorical 'type' column (white/red) into numerical binary flags
df = pd.get_dummies(df, columns=["type"], drop_first=True)

# Define target variable: 1 if quality score >= 6, else 0
df["best_quality"] = df["quality"].apply(lambda x: 1 if x >= 6 else 0)

# =====================================================================
# 3. EXPLORATORY DATA ANALYSIS (VISUALIZATION)
# =====================================================================
print("Generating Feature Correlation Heatmap...")
plt.figure(figsize=(12, 10))
sb.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Wine Features")
plt.tight_layout()
plt.show()  # Note: Close the pop-up graph window to let the script continue running!

# =====================================================================
# 4. FEATURE SELECTION & TRAIN-TEST SPLIT
# =====================================================================
X = df.drop(["quality", "best_quality"], axis=1)
y = df["best_quality"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=40
)

# Scale features uniformly between 0 and 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# =====================================================================
# 5. BASE MODEL TRAINING & EVALUATION
# =====================================================================
models = [LogisticRegression(), SVC(probability=True), XGBClassifier()]

print("\n--- Evaluating Base Models (ROC-AUC Scores) ---")
for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)

    train_auc = metrics.roc_auc_score(
        y_train, model.predict_proba(X_train)[:, 1]
    )
    val_auc = metrics.roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    print(
        f"{model_name:<20} -> Train AUC: {train_auc:.4f} | Validation AUC: {val_auc:.4f}"
    )

# =====================================================================
# 6. HYPERPARAMETER TUNING FOR XGBOOST
# =====================================================================
print("\n--- Starting XGBoost Hyperparameter Tuning ---")

# Fixed by defining list variables explicitly to prevent copy-paste clipping
depth_options = [3, 5, 7]
estimator_options = [50, 100, 200]

param_grid = {
    "max_depth": depth_options,
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": estimator_options,
    "subsample": [0.8, 1.0],
}



xgb = XGBClassifier(eval_metric="logloss")
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    verbose=1,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters Found for XGBoost:")
print(grid_search.best_params_)

# FIXED: Changed from best_estimator__ to best_estimator_
best_xgb = grid_search.best_estimator_

# =====================================================================
# 7. PRODUCTION METRICS (CONFUSION MATRIX)
# =====================================================================
print("\n================== CONFUSION MATRIX ==================")
predictions = best_xgb.predict(X_val)
print(metrics.confusion_matrix(y_val, predictions))

print("\n================ CLASSIFICATION REPORT ================")
print(metrics.classification_report(y_val, predictions))


# =====================================================================
# 8. SAVE MODEL AND SCALER FOR DEPLOYMENT
# =====================================================================
joblib.dump(best_xgb, "wine_model.pkl")
joblib.dump(scaler, "wine_scaler.pkl")
print("\nModel and Scaler successfully saved to files!")
