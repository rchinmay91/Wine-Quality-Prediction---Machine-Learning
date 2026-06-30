import joblib
import pandas as pd

# =====================================================================
# 1. LOAD THE TRAINED MODEL & SCALER
# =====================================================================
print("Loading saved model and scaler assets...")
model = joblib.load("wine_model.pkl")
scaler = joblib.load("wine_scaler.pkl")

# =====================================================================
# 2. INPUT FRESH WINE DATA
# =====================================================================
# Replace these placeholder numbers with any real chemical test values!
# Note: "type_white" is True for white wine, and False for red wine.
new_wine_sample = pd.DataFrame(
    [
        {
            "fixed acidity": 7.4,
            "volatile acidity": 0.25,
            "citric acid": 0.36,
            "residual sugar": 2.0,
            "chlorides": 0.045,
            "free sulfur dioxide": 30.0,
            "total sulfur dioxide": 120.0,
            "density": 0.994,
            "pH": 3.20,
            "sulphates": 0.52,
            "alcohol": 11.5,
            "type_white": True,
        }
    ]
)

# =====================================================================
# 3. PREPROCESS AND PREDICT
# =====================================================================
# The input must be scaled using the exact same scaler from training
scaled_sample = scaler.transform(new_wine_sample)

# Generate prediction (0 = Poor Quality, 1 = Great Quality)
prediction = model.predict(scaled_sample)[0]

# Calculate confidence score percentage
probability = model.predict_proba(scaled_sample)[0][1]

# =====================================================================
# 4. PRINT RESULTS
# =====================================================================
print("\n================== PREDICTION RESULT ==================")
if prediction == 1:
    print(f"Result: HIGH QUALITY WINE (Score 6-10)")
else:
    print(f"Result: LOW QUALITY WINE (Score 1-5)")

print(f"Model Confidence: {probability * 100:.2f}%")
print("=======================================================")
