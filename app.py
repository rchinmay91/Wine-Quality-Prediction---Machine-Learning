import gradio as gr
import joblib
import pandas as pd

# =====================================================================
# 1. LOAD THE TRAINED MODEL & SCALER
# =====================================================================
try:
    model = joblib.load("wine_model.pkl")
    scaler = joblib.load("wine_scaler.pkl")
except Exception as e:
    print(
        "Error loading model assets. Make sure 'wine_model.pkl' and 'wine_scaler.pkl' are in this folder!"
    )


# =====================================================================
# 2. PREDICTION FUNCTION
# =====================================================================
def predict_wine_quality(
    wine_type,
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    pH,
    sulphates,
    alcohol,
):
    # Convert text selection to the training feature format
    type_white = True if wine_type == "White" else False

    # Organize inputs into a matching DataFrame structure
    input_data = pd.DataFrame(
        [
            {
                "fixed acidity": fixed_acidity,
                "volatile acidity": volatile_acidity,
                "citric acid": citric_acid,
                "residual sugar": residual_sugar,
                "chlorides": chlorides,
                "free sulfur dioxide": free_sulfur_dioxide,
                "total sulfur dioxide": total_sulfur_dioxide,
                "density": density,
                "pH": pH,
                "sulphates": sulphates,
                "alcohol": alcohol,
                "type_white": type_white,
            }
        ]
    )

    # Scale raw values using training parameters
    scaled_input = scaler.transform(input_data)

    # Generate model score metrics
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    # Structure outputs beautifully
    result = "🍷 High Quality" if prediction == 1 else "🫗 Low Quality"
    confidence = (
        probabilities[1] if prediction == 1 else probabilities[0]
    ) * 100

    return f"Prediction: {result}\nConfidence: {confidence:.2f}%"


# =====================================================================
# 3. GRADIO INTERFACE SETUP
# =====================================================================
inputs = [
    gr.Dropdown(["White", "Red"], label="Wine Type", value="White"),
    gr.Slider(3.8, 15.9, value=7.0, step=0.1, label="Fixed Acidity"),
    gr.Slider(0.08, 1.58, value=0.3, step=0.01, label="Volatile Acidity"),
    gr.Slider(0.0, 1.66, value=0.3, step=0.01, label="Citric Acid"),
    gr.Slider(0.6, 65.8, value=5.0, step=0.1, label="Residual Sugar"),
    gr.Slider(0.009, 0.611, value=0.05, step=0.001, label="Chlorides"),
    gr.Slider(1.0, 289.0, value=30.0, step=1.0, label="Free Sulfur Dioxide"),
    gr.Slider(6.0, 440.0, value=115.0, step=1.0, label="Total Sulfur Dioxide"),
    gr.Slider(0.987, 1.039, value=0.996, step=0.0001, label="Density"),
    gr.Slider(2.72, 4.01, value=3.2, step=0.01, label="pH Level"),
    gr.Slider(0.22, 2.0, value=0.5, step=0.01, label="Sulphates"),
    gr.Slider(8.0, 14.9, value=10.5, step=0.1, label="Alcohol Content (%)"),
]

outputs = gr.Textbox(label="Model Verdict")

# Combine everything into an interactive window block
demo = gr.Interface(
    fn=predict_wine_quality,
    inputs=inputs,
    outputs=outputs,
    title="Wine Quality Prediction App",
    description="Adjust slider parameters to evaluate chemical configurations on the trained model.",
)

if __name__ == "__main__":
    demo.launch()
