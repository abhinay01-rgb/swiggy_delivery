from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path
import os

app = Flask(__name__)

# === Local Paths ===
LOCAL_MODEL_PATH = Path("models/model.pkl")
LOCAL_PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

# Ensure model and preprocessor exist
if not LOCAL_MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Model file not found at: {LOCAL_MODEL_PATH}")
if not LOCAL_PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(f"❌ Preprocessor file not found at: {LOCAL_PREPROCESSOR_PATH}")

# Load model and preprocessor
model = joblib.load(LOCAL_MODEL_PATH)
preprocessor = joblib.load(LOCAL_PREPROCESSOR_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Get form data
            form_data = {
                "age": int(request.form["age"]),
                "ratings": float(request.form["ratings"]),
                "pickup_time_minutes": int(request.form["pickup_time_minutes"]),
                "distance": float(request.form["distance"]),
                "weather": request.form["weather"],
                "type_of_order": request.form["type_of_order"],
                "type_of_vehicle": request.form["type_of_vehicle"],
                "festival": request.form["festival"],
                "city_type": request.form["city_type"],
                "is_weekend": request.form["is_weekend"],
                "order_time_of_day": request.form["order_time_of_day"],
                "traffic": request.form["traffic"],
                "distance_type": request.form["distance_type"],
                "multiple_deliveries": int(request.form["multiple_deliveries"]),
                "vehicle_condition": int(request.form["vehicle_condition"])
            }

            # Convert to DataFrame and Predict
            df = pd.DataFrame([form_data])
            transformed = preprocessor.transform(df)
            pred = model.predict(transformed)[0]
            prediction = round(pred, 2)

        except Exception as e:
            prediction = f"Prediction Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
