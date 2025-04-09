import pandas as pd
import joblib
import json
from io import BytesIO
from pathlib import Path
import boto3
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constants
TARGET = "time_taken"
S3_BUCKET = "deliveryswiggyapp"
S3_TEST_KEY = "processed/test_trans.csv"
S3_MODEL_KEY = "models/model.pkl"
S3_METRICS_KEY = "metrics/metrics.json"

# S3 Client
s3 = boto3.client("s3")

# === Helper Functions ===

# Load CSV from S3
def load_csv_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'])

# Load model from S3
def load_model_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return joblib.load(BytesIO(obj['Body'].read()))

# Upload metrics to S3
def upload_metrics_to_s3(metrics_dict, bucket, key):
    buffer = BytesIO()
    buffer.write(json.dumps(metrics_dict, indent=4).encode())
    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue(), ContentType='application/json')

# Split into X and y
def make_X_and_y(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

# === Main Logic ===
if __name__ == "__main__":
    # Load test data and model from S3
    test_data = load_csv_from_s3(S3_BUCKET, S3_TEST_KEY)
    model = load_model_from_s3(S3_BUCKET, S3_MODEL_KEY)

    # Prepare features and labels
    X_test, y_test = make_X_and_y(test_data, TARGET)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

    # Print evaluation
    print("✅ Model Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save metrics.json locally (optional)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Upload metrics.json to S3
    upload_metrics_to_s3(metrics, S3_BUCKET, S3_METRICS_KEY)

    print(f"📊 Metrics saved to s3://{S3_BUCKET}/{S3_METRICS_KEY}")
