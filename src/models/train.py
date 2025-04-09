import pandas as pd
import joblib
import yaml
from io import BytesIO
from pathlib import Path
import boto3
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor

# Constants
S3_BUCKET = "deliveryswiggyapp"
S3_TRAIN_PROCESSED_KEY = "processed/train_trans.csv"
S3_MODEL_SAVE_KEY = "models/model.pkl"
TARGET = "time_taken"

# Initialize S3 client
s3 = boto3.client("s3")

# === Helper Functions ===

# Load CSV from S3
def load_data_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])

# Save model to S3
def save_model_to_s3(model, bucket, key):
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

# Read YAML locally
def read_local_params(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

# Get X and y
def make_X_and_y(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

# Return model instance
def get_model(model_name, params):
    if model_name == "XGBoost":
        return XGBRegressor(**params)
    elif model_name == "RandomForest":
        return RandomForestRegressor(**params)
    elif model_name == "LightGBM":
        return LGBMRegressor(**params)
    elif model_name == "DecisionTree":
        return DecisionTreeRegressor(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

# === Main Flow ===
if __name__ == "__main__":
    # Load training data from S3
    data = load_data_from_s3(S3_BUCKET, S3_TRAIN_PROCESSED_KEY)
    X_train, y_train = make_X_and_y(data, TARGET)

    # Load params from local file
    root_path = Path(__file__).parent.parent.parent
    params_path = root_path / "params.yaml"
    params = read_local_params(params_path)

    model_name = params["Train"]["model_name"]
    model_params = params["Train"][model_name]

    # Create and train model
    model = get_model(model_name, model_params)
    model.fit(X_train, y_train)

    # Save model to S3
    save_model_to_s3(model, S3_BUCKET, S3_MODEL_SAVE_KEY)

    print(f"✅ {model_name} model trained and saved to s3://{S3_BUCKET}/{S3_MODEL_SAVE_KEY}")
