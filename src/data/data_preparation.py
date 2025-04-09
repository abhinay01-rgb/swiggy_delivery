import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import boto3
from io import StringIO
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

TARGET = "time_taken"

# Logger setup
logger = logging.getLogger("data_preparation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# S3 settings
S3_BUCKET = "deliveryswiggyapp"
S3_INPUT_KEY = "cleaned/swiggy_cleaned.csv"
S3_TRAIN_KEY = "interim/train.csv"
S3_TEST_KEY = "interim/test.csv"

# Initialize boto3 client
s3 = boto3.client("s3")

def load_data_from_s3(bucket: str, key: str) -> pd.DataFrame:
    response = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(response['Body'])
    logger.info(f"Loaded data from s3://{bucket}/{key} with shape: {df.shape}")
    return df

def split_data(data: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    logger.info(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    return train_data, test_data

def save_data_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    logger.info(f"Saved data to s3://{bucket}/{key}")

if __name__ == "__main__":

    # Load parameters
    root_path = Path(__file__).parent.parent.parent
    with open(root_path / "params.yaml", "r") as file:
        params = yaml.safe_load(file)
        testsize = params["Data_Preparation"]["test_size"]
        randomstate = params["Data_Preparation"]["random_state"]

    print(f"Test size: {testsize}")
    print(f"Random state: {randomstate}")

    # Load data
    df = load_data_from_s3(S3_BUCKET, S3_INPUT_KEY)

    # Split data
    train_data, test_data = split_data(df, testsize, randomstate)

    # Save data to S3
    save_data_to_s3(train_data, S3_BUCKET, S3_TRAIN_KEY)
    save_data_to_s3(test_data, S3_BUCKET, S3_TEST_KEY)
