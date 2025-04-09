import numpy as np
import pandas as pd
from pathlib import Path
import boto3
import io

# S3 CONFIG
S3_BUCKET = "deliveryswiggyapp"
RAW_FILE_KEY = "raw/swiggy.csv"
CLEANED_FILE_KEY = "cleaned/swiggy_cleaned.csv"

s3 = boto3.client("s3")

columns_to_drop = [
    'rider_id', 'restaurant_latitude', 'restaurant_longitude', 'delivery_latitude', 'delivery_longitude',
    'order_date', "order_time_hour", "order_day", "city_name", "order_day_of_week", "order_month"
]

def load_data_from_s3(bucket: str, key: str) -> pd.DataFrame:
    print(f"Loading data from S3: s3://{bucket}/{key}")
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(response["Body"].read()))

def upload_data_to_s3(bucket: str, key: str, df: pd.DataFrame):
    print(f"Uploading cleaned data to S3: s3://{bucket}/{key}")
    with io.StringIO() as csv_buffer:
        df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

# All your existing transformation functions (no changes needed)
def change_column_names(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(str.lower, axis=1).rename({
        "delivery_person_id": "rider_id",
        "delivery_person_age": "age",
        "delivery_person_ratings": "ratings",
        "delivery_location_latitude": "delivery_latitude",
        "delivery_location_longitude": "delivery_longitude",
        "time_orderd": "order_time",
        "time_order_picked": "order_picked_time",
        "weatherconditions": "weather",
        "road_traffic_density": "traffic",
        "city": "city_type",
        "time_taken(min)": "time_taken"
    }, axis=1)

def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    minors = data.loc[data['age'].astype('float') < 18].index
    six_star = data.loc[data['ratings'] == "6"].index

    return (
        data.drop(columns="id")
            .drop(index=minors)
            .drop(index=six_star)
            .replace("NaN ", np.nan)
            .assign(
                city_name=lambda x: x['rider_id'].str.split("RES").str.get(0),
                age=lambda x: x['age'].astype(float),
                ratings=lambda x: x['ratings'].astype(float),
                restaurant_latitude=lambda x: x['restaurant_latitude'].abs(),
                restaurant_longitude=lambda x: x['restaurant_longitude'].abs(),
                delivery_latitude=lambda x: x['delivery_latitude'].abs(),
                delivery_longitude=lambda x: x['delivery_longitude'].abs(),
                order_date=lambda x: pd.to_datetime(x['order_date'], dayfirst=True),
                order_day=lambda x: x['order_date'].dt.day,
                order_month=lambda x: x['order_date'].dt.month,
                order_day_of_week=lambda x: x['order_date'].dt.day_name().str.lower(),
                is_weekend=lambda x: x['order_date'].dt.day_name().isin(["Saturday", "Sunday"]).astype(int),
                order_time=lambda x: pd.to_datetime(x['order_time'], format='mixed'),
                order_picked_time=lambda x: pd.to_datetime(x['order_picked_time'], format='mixed'),
                pickup_time_minutes=lambda x: (x['order_picked_time'] - x['order_time']).dt.seconds / 60,
                order_time_hour=lambda x: x['order_time'].dt.hour,
                order_time_of_day=lambda x: time_of_day(x['order_time_hour']),
                weather=lambda x: x['weather'].str.replace("conditions ", "").str.lower().replace("nan", np.nan),
                traffic=lambda x: x['traffic'].str.rstrip().str.lower(),
                type_of_order=lambda x: x['type_of_order'].str.rstrip().str.lower(),
                type_of_vehicle=lambda x: x['type_of_vehicle'].str.rstrip().str.lower(),
                festival=lambda x: x['festival'].str.rstrip().str.lower(),
                city_type=lambda x: x['city_type'].str.rstrip().str.lower(),
                multiple_deliveries=lambda x: x['multiple_deliveries'].astype(float),
                time_taken=lambda x: x['time_taken'].str.replace("(min) ", "").astype(int)
            )
            .drop(columns=["order_time", "order_picked_time"])
    )

def clean_lat_long(data: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    location_columns = ['restaurant_latitude', 'restaurant_longitude', 'delivery_latitude', 'delivery_longitude']
    return data.assign(**{
        col: np.where(data[col] < threshold, np.nan, data[col]) for col in location_columns
    })

def time_of_day(series: pd.Series):
    return pd.cut(series, bins=[0, 6, 12, 17, 20, 24], right=True,
                  labels=["after_midnight", "morning", "afternoon", "evening", "night"])

def calculate_haversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    lat1 = np.radians(df['restaurant_latitude'])
    lon1 = np.radians(df['restaurant_longitude'])
    lat2 = np.radians(df['delivery_latitude'])
    lon2 = np.radians(df['delivery_longitude'])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return df.assign(distance=distance)

def create_distance_type(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(
        distance_type=pd.cut(data["distance"], bins=[0, 5, 10, 15, 25],
                             right=False, labels=["short", "medium", "long", "very_long"])
    )

def drop_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    return data.drop(columns=columns)

def perform_data_cleaning_and_upload():
    df = load_data_from_s3(S3_BUCKET, RAW_FILE_KEY)
    cleaned_df = (
        df
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
        .pipe(drop_columns, columns=columns_to_drop)
    )
    upload_data_to_s3(S3_BUCKET, CLEANED_FILE_KEY, cleaned_df)

if __name__ == "__main__":
    perform_data_cleaning_and_upload()
