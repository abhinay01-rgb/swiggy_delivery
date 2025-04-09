import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn import set_config
import joblib
from io import StringIO, BytesIO
import boto3

# Output as DataFrame
set_config(transform_output='pandas')

# S3 settings
S3_BUCKET = "deliveryswiggyapp"
S3_TRAIN_KEY = "interim/train.csv"
S3_TEST_KEY = "interim/test.csv"
S3_PROCESSED_TRAIN_KEY = "processed/train_trans.csv"
S3_PROCESSED_TEST_KEY = "processed/test_trans.csv"
S3_MODEL_KEY = "models/preprocessor.pkl"

# Initialize S3
s3 = boto3.client("s3")

# === Column Types ===
num_cols = ["age", "ratings", "pickup_time_minutes", "distance"]
nominal_cat_cols = ['weather', 'type_of_order', 'type_of_vehicle', "festival", "city_type", "is_weekend", "order_time_of_day"]
ordinal_cat_cols = ["traffic", "distance_type"]
target_col = "time_taken"

# === Ordinal Encoding Orders ===
traffic_order = ["low", "medium", "high", "jam"]
distance_type_order = ["short", "medium", "long", "very_long"]

# === Load Data from S3 ===
def load_csv_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(response['Body']).dropna()

train_df = load_csv_from_s3(S3_BUCKET, S3_TRAIN_KEY)
test_df = load_csv_from_s3(S3_BUCKET, S3_TEST_KEY)

# === Split Features & Target ===
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# === Preprocessor ===
preprocessor = ColumnTransformer([
    ("scale", MinMaxScaler(), num_cols),
    ("nominal", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), nominal_cat_cols),
    ("ordinal", OrdinalEncoder(categories=[traffic_order, distance_type_order],
                               encoded_missing_value=-999,
                               handle_unknown="use_encoded_value",
                               unknown_value=-1), ordinal_cat_cols)
], remainder="passthrough", verbose_feature_names_out=False)

# === Fit and Transform ===
preprocessor.fit(X_train)
X_train_trans = preprocessor.transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# === Save Transformed CSVs to S3 ===
def save_df_to_s3(df, bucket, key):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

train_trans_df = X_train_trans.join(y_train)
test_trans_df = X_test_trans.join(y_test)

save_df_to_s3(train_trans_df, S3_BUCKET, S3_PROCESSED_TRAIN_KEY)
save_df_to_s3(test_trans_df, S3_BUCKET, S3_PROCESSED_TEST_KEY)

# === Save Preprocessor to S3 ===
def save_pickle_to_s3(obj, bucket, key):
    pickle_buffer = BytesIO()
    joblib.dump(obj, pickle_buffer)
    pickle_buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=pickle_buffer.getvalue())

save_pickle_to_s3(preprocessor, S3_BUCKET, S3_MODEL_KEY)

print(f"✅ Transformed CSVs saved to s3://{S3_BUCKET}/processed/")
print(f"✅ Preprocessor saved to s3://{S3_BUCKET}/models/preprocessor.pkl")
