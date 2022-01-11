# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
local_path = "../data/clean_census.csv"
data = pd.read_csv(local_path)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
lgbm_class = train_model(X_train, y_train)
joblib.dump(lgbm_class, '../model/lgbm_class.pkl')

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

preds = inference(lgbm_class, X_test)
precision, recall, beta = compute_model_metrics(y_test, preds)
