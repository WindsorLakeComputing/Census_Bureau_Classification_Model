# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from ml.data import process_data
from ml.model import train_model
# Add the necessary imports for the starter code.

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
knn_model = train_model(X_train, y_train)
joblib.dump(knn_model, '../model/knn_model.pkl')
# Train and save a model.
