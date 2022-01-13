# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from ml.test_model import test_precision, test_recall, test_beta

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
joblib.dump(encoder, '../model/encoder.pkl')
joblib.dump(lb, '../model/lb.pkl')

X_test, y_test, new_encoder, new_lib  = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
print("The race is")
print(test['race'])
print("X_test is ")
print(X_test)

preds = inference(lgbm_class, X_test)
print("the preds are ")
print(preds)
print(y_test)
print(test)
precision, recall, beta = compute_model_metrics(y_test, preds)

test_precision(precision)
test_recall(recall)
test_beta(beta)

f = open("slice_output.txt", "w")
for unc_e in test["education"].unique():
    df_t = test[test["education"] == unc_e]

    X_test_u, y_test_u, encoder_u, lb_u = process_data(
        df_t, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    preds = inference(lgbm_class, X_test_u)
    precision, recall, beta = compute_model_metrics(y_test_u, preds)
    f.write(f"Inside of \"education\" column, the distinct value of \"{unc_e.strip()}\" has the following metrics: \n")
    f.write(f"precision, {precision} recall, {recall} beta, {beta}\n")
f.close()