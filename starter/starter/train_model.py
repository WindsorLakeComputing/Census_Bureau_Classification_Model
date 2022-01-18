# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from .ml.data import process_data, get_cat_features
from .ml.model import train_model, compute_model_metrics, inference

def train_lgbm_model(train_data, label):
    X_train, y_train, encoder, lb = process_data(
        train_data, categorical_features=get_cat_features(), label=label, training=True
    )
    lgbm_class = train_model(X_train, y_train)
    joblib.dump(lgbm_class, '../model/lgbm_class.pkl')
    joblib.dump(encoder, '../model/encoder.pkl')
    joblib.dump(lb, '../model/lb.pkl')

    return lgbm_class, encoder, lb

def get_train_test_data(data_path, test_size):
    data = pd.read_csv(data_path)
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train_data, test_data = train_test_split(data, test_size=test_size)
    print("The count of data is")
    assert (data['occupation'].count() < 32562)
    print(data['occupation'].count())

    return train_data, test_data

def make_predictions(model, label, encoder, lb, test_data):
    X_test, y_test, new_encoder, new_lib  = process_data(
        test_data, categorical_features=get_cat_features(), label=label, training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, X_test)

    return preds, y_test

def test_preditions(preds, y_test):
    precision, recall, beta = compute_model_metrics(y_test, preds)

    print("precision is")
    print(precision)
    print("recall is")
    print(recall)
    print("beta")
    print(beta)

def print_model_metrics(file_name, test_data, label, encoder, lb, model, categorical_col):
    f = open(file_name, "w")
    for unc_e in test_data[categorical_col].unique():
        df_t = test_data[test_data[categorical_col] == unc_e]
        preds, y_test = make_predictions(model, label, encoder, lb, df_t)
        precision, recall, beta = compute_model_metrics(y_test, preds)
        f.write(f"Inside of \"{categorical_col}\" column, the distinct value of \"{unc_e.strip()}\" has the following metrics: \n")
        f.write(f"precision, {precision} recall, {recall} beta, {beta}\n")
    f.close()

if __name__ == "__main__":
    train_data, test_data = get_train_test_data("../data/clean_census.csv", .2)
    lgbm_class, encoder, lb = train_lgbm_model(train_data, "salary")
    preds, y_test = make_predictions(lgbm_class, "salary", encoder, lb, test_data)
    test_preditions(preds, y_test)
    print_model_metrics("slice_output.txt", test_data, "salary", encoder, lb, lgbm_class, "education")