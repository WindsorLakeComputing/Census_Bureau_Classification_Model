import pandas as pd
import os
import sys
sys.path.insert(0, os.getcwd())
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

def test_data_amount(data):
    assert 32500 < data.shape[0] < 1000000

def test_true_positive(true_positive, model,encoder,lb,cat_features):
    df = pd.DataFrame([true_positive])

    X_test, y_test, new_encoder, new_lib = process_data(
        df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, X_test)
    assert(preds[0] == 1)

def test_true_negative(true_negative, model,encoder,lb,cat_features):
    df = pd.DataFrame([true_negative])

    X_test, y_test, new_encoder, new_lib = process_data(
        df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, X_test)
    assert(preds[0] == 0)