import pandas as pd
from ..starter.starter.ml.data import process_data

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

def test_precision(test_data):
    """ The ratio between the True Positives and all the Positives."""
    assert(precision >= .75)

def test_recall(recall):
    """ The measure of our model correctly identifying True Positives."""
    assert(recall >= .60)

def test_beta(beta):
    """The Fbeta-measure measure is an abstraction of the F-measure where the balance of precision and recall
     in the calculation of the harmonic mean is controlled by a coefficient called beta."""
    assert(beta >= .70)