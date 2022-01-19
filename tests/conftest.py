import pytest
import pandas as pd
import joblib

@pytest.fixture(scope="module")
def data():
    data = pd.read_csv("starter/data/clean_census.csv")

    return data

@pytest.fixture(scope="module")
def data():
    data = pd.read_csv("starter/data/clean_census.csv")

    return data

@pytest.fixture(scope="module")
def model():
    lgbm_class = joblib.load('starter/model/lgbm_class.pkl')

    return lgbm_class

@pytest.fixture(scope="module")
def encoder():
    encoder = joblib.load('starter/model/encoder.pkl')

    return encoder

@pytest.fixture(scope="module")
def lb():
    lb = joblib.load('starter/model/lb.pkl')

    return lb

@pytest.fixture(scope="module")
def true_positive():
    true_p = {'age': '47', 'workclass': 'Private-gov', 'fnlgt': '51835', 'education': 'Prof-school', 'education-num': '15','marital-status': 'Married-civ-spouse','occupation': 'Prof-specialty', 'relationship': 'Wife','race': 'White','sex': 'Female','capital-gain': '0','capital-loss':'1902','hours-per-week': '60', 'native-country': 'Honduras'}

    return true_p

@pytest.fixture(scope="module")
def true_negative(scope="module"):
    true_n = {'age': '50', 'workclass': 'Self-emp-not-inc', 'fnlgt': '83311', 'education': 'Bachelors', 'education-num': '13','marital-status': 'Married-civ-spouse','occupation': 'Exec-managerial', 'relationship': 'Husband','race': 'White','sex': 'Male','capital-gain': '0','capital-loss':'0','hours-per-week': '13', 'native-country': 'United-States'}

    return true_n

@pytest.fixture(scope="module")
def cat_features():
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

    return cat_features