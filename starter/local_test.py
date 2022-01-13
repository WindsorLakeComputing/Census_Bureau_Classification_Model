from fastapi.testclient import TestClient
import json

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    rsp = json.loads(r.text)
    assert (r.status_code == 200) and (rsp.get("greeting") == "Hello there!")

def test_get_over_50K():
    over_50k_entry = {'age': '47', 'workclass': 'Private-gov', 'fnlgt': '51835', 'education': 'Prof-school',
             'education_num': '15', 'marital_status': 'Married-civ-spouse', 'occupation': 'Prof-specialty',
             'relationship': 'Wife', 'race': 'White', 'sex': 'Female', 'capital_gain': '0', 'capital_loss': '1902',
             'hours_per_week': '60', 'native_country': 'Honduras'}

    r = client.post("/census/", data=json.dumps(over_50k_entry))
    assert r.text.strip('"') == "The prediction is that the salary is >50K"

def test_get_under_50K():
    over_50k_entry = {'age': '50', 'workclass': 'Self-emp-not-inc', 'fnlgt': '83311', 'education': 'Bachelors',
             'education_num': '13','marital_status': 'Married_civ_spouse','occupation': 'Exec_managerial',
             'relationship': 'Husband','race': 'White','sex': 'Male','capital_gain': '0','capital_loss':'0',
             'hours_per_week': '13', 'native_country': 'United-States'}

    r = client.post("/census/", data=json.dumps(over_50k_entry))
    assert r.text.strip('"') == "The prediction is that the salary is <=50K"

test_api_locally_get_root()
test_get_over_50K()
test_get_under_50K()