from fastapi.testclient import TestClient
import os
import sys
sys.path.insert(0, os.getcwd())
import json
from main import app


def test_get_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {
            "greeting": "Hello there!"}

def test_post_true_positive(true_positive):
    with TestClient(app) as client:
        response = client.post("/census/", data=json.dumps(true_positive))
        assert response.status_code == 200
        assert response.json()['response'] == 'The prediction is that the salary is >50K'

def test_post_true_positive(true_positive):
    with TestClient(app) as client:
        response = client.post("/census/", data=json.dumps(true_negative))
        assert response.status_code == 200
        assert response.json()['response'] == 'The prediction is that the salary is <=50K'