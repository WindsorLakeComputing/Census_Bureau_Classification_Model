from fastapi.testclient import TestClient
from fastapi import FastAPI
from main import app
import pytest
import os
import sys
sys.path.insert(0, os.getcwd())
from main import app
import json
from main import app

# Since I introduced model load on startup in main.py, there was an issue with testing
# models as they were not loading. This structure allows to pass tests with async model load



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
        assert response.json() == {
            "The prediction is that the salary is >50K"}