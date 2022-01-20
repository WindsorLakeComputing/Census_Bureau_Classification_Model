from fastapi.testclient import TestClient
from main import app
import pytest

# Since I introduced model load on startup in main.py, there was an issue with testing
# models as they were not loading. This structure allows to pass tests with async model load


def test_get_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "greeting": "Hello there?"}