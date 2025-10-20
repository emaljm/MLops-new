# tests/test_flask_app.py
import pytest
from flask_app.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    rv = client.get("/")
    assert rv.status_code == 200
    assert b"Sentiment Classifier" in rv.data

def test_prediction(client):
    rv = client.post("/", data={"text": "I love this!"})
    assert rv.status_code == 200
    assert b"Positive" in rv.data or b"Negative" in rv.data
