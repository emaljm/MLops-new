# tests/test_model.py
import pytest
import mlflow
import pickle
import os

MODEL_NAME = "SentimentClassifier"

def test_model_load():
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production", "Staging"])
    assert versions, "No model versions found"

    model_uri = f"models:/{MODEL_NAME}/{versions[0].version}"
    model = mlflow.pyfunc.load_model(model_uri)
    sample_input = ["I love this movie!"]
    prediction = model.predict([sample_input[0]])
    assert prediction is not None, "Prediction returned None"
