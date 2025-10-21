# tests/test_model.py
import pytest
import mlflow
import pickle
import tempfile
import os

MODEL_NAME = "SentimentClassifier"

def test_model_load():
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production", "Staging"])
    assert versions, "No model versions found"

    model_uri = f"models:/{MODEL_NAME}/{versions[0].version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load vectorizer from the same run
    run_id = versions[0].run_id
    local_dir = tempfile.mkdtemp()
    vec_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="vectorizer/vectorizer.pkl", dst_path=local_dir)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    sample_input = ["I love this movie!"]
    features = vectorizer.transform(sample_input)  # ✅ 2D
    prediction = model.predict(features)
    assert prediction is not None, "Prediction returned None"
