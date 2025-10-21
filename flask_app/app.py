from flask import Flask, request, render_template
import mlflow
import os
import pickle
import prometheus_client
from prometheus_client import Counter, Histogram
import time
import tempfile
import numpy as np

# ==========================================================
# CONFIGURATION
# ==========================================================
app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'flask_app_request_count', 'Total requests',
    ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'flask_app_request_latency_seconds', 'Request latency (seconds)'
)

dagshub_user = os.getenv("MLFLOW_TRACKING_USERNAME")
dagshub_token = os.getenv("MLFLOW_TRACKING_PASSWORD")


# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Construct tracking URI dynamically
tracking_uri = f"https://{dagshub_user}:{dagshub_token}@dagshub.com/{dagshub_user}/MLops-new.mlflow"

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri(tracking_uri)


MODEL_NAME = "SentimentClassifier"


# ==========================================================
# LOAD MODEL FROM DAGSHUB
# ==========================================================
def load_model_from_dagshub(model_name):
    try:
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])
        if not versions:
            raise Exception("No registered model versions found.")

        version_info = versions[0]
        print(f"🔍 Found model version: {version_info.version}, stage: {version_info.current_stage}")
        model_uri = f"models:/{model_name}/{version_info.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Loaded model version {version_info.version}")
        return model, version_info.run_id
    except Exception as e:
        print(f"❌ Error loading model from DagsHub: {e}")
        return None, None


# ==========================================================
# LOAD VECTORIZER FROM DAGSHUB ARTIFACTS
# ==========================================================
def load_vectorizer_from_dagshub(run_id):
    try:
        print(f"Fetching vectorizer from run {run_id}")

        local_dir = tempfile.mkdtemp()
        # NOTE: Ensure this matches the artifact_path you logged during model evaluation
        vec_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="vectorizer/vectorizer.pkl",  # Correct artifact path
            dst_path=local_dir
        )
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        print("✅ Vectorizer loaded from DagsHub artifacts.")
        return vectorizer
    except Exception as e:
        print(f"❌ Could not load vectorizer from DagsHub: {e}")
        return None


# Load model and vectorizer
model, run_id = load_model_from_dagshub(MODEL_NAME)
vectorizer = load_vectorizer_from_dagshub(run_id) if run_id else None


# ==========================================================
# FLASK ROUTES
# ==========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    start_time = time.time()
    try:
        if request.method == "POST":
            text = request.form.get("text", "").strip()
            if not text:
                REQUEST_COUNT.labels('POST', '/', 400).inc()
                return render_template("index.html", input_text="", prediction="Please enter some text.")

            if vectorizer is None:
                REQUEST_COUNT.labels('POST', '/', 500).inc()
                return render_template("index.html", input_text=text, prediction="Vectorizer not available.")

            if model is None:
                REQUEST_COUNT.labels('POST', '/', 500).inc()
                return render_template("index.html", input_text=text, prediction="Model not available.")

            # Vectorize input
            features = vectorizer.transform([text])
            # Ensure features are 2D
            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            pred = model.predict(features)[0]
            label = "Positive 😀" if pred == 1 else "Negative 😞"

            latency = time.time() - start_time
            REQUEST_COUNT.labels('POST', '/', 200).inc()
            REQUEST_LATENCY.observe(latency)

            return render_template("index.html", input_text=text, prediction=label)

        REQUEST_COUNT.labels('GET', '/', 200).inc()
        return render_template("index.html")

    except Exception as e:
        print(f"Prediction error: {e}")
        REQUEST_COUNT.labels(request.method, '/', 500).inc()
        return render_template("index.html", input_text="Error", prediction="Something went wrong.")


@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics."""
    return prometheus_client.generate_latest(), 200, {'Content-Type': prometheus_client.CONTENT_TYPE_LATEST}


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
