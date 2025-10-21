# src/model/register_model.py
import os
import mlflow
from mlflow.tracking.client import MlflowClient
import yaml
from src.logger import logger
import mlflow
import dagshub

# # Initialize DagsHub + MLflow
# dagshub.init(repo_owner="emaljm", repo_name="MLops-new", mlflow=True)

# # Explicitly set experiment (must be same in both scripts)
# mlflow.set_experiment("SentimentClassifierExperiment")

dagshub_user = os.getenv("MLFLOW_TRACKING_USERNAME")
dagshub_token = os.getenv("MLFLOW_TRACKING_PASSWORD")


# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Construct tracking URI dynamically
tracking_uri = f"https://{dagshub_user}:{dagshub_token}@dagshub.com/{dagshub_user}/MLops-new.mlflow"

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("SentimentClassifierExperiment")



# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Read latest run_id from evaluation
RUN_ID_FILE = "data/metrics/latest_run_id.txt"
if not os.path.exists(RUN_ID_FILE):
    raise FileNotFoundError("Run ID file not found. Please run model_evaluation.py first.")

with open(RUN_ID_FILE, "r") as f:
    run_id = f.read().strip()

REGISTERED_MODEL_NAME = "SentimentClassifier"
STAGE = "Staging"

# Initialize MLflow client
client = MlflowClient()
mlflow.set_experiment("SentimentClassifierExperiment")

# Register the model from the MLflow run
model_uri = f"runs:/{run_id}/model"

logger.info("Registering model %s from run %s", REGISTERED_MODEL_NAME, run_id)
model_version = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
version_number = model_version.version
logger.info("Created model version %s", version_number)

# Transition to staging
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME,
    version=version_number,
    stage=STAGE
)
logger.info("Transitioned model %s version %s to stage %s",
            REGISTERED_MODEL_NAME, version_number, STAGE)
