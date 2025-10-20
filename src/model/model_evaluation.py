# src/model/model_evaluation.py
import os
import json
import pickle
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import dagshub
from src.logger import logger

# ============================
# Initialize DagsHub + MLflow
# ============================
dagshub.init(repo_owner="emaljm", repo_name="MLops-new", mlflow=True)
mlflow.set_experiment("SentimentClassifierExperiment")

# ============================
# Load parameters
# ============================
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

FEATURE_TEST = params["data_paths"]["features_test"]
MODEL_PATH = params["data_paths"]["model_output"]
VECTORIZER_PATH = params["data_paths"]["vectorizer_output"]
EVAL_OUTPUT = params["data_paths"]["evaluation_output"]


def evaluate_model():
    try:
        # ----------------------------
        # Load test data
        # ----------------------------
        logger.info("Loading test features from %s", FEATURE_TEST)
        test_df = pd.read_csv(FEATURE_TEST)
        X_test = test_df.drop(columns=["label"]).values
        y_test = test_df["label"].values

        # ----------------------------
        # Load model
        # ----------------------------
        logger.info("Loading trained model from %s", MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        # ----------------------------
        # Load vectorizer
        # ----------------------------
        logger.info("Loading vectorizer from %s", VECTORIZER_PATH)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)

        # ----------------------------
        # Evaluate model
        # ----------------------------
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        logger.info("Evaluation metrics: %s", metrics)

        # ----------------------------
        # Save metrics locally
        # ----------------------------
        os.makedirs(os.path.dirname(EVAL_OUTPUT), exist_ok=True)
        with open(EVAL_OUTPUT, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Saved evaluation metrics to %s", EVAL_OUTPUT)

        # ----------------------------
        # Log model, vectorizer, and metrics to MLflow
        # ----------------------------
        with mlflow.start_run(run_name="Evaluation") as run:
            # Log metrics
            mlflow.log_metrics(metrics)

            # Log trained model
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Log vectorizer explicitly
            mlflow.log_artifact(VECTORIZER_PATH, artifact_path="vectorizer")
            logger.info("Logged model and vectorizer to MLflow.")

            # Save run_id for downstream stages
            run_id = run.info.run_id
            os.makedirs("data/metrics", exist_ok=True)
            with open("data/metrics/latest_run_id.txt", "w") as f:
                f.write(run_id)

            logger.info("MLflow run_id saved: %s", run_id)

        return run_id

    except Exception as e:
        logger.exception("Error during model evaluation: %s", e)
        raise


if __name__ == "__main__":
    evaluate_model()
