# src/model_building.py
import os
import pickle
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.logger import logger

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

TRAIN_FEATURES_PATH = params["data_paths"]["features_train"]
MODEL_OUTPUT_PATH = params["data_paths"]["model_output"]

def train_model():
    df = pd.read_csv(TRAIN_FEATURES_PATH)
    X_train = df.drop(columns=["label"]).values
    y_train = df["label"].values

    cfg = params["model_building"]
    model = LogisticRegression(
        penalty=cfg["penalty"],
        solver=cfg["solver"],
        C=cfg["C"],
        random_state=cfg["random_state"],
        max_iter=1000
    )
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model to %s", MODEL_OUTPUT_PATH)

if __name__ == "__main__":
    train_model()
