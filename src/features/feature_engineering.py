# src/feature_engineering.py
import os
import pickle
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from src.logger import logger

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

MAX_FEATURES = params["feature_engineering"]["max_features"]
TEXT_COL = params["preprocessing"]["text_column"]
TRAIN_FEATURES_PATH = params["data_paths"]["features_train"]
TEST_FEATURES_PATH = params["data_paths"]["features_test"]
VECTOR_PATH = params["data_paths"]["vectorizer_output"]
PROCESSED_DATA_PATH = params["data_paths"]["processed_data"]

def apply_bow():
    logger.info("Reading processed data from %s", PROCESSED_DATA_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[params["preprocessing"]["target_column"]], random_state=42)

    vectorizer = CountVectorizer(max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(train_df[TEXT_COL].astype(str))
    X_test = vectorizer.transform(test_df[TEXT_COL].astype(str))

    os.makedirs(os.path.dirname(VECTOR_PATH), exist_ok=True)
    with open(VECTOR_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Saved vectorizer to %s", VECTOR_PATH)

    os.makedirs(os.path.dirname(TRAIN_FEATURES_PATH), exist_ok=True)
    pd.DataFrame(X_train.toarray()).assign(label=train_df[params["preprocessing"]["target_column"]].values).to_csv(TRAIN_FEATURES_PATH, index=False)
    pd.DataFrame(X_test.toarray()).assign(label=test_df[params["preprocessing"]["target_column"]].values).to_csv(TEST_FEATURES_PATH, index=False)
    logger.info("Saved train and test features to %s and %s", TRAIN_FEATURES_PATH, TEST_FEATURES_PATH)

if __name__ == "__main__":
    apply_bow()
