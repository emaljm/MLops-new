# src/data_preprocessing.py
import os
import re
import string
import pandas as pd
import yaml
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logger

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

TEXT_COL = params["preprocessing"]["text_column"]
TARGET_COL = params["preprocessing"]["target_column"]
PROCESSED_PATH = params["data_paths"]["processed_data"]

def preprocess_data_labeling(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Filtering and encoding labels")
        df = df[df[TARGET_COL].isin(["positive", "negative"])].copy()
        df[TARGET_COL] = df[TARGET_COL].replace({"positive": 1, "negative": 0})
        logger.info("Label preprocessing completed, shape: %s", df.shape)
        return df
    except KeyError as e:
        logger.exception("Missing column: %s", e)
        raise

def preprocess_text_col(df: pd.DataFrame, col: str = TEXT_COL) -> pd.DataFrame:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = ''.join([c for c in text if not c.isdigit()])
        text = text.lower()
        text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
        return " ".join(tokens)

    if col not in df.columns:
        logger.error("Column '%s' not found", col)
        raise KeyError(col)

    df = df.copy()
    df[col] = df[col].astype(str).apply(clean_text)
    df = df.dropna(subset=[col])
    logger.info("Text preprocessing completed, shape: %s", df.shape)
    return df

def run_preprocessing():
    from src.data.data_ingestion import ingest_data
    df = ingest_data()
    df = preprocess_data_labeling(df)
    df = preprocess_text_col(df)
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    logger.info("Saved preprocessed data to %s", PROCESSED_PATH)

if __name__ == "__main__":
    run_preprocessing()
