# src/data_ingestion.py
import os
import pandas as pd
import yaml
from src.logger import logger

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

RAW_PATH = params["data_paths"]["raw_data"]          # data/raw/train.csv
SOURCE_DATASET = "notebooks/data.csv"               # your source file

def ingest_data(source_csv: str = SOURCE_DATASET, output_csv: str = RAW_PATH) -> pd.DataFrame:
    """
    Ingests raw data from source_csv and writes to output_csv.
    Returns the DataFrame.
    """
    if not os.path.exists(source_csv):
        logger.error("Source CSV not found: %s", source_csv)
        raise FileNotFoundError(source_csv)

    logger.info("Reading CSV from %s", source_csv)
    df = pd.read_csv(source_csv)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Saved ingested CSV to %s", output_csv)

    return df

if __name__ == "__main__":
    ingest_data()
