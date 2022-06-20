from datetime import datetime
from typing import List
import numpy as np

import pandas as pd

from db.constants import Database, DatabaseCollection
from db.session import init_connection

db_client = init_connection()


def get_datasets(limit: int = 100) -> List[dict]:
    db = db_client[Database.NAME.value]
    collection = db[DatabaseCollection.DATASETS.value]
    items = collection.find(limit=limit)
    return list(items)


def save_time_series(doc_name: str, time_series: pd.DataFrame, parameters: dict) -> int:
    document = df_to_document(doc_name=doc_name, df=time_series, parameters=parameters)
    db = db_client[Database.NAME.value]
    collection = db[DatabaseCollection.DATASETS.value]
    document_id = collection.insert_one(document)
    return document_id


def load_time_series_document(doc_name: str):
    db = db_client[Database.NAME.value]
    collection = db[DatabaseCollection.DATASETS.value]
    document = collection.find_one({"name": doc_name})
    return document


def load_time_series_as_numpy(doc_name: str):
    document = load_time_series_document(doc_name=doc_name)
    dataset = np.array(document["y"])  # only need the y-value (?)
    return dataset.reshape(1, -1)  # returns with shape (1, N)


def df_to_document(doc_name: str, df: pd.DataFrame, parameters: dict) -> dict:
    return {
        "name": doc_name,
        "last_modified": datetime.utcnow(),
        "x": df["x"].tolist(),
        "y": df["y"].tolist(),
        "parameters": parameters,
    }
