from datetime import datetime
from typing import List

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


def df_to_document(doc_name: str, df: pd.DataFrame, parameters: dict) -> dict:
    return {
        "name": doc_name,
        "last_modified": datetime.utcnow(),
        "x": df["x"].tolist(),
        "y": df["y"].tolist(),
        "parameters": parameters,
    }
