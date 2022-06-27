import collections
import pickle
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from pymongo.results import InsertOneResult, UpdateResult

from db.constants import Database, DatabaseCollection
from db.session import init_connection

db_client = init_connection()


def get_datasets(limit: int = 100) -> List[dict]:
    db = db_client[Database.NAME.value]
    collection = db[DatabaseCollection.DATASETS.value]
    items = collection.find(limit=limit)
    return list(items)


def save_time_series(
    doc_name: str, time_series: pd.DataFrame, parameters: dict
) -> InsertOneResult:
    document = _create_time_series(doc_name, time_series, parameters)
    db = db_client[Database.NAME.value]
    collection = db[DatabaseCollection.DATASETS.value]
    inserted_document = collection.insert_one(document)
    return inserted_document


def update_model(
    model_name: str, dataset: str, arch: str, data: collections.OrderedDict
) -> UpdateResult:
    document = _create_model(model_name, dataset, arch, data)
    db = db_client[Database.NAME.value]
    collection = db[DatabaseCollection.MODELS.value]
    update_doc = collection.update_one({"name": model_name}, {"$set": document})
    return update_doc


def create_model(
    model_name: str, dataset: str, arch: str, data: collections.OrderedDict
) -> InsertOneResult:
    document = _create_model(model_name, dataset, arch, data)
    db = db_client[Database.NAME.value]
    collection = db[DatabaseCollection.MODELS.value]
    # Create a unique document
    new_document = collection.insert_one(document)
    collection.create_index([("name", 1)], unique=True)
    return new_document


def load_model(model_name: str):
    db = db_client[Database.NAME.value]
    collection = db[DatabaseCollection.MODELS.value]
    document = collection.find_one({"name": model_name})
    if document is not None:
        return {
            "name": document["name"],
            "dataset": document["dataset"],
            "arch": document["arch"],
            "state_dict": pickle.loads(document["state_dict"]),
        }
    return None


def load_time_series(doc_name: str) -> dict:
    db = db_client[Database.NAME.value]
    collection = db[DatabaseCollection.DATASETS.value]
    document = collection.find_one({"name": doc_name})
    if document is None:
        raise FileNotFoundError(f"Dataset '{doc_name}' is not found.")
    return document


def load_time_series_as_numpy(doc_name: str) -> np.ndarray:
    document = load_time_series(doc_name=doc_name)
    dataset = np.array(document["y"])  # only need the y-value (?)
    return dataset.reshape(1, -1)  # returns with shape (1, N)


def _create_time_series(doc_name: str, df: pd.DataFrame, parameters: dict) -> dict:
    return {
        "name": doc_name,
        "x": df["x"].tolist(),
        "y": df["y"].tolist(),
        "parameters": parameters,
        "last_modified": datetime.utcnow(),
    }


def _create_model(
    model_name: str, dataset: str, arch: str, data: collections.OrderedDict
) -> dict:
    return {
        "name": model_name,
        "dataset": dataset,
        "arch": arch,
        "state_dict": pickle.dumps(data),
        "last_modified": datetime.utcnow(),
    }
