from datetime import datetime
from typing import List
import pandas as pd

import pymongo
import streamlit as st
from streamlit import secrets
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collation import Collation
from db.session import init_connection

DB_NAME = "db_timeseries"

db_client = init_connection()


@st.experimental_memo(ttl=600)  # only rerun when the query changes or after 10 min
def get_data_from_collection(collection_name: str, limit: int = 100) -> List[dict]:
    db = db_client[DB_NAME]
    collection = db[collection_name]
    items = collection.find(limit=limit)
    return list(items)  # make hashable for st.experimental_memo


@st.experimental_memo(ttl=600)  # only rerun when the query changes or after 10 min
def get_collections() -> List[str]:
    db = db_client[DB_NAME]
    collections = db.list_collection_names()
    return collections


@st.experimental_memo(ttl=600)  # only rerun when the query changes or after 10 min
def get_collection(col_name: str) -> List[str]:
    db = db_client[DB_NAME]
    collection = db[col_name]
    return collection


def save_time_series(col_name: str, doc_name: str, time_series: pd.DataFrame) -> int:
    document = df_to_document(doc_name=doc_name, df=time_series)
    db = db_client[DB_NAME]
    collection = db[col_name]
    document_id = collection.insert_one(document)
    return document_id


def df_to_document(doc_name: str, df: pd.DataFrame) -> dict:
    # TimeSeriesDocument TODO: add more fields?
    return {
        "name": doc_name,
        "last_modified": datetime.utcnow(),
        "x": df["x"].tolist(),
        "y": df["y"].tolist(),
    }
