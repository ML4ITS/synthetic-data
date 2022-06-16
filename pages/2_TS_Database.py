from datetime import datetime
from typing import List
import streamlit as st
import pandas as pd
from streamlit import secrets
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collation import Collation
from utils.utils import prettify_name, plot_timeseries_from_dict
from db.database import get_collections, get_data_from_collection


def run() -> None:

    container = st.container()
    container.header("Synthetic Time-Series Database")

    with st.sidebar:
        st.sidebar.header("Database")

        collection_names = get_collections()
        collection_prettynames = [
            prettify_name(colname) for colname in get_collections()
        ]
        collection_map = {
            key: value for (key, value) in zip(collection_prettynames, collection_names)
        }
        collection_name = st.selectbox("Select collection", collection_prettynames)
        collection_name = collection_map[collection_name]
        limit = st.number_input("Max documents", value=2, min_value=1, max_value=100)

        documents = get_data_from_collection(
            collection_name=collection_name, limit=limit
        )

        for document in documents:
            plot_timeseries_from_dict(container, document)

        # refresh database # TODO: FIX THIS
        if st.button("Refetch"):
            st.experimental_rerun()


if __name__ == "__main__":
    run()
