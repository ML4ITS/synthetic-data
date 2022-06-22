import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(dotenv_path=Path(".") / ".env")


@st.experimental_singleton
def init_connection():
    HOST = os.getenv("MONGO_HOST")
    PORT = int(os.getenv("MONGO_PORT"))
    USERNAME = os.getenv("MONGO_USERNAME")
    PASSWORD = os.getenv("MONGO_PASSWORD")
    return MongoClient(host=HOST, port=PORT, username=USERNAME, password=PASSWORD)
