import os

# import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# @st.experimental_singleton
def init_connection():
    # TODO: Fix environment variables when testing locally
    HOST = os.environ["MONGO_HOST"]
    PORT = int(os.environ["MONGO_PORT"])
    USERNAME = os.environ["MONGO_USERNAME"]
    PASSWORD = os.environ["MONGO_PASSWORD"]
    return MongoClient(host=HOST, port=PORT, username=USERNAME, password=PASSWORD)
