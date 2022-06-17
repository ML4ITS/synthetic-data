import streamlit as st
from pymongo import MongoClient


@st.experimental_singleton
def init_connection():
    return MongoClient(**st.secrets["mongo"])
