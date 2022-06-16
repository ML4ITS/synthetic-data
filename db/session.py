from pymongo import MongoClient
import streamlit as st


@st.experimental_singleton
def init_connection():
    return MongoClient(**st.secrets["mongo"])
