import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def run():
    st.set_page_config(page_title="STS", page_icon="🌀", layout="wide")
    st.write("# Synthetic Time-Series")
    st.write("This page intentionally left blank. Ideas?")


if __name__ == "__main__":
    run()
