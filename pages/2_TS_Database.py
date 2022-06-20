import streamlit as st
from db.database import get_datasets
from utils.vizu import preview_dataset


def run() -> None:

    container = st.container()
    container.header("Synthetic Time-Series Database")

    with st.sidebar:
        st.sidebar.header("Dataset of Time-Series")

        limit = st.sidebar.number_input(
            "Show max datasets", value=5, min_value=1, max_value=10
        )

        datasets = get_datasets(limit=limit)
        for dataset in datasets:
            preview_dataset(container, dataset)
        if not datasets:
            container.info(f"No dataset found ...")


if __name__ == "__main__":
    run()
