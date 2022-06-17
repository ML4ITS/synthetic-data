import streamlit as st
from db.database import get_datasets
from utils.utils import plot_timeseries_from_dict


def run() -> None:

    container = st.container()
    container.header("Synthetic Time-Series Database")

    with st.sidebar:
        st.sidebar.header("Dataset of Time-Series")

        limit = st.sidebar.number_input(
            "Max datasets", value=5, min_value=1, max_value=10
        )
        documents = get_datasets(limit=limit)

        for document in documents:
            plot_timeseries_from_dict(container, document)
        if not documents:
            container.info(f"No documents found ...")


if __name__ == "__main__":
    run()
