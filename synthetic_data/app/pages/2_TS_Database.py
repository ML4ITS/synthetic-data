import streamlit as st

from synthetic_data.common import api
from synthetic_data.common.vizu import preview_dataset


def run() -> None:

    container = st.container()
    container.header("Saved Time-Series")

    with st.sidebar:
        st.sidebar.header("Dataset of Time-Series")

        limit = st.sidebar.number_input(
            "Show max datasets", value=5, min_value=1, max_value=10
        )
        response = api.get_all_time_series(limit=limit)
        if "error" in response:
            container.warning(f"{response['error']}")
            if "stacktrace" in response:
                container.info(f"{response['stacktrace']}")
        else:
            for dataset in response["datasets"]:
                preview_dataset(container, dataset)


if __name__ == "__main__":
    run()
