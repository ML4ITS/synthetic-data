from typing import Any

import streamlit as st

from synthetic_data.common import api
from synthetic_data.common.vizu import vizualize_prediction


@st.cache(ttl=60)
def fetch_model_names() -> Any:
    """Returns the json-encoded content of a response, if any.

    Returns:
        Any: the response content.
    """
    return api.get_registrated_model_names()


@st.cache(ttl=60)
def fetch_model_versions(model_name: str) -> Any:
    """Returns the json-encoded content of a response, if any.

    Args:
        model_name (str): the name of the model.
    Returns:
        Any: the response content.
    """
    return api.get_version_by_model_names(model_name)


def get_model_names(container):
    """Fetches the model names from the server, and returns them.
    Renders warnings in the main container if there is any error.

    Args:
        container (DeltaGenerator): the main widget/canvas.

    Returns:
        list: the names of the registered models.
    """
    model_names = []

    fetched_model_names = fetch_model_names()

    if "error" in fetched_model_names:
        container.warning(f"{fetched_model_names['error']}")
        if "stacktrace" in fetched_model_names:
            container.info(f"{fetched_model_names['stacktrace']}")
    else:
        registered_models = fetched_model_names["models"]
        model_names = [rm["name"] for rm in registered_models]
    return model_names


def get_model_versions(container, model_name: str):
    """Fetches the model versions from the server, and returns them.
    Renders warnings in the main container if there is any error.

    Args:
        container (DeltaGenerator): the main widget/canvas.
        model_name (str): the name of the model.

    Returns:
        list: the versions of the registered model.
    """
    model_versions = []

    fetched_model_versions = fetch_model_versions(model_name)

    if "error" in fetched_model_versions:
        container.warning(f"{fetched_model_versions['error']}")
        if "stacktrace" in fetched_model_versions:
            container.info(f"{fetched_model_versions['stacktrace']}")
    else:
        registered_versions = fetched_model_versions["versions"]
        model_versions = [int(rv["version"]) for rv in registered_versions]
        model_versions = sorted(model_versions, reverse=True)

    return model_versions


def run() -> None:

    PAYLOAD_TYPES = ["conditional generation", "generation", "forecast"]

    container = st.container()
    container.header("Synthetic Operations")

    with st.sidebar:
        st.header("Configuration")

        model_names = get_model_names(container)
        model_name = st.selectbox("Model", model_names)

        model_versions = get_model_versions(container, model_name)
        model_version = st.selectbox("Version", model_versions)

        payload_type = st.selectbox("Payload type", PAYLOAD_TYPES)

        prediction_arguments = {}
        prediction_arguments["payload_type"] = payload_type
        prediction_arguments["model_name"] = model_name
        prediction_arguments["model_version"] = model_version

        if payload_type == "conditional generation":
            z_dim = st.number_input("Latent dimention", value=100)
            n_classes = st.number_input("Number of conditions", value=10)
            prediction_arguments["z_dim"] = z_dim
            prediction_arguments["n_classes"] = n_classes

        if payload_type == "generation":
            z_dim = st.number_input("Latent dimention", value=100)
            n_samples = st.number_input("Number of samples", value=10)

            prediction_arguments["z_dim"] = z_dim
            prediction_arguments["n_samples"] = n_samples

        if payload_type == "forecast":
            raise NotImplementedError("Forecast is not implemented yet.")

        with st.form("Configuration"):

            # Check if user has pressed the button
            submitted = st.form_submit_button("Generate")
            if submitted:
                # Check if button press as a missclick
                if model_name is None or model_version is None:
                    container.warning("Please select a model and a version")
                    return

                # Send inferences request to the API
                prediction = api.get_prediction(prediction_arguments)

                # If anything went wrong, show the error message in main widget/canvas
                if "error" in prediction:
                    container.warning(f"{prediction['error']}")
                    if "stacktrace" in prediction:
                        container.info(f"{prediction['stacktrace']}")
                else:
                    # If everything went well, show the prediction in main widget/canvas
                    vizualize_prediction(container, prediction_arguments, prediction)
            else:
                container.info(f"Ready for request ...")

        st.json(prediction_arguments)


# if payload_type == "forecast":
#     time_steps = st.number_input(
#         "Time-steps", value=1000, min_value=50, max_value=3000, step=50
#     )
#     data = st.number_input(
#         "data", value=0.00, min_value=-1.00, max_value=1.00, step=0.10
#     )
#     prediction_arguments["time_steps"] = time_steps
#     prediction_arguments["data"] = data


if __name__ == "__main__":
    run()
