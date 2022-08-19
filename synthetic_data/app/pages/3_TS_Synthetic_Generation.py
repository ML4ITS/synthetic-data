from typing import Any

import streamlit as st

from synthetic_data.common import api
from synthetic_data.common.config import LocalConfig
from synthetic_data.common.vizu import vizualize_prediction

cfg = LocalConfig()


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
    container.header("Synthetic Generation")

    with st.sidebar:

        st.header("Configuration")

        model_names = get_model_names(container)
        model_name = st.selectbox(
            label="Model",
            options=model_names,
            help="The trained machine learning models to use.",
        )

        model_versions = get_model_versions(container, model_name)
        model_version = st.selectbox(
            label="Version",
            options=model_versions,
            help="The version of the trained machine learning model",
        )

        payload_type = st.selectbox(
            label="Payload type",
            options=PAYLOAD_TYPES,
            help="The payload type determines the type of the model. \
                A 'conditional' type is meant for conditional \
                GAN models that can generate data \
                based on a condition (e.g. frequency). A 'generation' \
                type is meant for normal GAN models \
                that can generate 'some' data.",
        )

        prediction_arguments = {}
        prediction_arguments["payload_type"] = payload_type
        prediction_arguments["model_name"] = model_name
        prediction_arguments["model_version"] = model_version

        if payload_type == "conditional generation":
            z_dim = st.number_input(
                label="Latent dimention",
                value=100,
                help="The latent-space dimention of the model. \
                    Note that this should be the exact same as the selected model was trained on.",
            )
            n_classes = st.number_input(
                label="Number of conditions",
                value=10,
                help="The number of condition/classes/labels to use when predicting.\
                    For example, value of 3 means that the model can predict 3 different conditions \
                    (e.g. 1 Hz, 2 Hz and 3 Hz)",
            )
            prediction_arguments["z_dim"] = z_dim
            prediction_arguments["n_classes"] = n_classes

        if payload_type == "generation":
            z_dim = st.number_input(
                label="Latent dimention",
                value=100,
                help="The latent-space dimention of the model. \
                    Note that this should be the exact same as the selected model was trained on.",
            )
            n_samples = st.number_input(
                label="Number of samples",
                value=10,
                help="The number of samples to generate by the selected model",
            )

            prediction_arguments["z_dim"] = z_dim
            prediction_arguments["n_samples"] = n_samples

        if payload_type == "forecast":
            raise NotImplementedError("Forecast prediction is not implemented yet.")

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

        MLFLOW_REGISTRY = "Visit ML Flow Registry"
        MLFLOW_URI_TO_MODELS = cfg.URI_MODELREG_REMOTE + "/#/models"
        MLFLOW_MARKDOWN = f"[{MLFLOW_REGISTRY}]({MLFLOW_URI_TO_MODELS})"
        st.markdown(MLFLOW_MARKDOWN, unsafe_allow_html=True)


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
