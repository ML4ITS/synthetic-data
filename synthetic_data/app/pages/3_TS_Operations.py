import streamlit as st

from synthetic_data.common import api
from synthetic_data.common.vizu import vizualize_prediction


def get_model_names(container):
    model_names = []

    fetched_model_names = api.get_registrated_model_names()
    if "error" in fetched_model_names:
        container.warning(f"{fetched_model_names['error']}")
        if "stacktrace" in fetched_model_names:
            container.info(f"{fetched_model_names['stacktrace']}")
    else:
        registered_models = fetched_model_names["models"]
        model_names = [rm["name"] for rm in registered_models]
    return model_names


def get_model_versions(container, model_name: str):
    model_versions = []

    fetched_model_versions = api.get_version_by_model_names(model_name)

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

        container.json(prediction_arguments)

        if payload_type == "conditional generation":
            z_dim = st.number_input("z_dim", value=100)
            n_classes = st.number_input("n_classes", value=10)
            prediction_arguments["z_dim"] = z_dim
            prediction_arguments["n_classes"] = n_classes

        if payload_type == "generation":
            z_dim = st.number_input("z_dim", value=100)
            prediction_arguments["z_dim"] = z_dim

        with st.form("Configuration"):

            # -------------------------------------------------------------------
            submitted = st.form_submit_button("Generate")
            if submitted:
                if model_name is None or model_version is None:
                    container.warning("Please select a model and a version")
                    return

                container.text("Generating...")
                container.json(prediction_arguments)
                # prediction = api.get_prediction(prediction_arguments)

                # # -----------------
                # if "error" in prediction:
                #     container.warning(f"{prediction['error']}")
                #     if "stacktrace" in prediction:
                #         container.info(f"{prediction['stacktrace']}")
                # else:
                #     vizualize_prediction(container, prediction_arguments, prediction)
            else:
                container.info(f"Ready for request ...")


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
