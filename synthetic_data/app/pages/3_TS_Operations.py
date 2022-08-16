import streamlit as st

from synthetic_data.common import api
from synthetic_data.common.vizu import vizualize_prediction


@st.cache
def fetch_models():
    return api.get_registrated_models()


def run() -> None:

    container = st.container()
    container.header("Synthetic Operations")

    with st.sidebar:

        with st.form("Configuration"):
            st.header("Configuration")

            model_names = []
            model_versions = []
            registrated_models = fetch_models()

            if "error" in registrated_models:
                container.warning(f"{registrated_models['error']}")
                if "stacktrace" in registrated_models:
                    container.info(f"{registrated_models['stacktrace']}")
            else:
                models = registrated_models["models"]
                model_names = [m["name"] for m in models]
                model_versions = [m["version"] for m in models]

            model_name = st.selectbox("Model", model_names)
            model_version = st.selectbox("Version", model_versions)

            PAYLOAD_TYPES = ["conditional generation", "generation", "forecast"]
            payload_type = st.selectbox("Payload type", PAYLOAD_TYPES)

            prediction_arguments = {}
            prediction_arguments["payload_type"] = payload_type
            prediction_arguments["model_name"] = model_name
            prediction_arguments["model_version"] = model_version

            # if payload_type == "forecast":
            #     time_steps = st.number_input(
            #         "Time-steps", value=1000, min_value=50, max_value=3000, step=50
            #     )
            #     data = st.number_input(
            #         "data", value=0.00, min_value=-1.00, max_value=1.00, step=0.10
            #     )
            #     prediction_arguments["time_steps"] = time_steps
            #     prediction_arguments["data"] = data
            if payload_type == "conditional generation":
                z_dim = st.number_input("z_dim", value=100)
                n_classes = st.number_input("n_classes", value=10)

                prediction_arguments["z_dim"] = z_dim
                prediction_arguments["n_classes"] = n_classes
            # if payload_type == "generation":
            #     z_dim = st.number_input("z_dim", value=100)
            #     prediction_arguments["z_dim"] = z_dim

            # -------------------------------------------------------------------
            submitted = st.form_submit_button("Generate")
            if submitted:
                if model_name is None or model_version is None:
                    container.warning("Please select a model and a version")
                    return

                prediction = api.get_prediction(prediction_arguments)

                # -----------------
                if "error" in prediction:
                    container.warning(f"{prediction['error']}")
                    if "stacktrace" in prediction:
                        container.info(f"{prediction['stacktrace']}")
                else:
                    vizualize_prediction(container, prediction_arguments, prediction)
            else:
                container.info(f"Ready for request ...")


if __name__ == "__main__":
    run()
