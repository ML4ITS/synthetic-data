import streamlit as st
from common import api
from common.config import Config
from common.vizu import plot_forecast_meta

cfg = Config()


def run() -> None:

    container = st.container()
    container.header("Forecast Time-Series")

    with st.sidebar:

        with st.form("Configuration"):
            st.header("Configuration")

            model_names = []
            model_versions = []

            response = api.get_model_meta()

            if "error" in response:
                container.warning(f"{response['error']}")
                if "stacktrace" in response:
                    container.info(f"{response['stacktrace']}")
            else:
                models = response["models"]
                model_names = [m["name"] for m in models]
                model_versions = [m["version"] for m in models]

            model_name = st.selectbox("Model", model_names)
            model_version = st.selectbox("Version", model_versions)
            time_steps = st.number_input(
                "Time-steps", value=1000, min_value=50, max_value=3000, step=50
            )
            data = st.number_input(
                "data", value=0.00, min_value=-1.00, max_value=1.00, step=0.10
            )
            submitted = st.form_submit_button("Generate new forecast")
            if submitted:
                if model_name is None or model_version is None:
                    container.warning("Please select a model and a version")
                    return

                response = api.get_forecast_meta(
                    model_name, model_version, time_steps, data
                )
                if "error" in response:
                    container.warning(f"{response['error']}")
                    if "stacktrace" in response:
                        container.info(f"{response['stacktrace']}")
                else:
                    plot_forecast_meta(container, response["forecast"])
            else:
                container.info(f"No forecast generated ...")


if __name__ == "__main__":
    run()
