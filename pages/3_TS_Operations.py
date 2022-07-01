import os
from bokeh.plotting import figure
import requests
import streamlit as st
import requests


FLASK_HOST = os.getenv("FLASK_HOST")
FLASK_PORT = os.getenv("FLASK_PORT")
FLASK_API = f"http://{FLASK_HOST}:{FLASK_PORT}"


def get_model_meta(base_url):
    r = requests.get(f"{base_url}/models")
    return r.json()


def get_forecast_meta(base_url, model_name, model_version, timesteps, data):
    url = f"{base_url}/forecast"
    payload = {
        "model_name": model_name,
        "model_version": model_version,
        "timesteps": timesteps,
        "data": data,
    }
    r = requests.post(url, data=payload)
    return r.json()


def get_names_from_model_meta(meta):
    return [m["name"] for m in meta]


def get_version_from_model_meta(meta):
    return [m["version"] for m in meta]


def plot_forecast_meta(container, meta) -> None:
    if not meta:
        return

    x1 = meta["x1"]
    y1 = meta["y1"]
    x2 = meta["x2"]
    y2 = meta["y2"]

    fig = figure(
        x_axis_label="x",
        y_axis_label="y",
        max_height=300,
        height_policy="max",
    )

    fig.line(x1, y1, legend_label="Regular", line_width=2)
    fig.circle(
        x2,
        y2,
        legend_label="Regular",
        line_width=2,
        fill_color="blue",
        size=5,
    )
    container.bokeh_chart(fig, use_container_width=True)


def run() -> None:

    container = st.container()
    container.header("Forecast Time-Series")

    with st.sidebar:

        with st.form("Configuration"):
            st.header("Configuration")

            model_meta = get_model_meta(FLASK_API)
            models = get_names_from_model_meta(model_meta)
            versions = get_version_from_model_meta(model_meta)

            model_name = st.selectbox("Model", models)
            model_version = st.selectbox("Version", versions)
            time_steps = st.number_input(
                "Time-steps", value=1000, min_value=50, max_value=3000, step=50
            )
            data = st.number_input(
                "data", value=0.00, min_value=-1.00, max_value=1.00, step=0.10
            )
            submitted = st.form_submit_button("Generate new forecast")
            if submitted:
                meta = get_forecast_meta(
                    FLASK_API, model_name, model_version, time_steps, data
                )
                plot_forecast_meta(container, meta)
            else:
                container.info(f"No forecast generated ...")


if __name__ == "__main__":
    run()
