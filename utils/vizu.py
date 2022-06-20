import numpy as np
import pandas as pd
from streamlit.delta_generator import DeltaGenerator
from bokeh.plotting import figure


def plot_timeseries(
    container: DeltaGenerator, time_samples: np.ndarray, samples: np.ndarray
) -> None:
    if not len(time_samples) > 0:
        return

    fig = figure(
        x_axis_label="x",
        y_axis_label="y",
        max_height=300,
        height_policy="max",
    )

    fig.line(time_samples, samples, legend_label="Regular", line_width=2)
    fig.circle(
        time_samples,
        samples,
        legend_label="Regular",
        line_width=2,
        fill_color="blue",
        size=5,
    )
    container.bokeh_chart(fig, use_container_width=True)


def preview_dataset(container: DeltaGenerator, dataset: dict):
    info = pd.DataFrame(
        {
            "id": [str(dataset["_id"])],
            "name": [dataset["name"]],
            "last_modified": [dataset["last_modified"].strftime("%d-%m-%Y %H:%M")],
        }
    )
    params = pd.DataFrame([dataset["parameters"]])
    meta = pd.concat([info, params], axis=1)
    meta.set_index("id", inplace=True)
    container.dataframe(meta)
    plot_timeseries(container, dataset["x"], dataset["y"])
