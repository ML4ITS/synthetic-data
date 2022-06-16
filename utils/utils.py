import numpy as np
import pandas as pd
from streamlit.delta_generator import DeltaGenerator
from bokeh.plotting import figure


def plot_timeseries(
    title: str, container: DeltaGenerator, time_samples: np.ndarray, samples: np.ndarray
) -> None:
    if not len(time_samples) > 0:
        return

    fig = figure(
        title=title,
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


def plot_timeseries_from_dict(container: DeltaGenerator, time_series_dict: dict):
    id = time_series_dict["_id"]
    name = time_series_dict["name"]
    last_modified = time_series_dict["last_modified"]
    time_samples = time_series_dict["x"]
    samples = time_series_dict["y"]
    container.subheader(f"Time-Series Document")
    container.write(f"  ID: {id}")
    container.write(f"Name: {name}")
    container.write(f"From: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
    plot_timeseries(name, container, time_samples, samples)


def df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def prettify_name(name: str) -> str:
    return name.replace("_", " ").title()


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"Invalid truth value '{val}'")
