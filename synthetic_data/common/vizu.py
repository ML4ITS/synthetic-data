import os

import numpy as np
import pandas as pd
import torch
from bokeh.plotting import figure
from matplotlib import pyplot as plt


def vizualize_and_save_prediction(
    outdir: str,
    predictions: np.ndarray,
    n_samples: torch.Tensor,
    future: int,
    epoch: int,
) -> None:
    plt.figure(figsize=(30, 10), dpi=100)
    plt.title(f"Epoch {epoch}", fontsize=40)
    plt.xlabel("x", fontsize=30)
    plt.ylabel("y", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    random_sample = predictions[0]

    # Actual time-series
    plt.plot(
        np.arange(n_samples),
        random_sample[:n_samples],
        "b",
        linewidth=3,
    )
    # Forecasted time-series
    plt.plot(
        np.arange(n_samples, n_samples + future),
        random_sample[n_samples:],
        "b:",
        linewidth=3,
    )
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/epoch{epoch}.png")
    plt.close()


def vizualize_and_view_prediction(
    predictions: np.ndarray,
    n_samples: torch.Tensor,
    future: int,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.title(f"Prediction", fontsize=16)

    random_sample = predictions[0]

    # Actual time-series
    plt.plot(
        np.arange(n_samples),
        random_sample[:n_samples],
        "b",
        linewidth=3,
    )
    # Forecasted time-series
    plt.plot(
        np.arange(n_samples, n_samples + future),
        random_sample[n_samples:],
        "b:",
        linewidth=3,
    )
    plt.show()


def vizualize_dataset(data: np.ndarray) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    ax[0].plot(data.reshape(-1))
    ax[0].set_title("Dataset preview", fontsize=16)
    ax[1].plot(data[0])
    ax[1].set_title("Batch preview", fontsize=16)
    plt.show()


def plot_timeseries(container, time_samples: np.ndarray, samples: np.ndarray) -> None:
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


def preview_dataset(container, dataset: dict) -> None:
    container.dataframe(data=dataset_to_dataframe(dataset=dataset))
    plot_timeseries(container, dataset["x"], dataset["y"])


def dataset_to_dataframe(dataset: dict) -> pd.DataFrame:
    info = pd.DataFrame(
        {
            "id": [str(dataset["_id"]["$oid"])],
            "name": dataset["name"],
        }
    )
    params = pd.DataFrame([dataset["parameters"]])
    meta = pd.concat([info, params], axis=1)
    meta.set_index("id", inplace=True)
    return meta
