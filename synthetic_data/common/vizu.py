import os

import numpy as np
import pandas as pd
import torch
from bokeh.plotting import figure
from matplotlib import pyplot as plt

RED_COLOR = "#DE5D4F"


def vizualize_and_save_prediction(
    outdir: str,
    predictions: np.ndarray,
    n_samples: int,
    future: int,
    epoch: int,
) -> None:
    plt.figure(figsize=(30, 10), dpi=100)
    plt.title(f"Epoch {epoch}", fontsize=40)
    plt.xlabel("Time steps", fontsize=30)
    plt.ylabel("Amplitude", fontsize=30)
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


def plot_timeseries(container, time_samples: np.ndarray, samples: np.ndarray) -> None:
    # preview only the first sequence of the dataset,
    # regardless if its just one or multiple sequences
    if time_samples.ndim == 2 and samples.ndim == 2:
        preview_sample = samples[0]
        preview_steps = time_samples[0]
    else:
        preview_sample = samples
        preview_steps = time_samples

    fig = figure(
        x_axis_label="Time steps",
        y_axis_label="Amplitude",
        max_height=300,
        height_policy="max",
    )

    fig.line(
        preview_steps,
        preview_sample,
        legend_label="Regular",
        line_width=2,
        line_color=RED_COLOR,
    )
    fig.circle(
        preview_steps,
        preview_sample,
        legend_label="Regular",
        line_width=2,
        color=RED_COLOR,
        fill_color=RED_COLOR,
        size=5,
    )
    container.bokeh_chart(fig, use_container_width=True)


def vizualize_prediction(
    container, prediction_arguments: dict, prediction: dict
) -> None:
    """Render prediction to the container, based on the provided arguments.

    Args:
        container (DeltaGenerator): the container to render the prediction to.
        prediction_arguments (dict): the user arguments for the prediction.
        prediction (dict): the prediction response from the API.

    Raises:
        NotImplementedError: if the prediction_arguments["type"] is not supported.
    """
    payload_type = prediction_arguments["payload_type"]
    sequences = np.array(prediction["response"])

    if payload_type == "forecast":
        raise NotImplementedError
    elif payload_type in ["conditional generation", "generation"]:
        plot_generic_generation_prediction(container, sequences)
    else:
        raise ValueError(f"Unsupported payload_type: {payload_type}")


def plot_generic_generation_prediction(container, sequences: np.ndarray) -> None:
    """Render sequences generated to the container

    Args:
        container (DeltaGenerator): the container to render the prediction to.
        sequences (np.ndarray): the prediction response from the API
    """
    n_samples = sequences.shape[1]
    time_steps = np.arange(n_samples)

    for i, sequence in enumerate(sequences):
        fig = figure(
            x_axis_label="Time steps",
            y_axis_label="Amplitude",
            max_height=300,
            height_policy="max",
            title=f"Sequence {i+1}",
        )

        fig.line(
            time_steps,
            sequence,
            legend_label=f"Sequence: {i+1}",
            line_width=2,
            line_color=RED_COLOR,
        )
        fig.circle(
            time_steps,
            sequence,
            legend_label=f"Sequence: {i+1}",
            line_width=2,
            color=RED_COLOR,
            fill_color=RED_COLOR,
            size=5,
        )
        container.bokeh_chart(fig, use_container_width=True)


def plot_forecast_meta(container, meta) -> None:
    raise NotImplementedError

    # if not meta:
    #     return

    # x1 = meta["x1"]
    # y1 = meta["y1"]
    # x2 = meta["x2"]
    # y2 = meta["y2"]

    # fig = figure(
    #     x_axis_label="Time steps",
    #     y_axis_label="Amplitude",
    #     max_height=300,
    #     height_policy="max",
    # )

    # fig.line(x1, y1, legend_label="Regular", line_width=2, line_color=RED_COLOR)
    # fig.circle(
    #     x2,
    #     y2,
    #     legend_label="Regular",
    #     line_width=2,
    #     color=RED_COLOR,
    #     fill_color=RED_COLOR,
    #     size=5,
    # )
    # container.bokeh_chart(fig, use_container_width=True)


def preview_dataset(container, dataset: dict) -> None:
    raise DeprecationWarning(
        "preview_dataset is deprecated, use preview_dataset_by_sample"
    )

    # prewview only a sequence of the dataset
    container.dataframe(data=dataset_to_dataframe(dataset=dataset))
    data = np.array(dataset["data"])
    # preview only the first sequence of the dataset,
    # regardless if its just one or multiple sequences
    if data.ndim == 1:
        preview_sample = data
        preview_steps = np.arange(len(data))
    elif data.ndim == 2:
        preview_sample = data[0]
        preview_steps = np.arange(len(preview_sample))
    else:
        raise ValueError(f"Invalid data dimension {data.ndim}")
    plot_timeseries(container, preview_steps, preview_sample)


def preview_dataset_by_sample(container, dataset: dict) -> None:
    """Preview a sample from the provided dataset on the container.
    Dataset (dict) shouldn't contain the main "data" field, only the "sample".
    This way the user can preview a sample from the dataset without having to
    encode/decode the whole dataset.

    Args:
        container (DeltaGenerator): the container to render the prediction to.
        dataset (dict): the dataset meta.

    Raises:
        ValueError: when dimensions of the dataset are invalid.
    """

    # prewview only a sequence of the dataset
    container.dataframe(data=dataset_to_dataframe(dataset=dataset))
    data = np.array(dataset["sample"])
    if data.ndim == 1:
        preview_sample = data
        preview_steps = np.arange(len(data))
    else:
        raise ValueError(f"Invalid data dimension {data.ndim}")
    plot_timeseries(container, preview_steps, preview_sample)


def dataset_to_dataframe(dataset: dict) -> pd.DataFrame:
    """Convert the dataset tags/meta to a pandas dataframe.
    Creates a viewable 'id' column for the dataset, as MongoDB returns a
    auto-generated id for each document.

    Args:
        dataset (dict): the dataset meta.

    Returns:
        pd.DataFrame: the dataset meta as a pandas dataframe.
    """
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
