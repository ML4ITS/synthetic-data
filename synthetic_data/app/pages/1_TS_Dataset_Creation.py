import time
from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import timesynth as ts
from timesynth.noise.base_noise import BaseNoise
from timesynth.noise.gaussian_noise import GaussianNoise
from timesynth.signals.ar import AutoRegressive
from timesynth.signals.car import CAR
from timesynth.signals.gaussian_process import GaussianProcess
from timesynth.signals.narma import NARMA
from timesynth.signals.pseudoperiodic import PseudoPeriodic
from timesynth.signals.sinusoidal import Sinusoidal

from synthetic_data.common import api
from synthetic_data.common.helpers import strtobool
from synthetic_data.common.vizu import plot_timeseries


class GaussianNoise2D(GaussianNoise):
    """Supports gaussian noise with 2D"""

    def sample_vectorized(self, time_vector):
        return np.random.normal(loc=self.mean, scale=self.std, size=time_vector.shape)


class ProcessType(Enum):
    HARMONIC = "Harmonic"
    GAUSSIAN_PROCESS = "GaussianProcess"
    PSEUDO_PERIODIC = "PseudoPeriodic"
    AUTO_REGRESSIVE = "AutoRegressive"
    CAR = "CAR"
    NARMA = "NARMA"


class ProcessKernel(Enum):
    CONSTANT = "Constant"
    EXPONENTIAL = "Exponential"
    SE = "SE"
    RQ = "RQ"
    LINEAR = "Linear"
    MATERN = "Matern"
    PERIODIC = "Periodic"


def get_gaussian_process_signal(**kwargs):
    """Returns the appropriate gaussian process signal depending on the
    selected covariance function.

    Returns:
        GaussianProcess: the Gaussian Process time series sampler
    """
    cov_function = kwargs.get("kernel")  # selected covariance function
    if cov_function == ProcessKernel.CONSTANT.value:
        return GaussianProcess(kernel=cov_function, variance=kwargs.get("variance"))
    if cov_function == ProcessKernel.EXPONENTIAL.value:
        return GaussianProcess(kernel=cov_function, gamma=kwargs.get("gamma"))
    if cov_function == ProcessKernel.SE.value:
        return GaussianProcess(kernel=cov_function)
    if cov_function == ProcessKernel.RQ.value:
        return GaussianProcess(kernel=cov_function, alpha=kwargs.get("alpha"))
    if cov_function == ProcessKernel.LINEAR.value:
        return GaussianProcess(
            kernel=cov_function, c=kwargs.get("c"), offset=kwargs.get("offset")
        )
    if cov_function == ProcessKernel.MATERN.value:
        return GaussianProcess(kernel=cov_function, nu=kwargs.get("nu"))
    if cov_function == ProcessKernel.PERIODIC.value:
        return GaussianProcess(kernel=cov_function, p=kwargs.get("period"))


def get_time_samples(
    seq_length: int, keep_percentage: int, is_irregular: bool
) -> np.ndarray:
    """Returns the time samples for the given sequence length and keep percentage.

    Args:
        seq_length (int): the sequence length
        keep_percentage (int): the percentage of samples to keep
        is_irregular (bool): whether the time samples are irregular or not

    Returns:
        np.ndarray: the time samples
    """
    time_sampler = ts.TimeSampler(stop_time=1)  # default 1?
    if is_irregular:
        return time_sampler.sample_irregular_time(
            seq_length=seq_length, keep_percentage=keep_percentage
        )
    return time_sampler.sample_regular_time(num_points=seq_length)


def generate_data(
    process_type: str,
    batch_size: int,
    seq_length: int,
    keep_percentage: int,
    irregular: bool,
    std_noise: float,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generates the time series given all of its parameters.

    Args:
        process_type (str): the type of process to generate
        batch_size (int): how many time series to generate
        seq_length (int): the sequence length of each time series
        keep_percentage (int): the percentage of samples to keep
        irregular (bool): whether the time samples are irregular or not
        std_noise (float): the standard deviation of the noise

    Returns:
        (np.ndarray, np.ndarray, dict): the time series, the time samples, and the parameters
    """

    signal = None

    # default
    default_parameters = {
        "process_type": process_type,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "keep_percentage": keep_percentage,
        "irregular": irregular,
        "std_noise": std_noise,
    }
    # extra
    parameters = {**default_parameters, **kwargs}

    if process_type == ProcessType.HARMONIC.value:
        signal = Sinusoidal(
            amplitude=kwargs.get("amplitude"), frequency=kwargs.get("frequency")
        )

    if process_type == ProcessType.GAUSSIAN_PROCESS.value:
        signal = get_gaussian_process_signal(**kwargs)

    if process_type == ProcessType.PSEUDO_PERIODIC.value:
        signal = PseudoPeriodic(
            amplitude=kwargs.get("amplitude"),
            frequency=kwargs.get("frequency"),
            ampSD=kwargs.get("ampSD"),
            freqSD=kwargs.get("freqSD"),
        )

    if process_type == ProcessType.AUTO_REGRESSIVE.value:
        signal = AutoRegressive(ar_param=kwargs.get("ar_param"), sigma=std_noise)

    if process_type == ProcessType.CAR.value:
        signal = CAR(ar_param=kwargs.get("ar_param"), sigma=std_noise)

    if process_type == ProcessType.NARMA.value:
        signal = NARMA(order=kwargs.get("order"))

    time_samples = get_time_samples(seq_length, keep_percentage, irregular)
    if batch_size > 1 and process_type == ProcessType.HARMONIC.value:
        # reconstruct the time_samples array by repeating time_samples,
        # the number of times given by the batch size.
        # e.g. (20,) => (batch_size, 20)
        time_samples = np.tile(time_samples, (batch_size, 1))

    noise = GaussianNoise2D(std=std_noise)
    timeseries = ts.TimeSeries(signal, noise)
    samples, _, _ = timeseries.sample(time_samples)
    return time_samples, samples, parameters


def run() -> None:
    container = st.container()
    container.header("Create your time-series dataset")

    with st.sidebar:
        st.sidebar.header("Configuration")

        process_types = [key.value for key in ProcessType]
        process_type = st.selectbox("Process type", process_types)

        seq_length = st.number_input("Sequence length", 0, 5000, 100, 10)

        # NOTE: Generating Time-Series with more then 1 sequence is
        # only currently supported for HARMONIC
        # I've not tested other process types yet, therefore disabled on purpose
        batch_size = st.number_input(
            "Number of sequences",
            0,
            100_000,
            1,
            100,
            disabled=process_type != ProcessType.HARMONIC.value,
        )

        irregular = strtobool(st.radio("Irregular", ("False", "True"), horizontal=True))
        keep_percentage = st.slider(
            "Keep",
            1,
            100,
            100,
            5,
            format="%d%%",
            help="Percentage of points to be retained in the irregular series",
            disabled=not irregular,
        )

        std_noise = st.slider(
            "Noise stdv",
            0.0,
            1.0,
            0.3,
            0.01,
            help="Standard deviation of the white noise",
        )
        time_samples, samples, default_parameters = [], [], {}

        if process_type == ProcessType.HARMONIC.value:
            amplitude = st.slider(
                "Amplitude",
                1.0,
                10.0,
                1.0,
                0.1,
                help="Amplitude of the harmonic series",
            )
            frequency = st.slider(
                "Frequency",
                1,
                100,
                1,
                1,
                help="Frequency of the harmonic series",
            )

            time_samples, samples, default_parameters = generate_data(
                process_type=process_type,
                batch_size=batch_size,
                seq_length=seq_length,
                keep_percentage=keep_percentage,
                irregular=irregular,
                std_noise=std_noise,
                frequency=frequency,
                amplitude=amplitude,
            )
            plot_timeseries(container, time_samples, samples)

        if process_type == ProcessType.GAUSSIAN_PROCESS.value:

            kernel_types = [key.value for key in ProcessKernel]
            kernel = st.radio("Kernel", kernel_types)

            if kernel == ProcessKernel.SE.value:
                time_samples, samples, default_parameters = generate_data(
                    process_type=process_type,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    keep_percentage=keep_percentage,
                    irregular=irregular,
                    std_noise=std_noise,
                    kernel=kernel,
                )
                plot_timeseries(container, time_samples, samples)

            if kernel == ProcessKernel.CONSTANT.value:
                variance = st.slider("variance", 0.0, 1.0, 1.0, 0.1)
                time_samples, samples, default_parameters = generate_data(
                    process_type=process_type,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    keep_percentage=keep_percentage,
                    irregular=irregular,
                    std_noise=std_noise,
                    kernel=kernel,
                    variance=variance,
                )
                plot_timeseries(container, time_samples, samples)

            if kernel == ProcessKernel.EXPONENTIAL.value:
                gamma = st.slider("gamma", 0.0, 1.0, 1.0, 0.1)
                time_samples, samples, default_parameters = generate_data(
                    process_type=process_type,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    keep_percentage=keep_percentage,
                    irregular=irregular,
                    std_noise=std_noise,
                    kernel=kernel,
                    gamma=gamma,
                )
                plot_timeseries(container, time_samples, samples)

            if kernel == ProcessKernel.RQ.value:
                alpha = st.slider("alpha", 0.0, 1.0, 1.0, 0.1)
                time_samples, samples, default_parameters = generate_data(
                    process_type=process_type,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    keep_percentage=keep_percentage,
                    irregular=irregular,
                    std_noise=std_noise,
                    kernel=kernel,
                    alpha=alpha,
                )
                plot_timeseries(container, time_samples, samples)

            if kernel == ProcessKernel.LINEAR.value:
                c = st.slider("c", 0.0, 1.0, 1.0, 0.1)
                offset = st.slider("offset", 0.0, 1.0, 1.0, 0.1)
                time_samples, samples, default_parameters = generate_data(
                    process_type=process_type,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    keep_percentage=keep_percentage,
                    irregular=irregular,
                    std_noise=std_noise,
                    kernel=kernel,
                    c=c,
                    offset=offset,
                )
                plot_timeseries(container, time_samples, samples)

            if kernel == ProcessKernel.MATERN.value:
                nu = st.slider("nu", 0.0, 1.0, 1.0, 0.1)
                time_samples, samples, default_parameters = generate_data(
                    process_type=process_type,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    keep_percentage=keep_percentage,
                    irregular=irregular,
                    std_noise=std_noise,
                    kernel=kernel,
                    nu=nu,
                )
                plot_timeseries(container, time_samples, samples)

            if kernel == ProcessKernel.PERIODIC.value:
                period = st.slider("period", 0.0, 1.0, 1.0, 0.1)
                time_samples, samples, default_parameters = generate_data(
                    process_type=process_type,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    keep_percentage=keep_percentage,
                    irregular=irregular,
                    std_noise=std_noise,
                    kernel=kernel,
                    period=period,
                )
                plot_timeseries(container, time_samples, samples)

        if process_type == ProcessType.PSEUDO_PERIODIC.value:
            amplitude = st.slider(
                "Amplitude",
                0.0,
                10.0,
                1.0,
                0.1,
                help="Amplitude of the harmonic series",
            )
            ampSD = st.slider(
                "Amplitude stdv",
                0.0,
                1.0,
                0.1,
                0.01,
                help="Amplitude standard deviation",
            )
            frequency = st.slider(
                "Frequency",
                0.0,
                100.0,
                1.0,
                0.5,
                help="Frequency of the harmonic series",
            )
            freqSD = st.slider(
                "Frequency stdv",
                0.0,
                1.0,
                0.1,
                0.01,
                help="Frequency standard deviation",
            )

            time_samples, samples, default_parameters = generate_data(
                process_type=process_type,
                batch_size=batch_size,
                seq_length=seq_length,
                keep_percentage=keep_percentage,
                irregular=irregular,
                std_noise=std_noise,
                amplitude=amplitude,
                ampSD=ampSD,
                frequency=frequency,
                freqSD=freqSD,
            )
            plot_timeseries(container, time_samples, samples)

        if process_type == ProcessType.AUTO_REGRESSIVE.value:
            use_ar2 = st.checkbox("Use AR2, else AR1")
            phi_1 = st.slider(f"phi_1", 0.0, 2.0, 1.0, 0.1)
            phi_2 = st.slider(f"phi_2", 0.0, 2.0, 1.0, 0.1, disabled=not use_ar2)

            if use_ar2:
                ar_param = [phi_1, phi_2]
            else:
                ar_param = [phi_1]

            time_samples, samples, default_parameters = generate_data(
                process_type=process_type,
                batch_size=batch_size,
                seq_length=seq_length,
                keep_percentage=100,  # required
                irregular=False,  # required
                std_noise=std_noise,
                ar_param=ar_param,
            )
            plot_timeseries(container, time_samples, samples)

        if process_type == ProcessType.CAR.value:
            ar_param = st.slider(
                f"ar_param", 0.0, 2.0, 1.0, 0.1, help="Parameter of the AR(1) process"
            )
            time_samples, samples, default_parameters = generate_data(
                process_type=process_type,
                batch_size=batch_size,
                seq_length=seq_length,
                keep_percentage=100,  # required
                irregular=False,  # required
                std_noise=std_noise,
                ar_param=ar_param,
            )
            plot_timeseries(container, time_samples, samples)

        if process_type == ProcessType.NARMA.value:
            order = st.slider("order", 1, 10, 3, 1, help="Order of the NARMA process")
            time_samples, samples, default_parameters = generate_data(
                process_type=process_type,
                batch_size=batch_size,
                seq_length=seq_length,
                keep_percentage=100,  # required
                irregular=False,  # required
                std_noise=std_noise,
                order=order,
            )
            plot_timeseries(container, time_samples, samples)

        def form_callback():
            """On submit, fetch the input parameters,
            and save its corresponding time-series to the database. On success,
            reset the form, else show an error message.
            """
            name = st.session_state.database_name

            if name == "":
                return
            response = api.save_time_series(name, samples, default_parameters)

            if "error" in response:
                st.error(response["error"])
                if "stacktrace" in response:
                    st.warning(response["stacktrace"])
            else:
                st.session_state.database_name = ""

        with st.form("MongoDB Form"):
            st.header("MongoDB")
            label = "Save dataset to MongoDB"
            _ = st.text_input("Save with name:", key="database_name")
            _ = st.form_submit_button(label, on_click=form_callback)

        # padding
        st.write("--------------------")


if __name__ == "__main__":
    run()
