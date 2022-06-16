from typing import List, Tuple, Union, Dict, Any

import timesynth as ts
import streamlit as st
import pandas as pd
import numpy as np

from streamlit.delta_generator import DeltaGenerator
from timesynth.noise.red_noise import RedNoise
from timesynth.noise.gaussian_noise import GaussianNoise
from timesynth.signals.gaussian_process import GaussianProcess
from timesynth.signals.pseudoperiodic import PseudoPeriodic
from timesynth.signals.sinusoidal import Sinusoidal
from timesynth.signals.ar import AutoRegressive
from timesynth.signals.narma import NARMA
from timesynth.signals.car import CAR

from db.database import save_time_series
from utils.utils import strtobool, plot_timeseries
from enum import Enum


class ProcessType(Enum):
    Harmonic = "Harmonic"
    GaussianProcess = "GaussianProcess"
    PseudoPeriodic = "PseudoPeriodic"
    AutoRegressive = "AutoRegressive"
    CAR = "CAR"
    NARMA = "NARMA"


class ProcessKernel(Enum):
    Constant = "Constant"
    Exponential = "Exponential"
    SE = "SE"
    RQ = "RQ"
    Linear = "Linear"
    Matern = "Matern"
    Periodic = "Periodic"


def get_gaussian_process_signal(**kwargs):
    cov_function = kwargs.get("kernel")  # selected covariance function
    if cov_function == ProcessKernel.Constant.value:
        return GaussianProcess(kernel=cov_function, variance=kwargs.get("variance"))
    if cov_function == ProcessKernel.Exponential.value:
        return GaussianProcess(kernel=cov_function, gamma=kwargs.get("gamma"))
    if cov_function == ProcessKernel.SE.value:
        return GaussianProcess(kernel=cov_function)
    if cov_function == ProcessKernel.RQ.value:
        return GaussianProcess(kernel=cov_function, alpha=kwargs.get("alpha"))
    if cov_function == ProcessKernel.Linear.value:
        return GaussianProcess(
            kernel=cov_function, c=kwargs.get("c"), offset=kwargs.get("offset")
        )
    if cov_function == ProcessKernel.Matern.value:
        return GaussianProcess(kernel=cov_function, nu=kwargs.get("nu"))
    if cov_function == ProcessKernel.Periodic.value:
        return GaussianProcess(kernel=cov_function, p=kwargs.get("period"))


def get_time_samples(
    stop_time: int, num_points: int, keep_percentage: int, is_irregular: bool
) -> np.ndarray:
    time_sampler = ts.TimeSampler(stop_time=stop_time)
    if is_irregular:
        return time_sampler.sample_irregular_time(
            num_points=num_points, keep_percentage=keep_percentage
        )
    return time_sampler.sample_regular_time(num_points=num_points)


def generate_data(
    process_type: str = ProcessType.Harmonic.value,
    stop_time: int = 1,
    num_points: int = 50,
    keep_percentage: int = 50,
    irregular: bool = True,
    std_noise: float = 0.3,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:

    signal = None

    if process_type == ProcessType.Harmonic.value:
        signal = Sinusoidal(frequency=kwargs.get("frequency"))

    if process_type == ProcessType.GaussianProcess.value:
        signal = get_gaussian_process_signal(**kwargs)

    if process_type == ProcessType.PseudoPeriodic.value:
        signal = PseudoPeriodic(
            frequency=kwargs.get("frequency"),
            freqSD=kwargs.get("freqSD"),
            ampSD=kwargs.get("ampSD"),
        )

    if process_type == ProcessType.AutoRegressive.value:
        signal = AutoRegressive(ar_param=kwargs.get("ar_param"), sigma=std_noise)

    if process_type == ProcessType.CAR.value:
        signal = CAR(ar_param=kwargs.get("ar_param"), sigma=std_noise)

    if process_type == ProcessType.NARMA.value:
        signal = NARMA(order=kwargs.get("order"))

    time_samples = get_time_samples(stop_time, num_points, keep_percentage, irregular)
    noise = GaussianNoise(std=std_noise)  # we only use the white noise

    timeseries = ts.TimeSeries(signal, noise_generator=noise)
    samples, _, _ = timeseries.sample(time_samples)
    return time_samples, samples


# Your app goes in the function run()
def run() -> None:
    st.subheader("Synthetic Time-Series Generation")
    container = st.container()

    with st.sidebar:
        st.sidebar.header("Configuration")

        process_types = tuple(ProcessType.__members__.keys())
        process_type = st.selectbox("Process type", process_types)

        num_points = st.slider("Number of points", 0, 2500, 100, 5)
        # num_timeseries = st.slider("Number of TS", 1, 1000, 5, 5)  # TODO: change?
        # MAX_NUM_TIMESERIES = st.slider("Max number of TS to Plot", 1, 10, 1, 1)

        irregular = strtobool(st.radio("Irregular", ("True", "False"), horizontal=True))
        keep_percentage = st.slider(
            "Keep",
            0,
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
        time_samples, samples = [], []

        df = pd.DataFrame(columns=["x", "y"])  # "ID", "x", "y"]
        df_temp = pd.DataFrame(columns=["x", "y"])  # "ID", "x", "y"]

        if process_type == ProcessType.Harmonic.value:
            amplitude = st.slider(
                "Amplitude",
                0.0,
                10.0,
                1.0,
                0.1,
                help="Amplitude of the harmonic series",
            )
            frequency = st.slider(
                "Frequency",
                0.0,
                100.0,
                1.0,
                0.1,
                help="Frequency of the harmonic series",
            )
            # for i in range(num_timeseries):
            time_samples, samples = generate_data(
                process_type=process_type,
                num_points=num_points,
                irregular=irregular,
                keep_percentage=keep_percentage,
                std_noise=std_noise,
                frequency=frequency,
            )
            df_temp.x = time_samples
            df_temp.y = samples
            df = pd.concat([df, df_temp], axis=0, ignore_index=True)
            plot_timeseries(process_type, container, time_samples, samples)

        if process_type == ProcessType.GaussianProcess.value:
            df = pd.DataFrame(columns=["ID", "x", "y"])
            df_temp = pd.DataFrame(columns=["ID", "x", "y"])

            kernel_types = tuple(ProcessKernel.__members__.keys())
            kernel = st.radio("Kernel", kernel_types)

            if kernel == ProcessKernel.SE.value:
                # the squared exponential
                # for i in range(num_timeseries):
                time_samples, samples = generate_data(
                    process_type=process_type,
                    num_points=num_points,
                    irregular=irregular,
                    keep_percentage=keep_percentage,
                    std_noise=std_noise,
                    kernel=kernel,
                )
                df_temp.x = time_samples
                df_temp.y = samples
                df = pd.concat([df, df_temp], axis=0, ignore_index=True)
                plot_timeseries(process_type, container, time_samples, samples)

            if kernel == ProcessKernel.Constant.value:
                # All covariances set to `variance`
                variance = st.slider("variance", 0.0, 1.0, 1.0, 0.1)
                # for i in range(num_timeseries):
                time_samples, samples = generate_data(
                    process_type=process_type,
                    num_points=num_points,
                    irregular=irregular,
                    keep_percentage=keep_percentage,
                    std_noise=std_noise,
                    kernel=kernel,
                    variance=variance,
                )
                df_temp.x = time_samples
                df_temp.y = samples
                df = pd.concat([df, df_temp], axis=0, ignore_index=True)
                plot_timeseries(process_type, container, time_samples, samples)

            if kernel == ProcessKernel.Exponential.value:
                gamma = st.slider("gamma", 0.0, 1.0, 1.0, 0.1)  # TODO: check range
                # for i in range(num_timeseries):
                time_samples, samples = generate_data(
                    process_type=process_type,
                    num_points=num_points,
                    irregular=irregular,
                    keep_percentage=keep_percentage,
                    std_noise=std_noise,
                    kernel=kernel,
                    gamma=gamma,
                )
                df_temp.x = time_samples
                df_temp.y = samples
                df = pd.concat([df, df_temp], axis=0, ignore_index=True)
                plot_timeseries(process_type, container, time_samples, samples)

            if kernel == ProcessKernel.RQ.value:
                alpha = st.slider("alpha", 0.0, 1.0, 1.0, 0.1)  # TODO: check range
                # for i in range(num_timeseries):
                time_samples, samples = generate_data(
                    process_type=process_type,
                    num_points=num_points,
                    irregular=irregular,
                    keep_percentage=keep_percentage,
                    std_noise=std_noise,
                    kernel=kernel,
                    alpha=alpha,
                )
                df_temp.x = time_samples
                df_temp.y = samples
                df = pd.concat([df, df_temp], axis=0, ignore_index=True)
                plot_timeseries(process_type, container, time_samples, samples)

            if kernel == ProcessKernel.Linear.value:
                c = st.slider("c", 0.0, 1.0, 1.0, 0.1)  # TODO: check range
                offset = st.slider("offset", 0.0, 1.0, 1.0, 0.1)  # TODO: check range
                # for i in range(num_timeseries):
                time_samples, samples = generate_data(
                    process_type=process_type,
                    num_points=num_points,
                    irregular=irregular,
                    keep_percentage=keep_percentage,
                    std_noise=std_noise,
                    kernel=kernel,
                    c=c,
                    offset=offset,
                )
                df_temp.x = time_samples
                df_temp.y = samples
                df = pd.concat([df, df_temp], axis=0, ignore_index=True)
                plot_timeseries(process_type, container, time_samples, samples)
                plot_timeseries(process_type, container, time_samples, samples)

            if kernel == ProcessKernel.Matern.value:
                nu = st.slider("nu", 0.0, 1.0, 1.0, 0.1)  # TODO: check range
                # for i in range(num_timeseries):
                time_samples, samples = generate_data(
                    process_type=process_type,
                    num_points=num_points,
                    irregular=irregular,
                    keep_percentage=keep_percentage,
                    std_noise=std_noise,
                    kernel=kernel,
                    nu=nu,
                )
                df_temp.x = time_samples
                df_temp.y = samples
                df = pd.concat([df, df_temp], axis=0, ignore_index=True)
                plot_timeseries(process_type, container, time_samples, samples)

            if kernel == ProcessKernel.Periodic.value:
                period = st.slider("period", 0.0, 1.0, 1.0, 0.1)  # TODO: check range
                # for i in range(num_timeseries):
                time_samples, samples = generate_data(
                    process_type=process_type,
                    num_points=num_points,
                    irregular=irregular,
                    keep_percentage=keep_percentage,
                    std_noise=std_noise,
                    kernel=kernel,
                    period=period,
                )
                df_temp.x = time_samples
                df_temp.y = samples
                df = pd.concat([df, df_temp], axis=0, ignore_index=True)
                plot_timeseries(process_type, container, time_samples, samples)

        if process_type == ProcessType.PseudoPeriodic.value:
            df = pd.DataFrame(columns=["ID", "x", "y"])
            df_temp = pd.DataFrame(columns=["ID", "x", "y"])

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

            # for i in range(num_timeseries):
            time_samples, samples = generate_data(
                process_type=process_type,
                num_points=num_points,
                irregular=irregular,
                keep_percentage=keep_percentage,
                std_noise=std_noise,
                frequency=frequency,
                freqSD=freqSD,
                ampSD=ampSD,
                amplitude=amplitude,
            )

            df_temp.x = time_samples
            df_temp.y = samples
            df = pd.concat([df, df_temp], axis=0, ignore_index=True)
            plot_timeseries(process_type, container, time_samples, samples)

        if process_type == ProcessType.AutoRegressive.value:
            df = pd.DataFrame(columns=["ID", "x", "y"])
            df_temp = pd.DataFrame(columns=["ID", "x", "y"])
            use_ar2 = st.checkbox("Use AR2, else AR1")
            phi_1 = st.slider(f"phi_1", 0.0, 2.0, 1.0, 0.1)
            phi_2 = st.slider(f"phi_2", 0.0, 2.0, 1.0, 0.1, disabled=not use_ar2)

            if use_ar2:
                ar_param = [phi_1, phi_2]
            else:
                ar_param = [phi_1]

            # for i in range(num_timeseries):
            time_samples, samples = generate_data(
                process_type=process_type,
                num_points=num_points,
                irregular=False,
                std_noise=std_noise,
                ar_param=ar_param,
            )

            df_temp.x = time_samples
            df_temp.y = samples
            df = pd.concat([df, df_temp], axis=0, ignore_index=True)
            plot_timeseries(process_type, container, time_samples, samples)

        if process_type == ProcessType.CAR.value:
            df = pd.DataFrame(columns=["ID", "x", "y"])
            df_temp = pd.DataFrame(columns=["ID", "x", "y"])
            ar_param = st.slider(
                f"ar_param", 0.0, 2.0, 1.0, 0.1, help="Parameter of the AR(1) process"
            )
            # for i in range(num_timeseries):
            time_samples, samples = generate_data(
                process_type=process_type,
                num_points=num_points,
                irregular=False,
                std_noise=std_noise,
                ar_param=ar_param,
            )

            df_temp.x = time_samples
            df_temp.y = samples
            df = pd.concat([df, df_temp], axis=0, ignore_index=True)
            plot_timeseries(process_type, container, time_samples, samples)

        if process_type == ProcessType.NARMA.value:
            df = pd.DataFrame(columns=["ID", "x", "y"])
            df_temp = pd.DataFrame(columns=["ID", "x", "y"])
            order = st.slider("order", 1, 10, 3, 1, help="Order of the NARMA process")
            coefficients = None  # TODO: implement?
            initial_condition = None  # TODO: implement?
            # for i in range(num_timeseries):
            time_samples, samples = generate_data(
                process_type=process_type,
                num_points=num_points,
                irregular=False,
                std_noise=std_noise,
                order=order,
            )

            df_temp.x = time_samples
            df_temp.y = samples
            df = pd.concat([df, df_temp], axis=0, ignore_index=True)
            plot_timeseries(process_type, container, time_samples, samples)

        # Save to MongoDB
        st.sidebar.header("Save to MongoDB")
        collection_name = "time_series"
        document_name = st.text_input("Document name", "")
        save_db = st.button("Save")
        if save_db and document_name:
            doc_id = save_time_series(
                col_name=collection_name, doc_name=document_name, time_series=df
            )
            st.success("Saved document!")


if __name__ == "__main__":
    run()
