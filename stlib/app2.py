import pandas as pd
import numpy as np
import utils
from bokeh.plotting import figure
import matplotlib.pyplot as plt
import scipy.signal as signal

description = "MFCC and Lomb-Scargle Spectrogram"

# Your app goes in the function run()
def run():

    import streamlit as st
    import pandas as pd
    import numpy as np
    import utils.utils as utils
    from bokeh.plotting import figure
    import matplotlib.pyplot as plt
    import scipy.signal as signal

    # st.set_page_config(layout="wide")

    st.header("MFCC and Lomb-Scargle Spectrogram")

    # AND in st.sidebar!
    with st.sidebar:
        st.write("-------------------------")
        n_points = st.slider("Number of Element in the Sequence", 1, 1000, 500)

        with st.container():
            st.subheader("Parameter for MFCC")
            window = st.slider("Window", 1, 1000, 50)
            step = st.slider("Step", 1, 1000, 10)
            fmin = st.slider("fmin", 1, 100, 1)
            fmax = st.slider("fmax", 1, 1000, 500)

    t, irregular_t, input_signal = utils.generate_signals(n_points=n_points)

    c1, c2 = st.columns((1, 1))

    with c2:
        p_irr = figure(
            title="Irregular time series",
            x_axis_label="x",
            y_axis_label="y",
            max_height=300,
            height_policy="max",
        )

        p_irr.line(irregular_t, input_signal, legend_label="Irregular", line_width=2)
        st.bokeh_chart(p_irr, use_container_width=True)

        with st.expander("See Distribution of Delta T"):
            dt_irreg = pd.Series(np.diff(irregular_t), name="dt")
            # st.write(dt_irreg)

            hist, edges = np.histogram(
                dt_irreg, density=True, range=(np.min(dt_irreg), np.max(dt_irreg))
            )
            p_dt_irreg = figure(max_height=300, height_policy="max")
            p_dt_irreg.quad(
                top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white"
            )
            st.bokeh_chart(p_dt_irreg, use_container_width=True)

        with st.container():
            # --------------- MFCC ---------------
            fig, ax_list = plt.subplots(nrows=2, figsize=(30, 15))

            # Pot MFCC spectrogram of input signal
            ax = ax_list[0]
            data = utils.convert_to_mfccs(input_signal, window=window, step=step)
            presentation_steps = np.arange(data.shape[0])
            ax.imshow(data.T, aspect="auto", interpolation="nearest")
            ax.set_ylabel("MFCC features")

            # Plot Lomb-Scargle spectrogram of input signal
            ax = ax_list[1]
            n_steps = (irregular_t.shape[0] - window) // step
            # fmin = 1  # smallest frequency in spectrogram channels
            # fmax = 500  # largest frequency in spectrogram channels
            f = np.linspace(fmin, fmax, 32)
            powers = []
            # Comput LS spectrogram on sliding window (bin) over input signal
            for step_idx in range(n_steps):
                x = irregular_t[step_idx * step : (step_idx * step) + window]
                y = input_signal[step_idx * step : (step_idx * step) + window]
                pgram = signal.lombscargle(x, y - np.mean(y), f, normalize=True)
                powers.append(pgram)
            ax.imshow(np.array(powers).T, aspect="auto", interpolation="nearest")
            ax.set_ylabel("Lomb-Scargle features")
            st.pyplot(fig)

    with c1:
        p = figure(
            title="Regular time series",
            x_axis_label="x",
            y_axis_label="y",
            max_height=300,
            height_policy="max",
        )

        p.line(t, input_signal, legend_label="Regular", line_width=2)
        st.bokeh_chart(p, use_container_width=True)

        with st.expander("See Distribution of Delta T"):
            dt = pd.Series(np.diff(t), name="dt")
            # st.write(dt)

            hist, edges = np.histogram(
                dt, density=True, range=(np.min(dt_irreg), np.max(dt_irreg))
            )
            p_dt_reg = figure(max_height=300, height_policy="max")
            p_dt_reg.quad(
                top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white"
            )
            st.bokeh_chart(p_dt_reg, use_container_width=True)

        with st.container():
            # --------------- MFCC ---------------
            fig, ax_list = plt.subplots(nrows=2, figsize=(30, 15))

            # Pot MFCC spectrogram of input signal
            ax = ax_list[0]
            data = utils.convert_to_mfccs(input_signal, window=window, step=step)
            presentation_steps = np.arange(data.shape[0])
            ax.imshow(data.T, aspect="auto", interpolation="nearest")
            ax.set_ylabel("MFCC features")

            # Plot Lomb-Scargle spectrogram of input signal
            ax = ax_list[1]
            n_steps = (t.shape[0] - window) // step
            # fmin = 1  # smallest frequency in spectrogram channels
            # fmax = 500  # largest frequency in spectrogram channels
            f = np.linspace(fmin, fmax, 32)
            powers = []
            # Comput LS spectrogram on sliding window (bin) over input signal
            for step_idx in range(n_steps):
                x = t[step_idx * step : (step_idx * step) + window]
                y = input_signal[step_idx * step : (step_idx * step) + window]
                pgram = signal.lombscargle(x, y - np.mean(y), f, normalize=True)
                powers.append(pgram)
            ax.imshow(np.array(powers).T, aspect="auto", interpolation="nearest")
            ax.set_ylabel("Lomb-Scargle features")
            st.pyplot(fig)


# end of app

# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    run()
