import numpy as np
import scipy.signal as signal
from python_speech_features import mfcc
import pandas as pd
import matplotlib.pyplot as plt


def generate_signals(n_points=10):
    t = np.linspace(0, 1, n_points, endpoint=False)

    # Generate our variable time step durations (dt-s) by combining sine, square and pwm waves
    sq = signal.square(2 * np.pi * 5 * t)  # square wave
    sig = np.sin(3 * np.pi * t)  # sine wave
    pwm = signal.square(4 * np.pi * 30 * t, duty=(sig + 1) / 2)  # pwm wave
    ts = 10 * (pwm + sig + sq)  # amplify
    ts -= np.min(ts)  # move to positive range

    # Our actual signal values will be the sum of sine wave and gaussian noise
    sig_reg = np.sin(8 * np.pi * t)
    noise = np.random.normal(0, 0.2, sig_reg.shape[0])
    input_signal = sig_reg + noise

    # Signal as a function of time determined by our dt-s
    irregular_t = np.cumsum(ts)
    irregular_t /= np.max(irregular_t)

    return t, irregular_t, input_signal


def convert_to_mfccs(wav, step, window):
    return mfcc(wav, 1, winstep=step, winlen=window, numcep=32)


def df_to_csv(df):
    return df.to_csv().encode("utf-8")


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
