import matplotlib
from mlops.tools.api import load_dataset

from synthetic_data.common.config import RemoteConfig
from synthetic_data.common.preprocessing import normalize_dataset, preprocess
from synthetic_data.common.vizu import vizualize_dataset

matplotlib.use("Qt5Agg")

SPLIT_SIZE = 40
SPLIT_RATIO = 0.3
DATASET_NAME = "Simple"

# ---
cfg = RemoteConfig()

raw_dataset = load_dataset(cfg, DATASET_NAME)
dataset = preprocess(raw_dataset, split_ratio=SPLIT_RATIO, split_size=SPLIT_SIZE)
x_train, y_train, x_test, y_test = normalize_dataset(dataset)

signal = raw_dataset.reshape(-1)
batch = x_train[0].numpy()

vizualize_dataset(signal, batch)
