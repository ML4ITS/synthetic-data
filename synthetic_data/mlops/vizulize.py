from bokeh.plotting import figure, show
from torch.utils.data import DataLoader

from synthetic_data.mlops.tmp.data import HarmonicDataset

BATCH_SIZE = 10
SPLIT_SIZE = 100
SPLIT_RATIO = 1
DATASET_NAME = "AMP20"

dataset = HarmonicDataset(DATASET_NAME, SPLIT_SIZE, SPLIT_RATIO)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

N_SAMPLES = 5

fig = figure(
    title=f"Dataset: {DATASET_NAME}",
    x_axis_label="Timesteps",
    y_axis_label="Amplitude",
    max_height=300,
    height_policy="max",
)

for i, data in enumerate(dataloader):
    y = data[0].numpy()
    x = range(len(y))
    if i < N_SAMPLES:
        fig.line(x, y, legend_label="Regular")
        fig.circle(
            x=x,
            y=y,
            legend_label="Regular",
            fill_color="blue",
            size=1,
        )
show(fig)
