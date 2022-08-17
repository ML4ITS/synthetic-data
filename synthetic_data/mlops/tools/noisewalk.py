import os
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from synthetic_data.common.config import LocalConfig
from synthetic_data.common.torchutils import get_device
from synthetic_data.mlops.tools.analysis import slerp

cfg = LocalConfig()
mlflow.set_tracking_uri(cfg.URI_MODELREG_REMOTE)
mlflow.set_registry_uri(cfg.URI_MODELREG_REMOTE)

device = get_device()

OUT_DIR = "./tmp/test"
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(1337)
np.random.seed(1337)


def save_sequence(sequence: torch.Tensor, n: int, m: int) -> None:
    sequence = sequence.detach().numpy()[0]

    plt.figure(figsize=(10, 3), dpi=200)
    plt.plot(sequence)
    plt.title("frame %04d / %04d" % (n, m))
    plt.savefig("./tmp/test/frame%04d.jpg" % n)
    plt.close()


# Load model
name = "WGAN-GP"
version = "3"
model_uri = f"models:/{name}/{version}"
model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=device)
model.eval()

# Load initial data
noise_shape = (1, 100)  # batch_size, z_dim
noise_1 = torch.randn(noise_shape)  # initial noise

cur_iter = 0
max_iter = 100

t1 = time.monotonic()
while cur_iter < max_iter:

    # sample new destination
    noise_2 = torch.randn(noise_shape)

    for value in np.linspace(0, 1, 50):
        noise = slerp(float(value), noise_1, noise_2)
        sequence = model(noise)
        save_sequence(sequence, cur_iter, max_iter)
        cur_iter += 1

        if cur_iter % 100 == 0:
            et = time.monotonic() - t1
            print(f"{cur_iter: 3} / {max_iter} | {et:.2f} s")

    noise_1 = noise_2

et = time.monotonic() - t1
print(f"Elapsed time: {et:.2f} s")

# Create GIF from sequence images, and remove saved images after
cmd_images_to_gif = "ffmpeg -f image2 -framerate 10 -i tmp/test/frame%04d.jpg -loop -0 -pix_fmt bgr8 test123.gif"
cmd_remove_images = "rm tmp/test/frame*.jpg"
os.system(cmd_images_to_gif)
os.system(cmd_remove_images)
