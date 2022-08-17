import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from mlflow.tracking import MlflowClient

from synthetic_data.common.config import LocalConfig

cfg = LocalConfig()
mlflow.set_tracking_uri(cfg.URI_MODELREG_REMOTE)

from pprint import pprint

client = MlflowClient()
for rm in client.list_registered_models():
    model_name = rm.name
    for mod in client.search_model_versions(f"name='{model_name}'"):
        version_id = int(mod.version)
        print(
            f"model_name: {model_name}, version_id: {version_id} (type= {type(version_id)})"
        )

# name = "WGAN-GP"
# version = 2
# device = torch.device("cpu")
# model_uri = f"models:/{name}/{version}"

# model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=device)
# model.to(device)
# model.eval()

# noise = torch.randn((10, 100))
# sequences = model(noise)
# sequences = sequences.detach().numpy().tolist()


# fig, axis = plt.subplots(
#     nrows=10,
#     ncols=1,
#     figsize=(14, 12),
#     dpi=300,
#     sharex=True,
#     sharey=True,
#     constrained_layout=True,
# )

# for i in range(10):
#     sequence = sequences[i]
#     axis[i].plot(sequence)

# fig.suptitle("Generated frequencies")
# fig.supxlabel("Time steps")
# fig.supylabel("Amplitude")
# plt.show()
