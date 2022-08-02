from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from synthetic_data.mlops.tmp.data import HarmonicDataset

BATCH_SIZE = 50
SPLIT_SIZE = 50
SPLIT_RATIO = 1
DATASET_NAME = "AMP20"

dataset = HarmonicDataset(DATASET_NAME, SPLIT_SIZE, SPLIT_RATIO)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

shapes = dataset.dataset.numpy().shape
batches = len(iter(dataloader))
print(f"Data  : {batches} x {shapes}")

for i, data in enumerate(dataloader):
    print(f"Batch {i}: {data.numpy().shape}")

plt.figure(figsize=(10, 4))
for i, data in enumerate(dataloader):
    y = data[0].numpy()
    x = range(len(y))

    plt.plot(x, y, label=f"Batch {i}")
    break
plt.show()
