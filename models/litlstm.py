from typing import Optional

import pytorch_lightning as pl
import torch

# from db import database as db
# from torch.utils.data import DataLoader, Dataset
# from utils.modelling import load_and_split, normalize_dataset


# class TimeSeriesDataset(Dataset):
#     def __init__(self, data) -> None:
#         super().__init__()
#         self.data = data

#     def __getitem__(self, idx):
#         return self.data

#     def __len__(self):
#         return len(self.data)


# class TimeSeriesModule(pl.LightningDataModule):
#     def __init__(self, name, ratio, bs):
#         super().__init__()
#         self.name = name
#         self.ratio = ratio
#         self.bs = bs
#         self.trainset = None
#         self.valset = None

#     def setup(self, stage: Optional[str] = None) -> None:
#         self.dataset = db.load_time_series_as_numpy(self.name)
#         self.dataset = load_and_split(self.dataset, self.ratio, self.bs)
#         self.dataset = normalize_dataset(self.dataset)

#     def train_dataloader(self):
#         return DataLoader(TimeSeriesDataset(self.dataset[:2]), batch_size=2)

#     def val_dataloader(self):
#         return DataLoader(TimeSeriesDataset(self.dataset[2:]), batch_size=2)


class LitLSTM(pl.LightningModule):
    def __init__(
        self,
        prediction_steps: int,
        learning_rate: float,
        hidden_layers: int = 64,
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.prediction_steps = prediction_steps
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()
        self.lstm1 = torch.nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = torch.nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = torch.nn.Linear(self.hidden_layers, 1)
        self.automatic_optimization = False
        self.double()

    def forward(self, x_train: torch.Tensor, future: int = 0):
        outputs = []
        print("x_train has shape", x_train.shape)
        h_t = torch.zeros(x_train.size(0), self.hidden_layers, dtype=torch.double)
        c_t = torch.zeros(x_train.size(0), self.hidden_layers, dtype=torch.double)
        h_t2 = torch.zeros(x_train.size(0), self.hidden_layers, dtype=torch.double)
        c_t2 = torch.zeros(x_train.size(0), self.hidden_layers, dtype=torch.double)

        for input_t in x_train.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for _ in range(future):
            print()
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        return torch.cat(outputs, dim=1)

    def compute_training_loss(self, x, y):
        out = self(x)
        loss = self.criterion(out, y)
        return loss

    def compute_val_loss(self, x_test, y_test, future: int = 0):
        prediction = self(x_test, future=future)
        loss = self.criterion(prediction[:, :-future], y_test)
        self.log(f"Loss/test:", loss)
        return prediction  # , prediction

    def training_step(self, batch: list, batch_idx: Optional[int]):
        x_train, y_train = batch
        x_train, _ = x_train
        y_train, _ = y_train
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.compute_training_loss(x_train, y_train)
        self.log("Loss/train:", loss)
        self.manual_backward(loss)
        opt.step()

    def validation_step(self, batch: list, batch_idx: Optional[int]):
        x_val, y_val = batch
        x_val, _ = x_val
        y_val, _ = y_val
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.compute_val_loss(x_val, y_val, self.prediction_steps)
        self.log("Loss/val:", loss)
        self.manual_backward(loss)
        opt.step()

        """
        prediction = model(x_test, future=future)
        loss = criterion(prediction[:, :-future], y_test)
        logger.info(f"Loss/test: {loss.item()}")
        return prediction.detach().numpy(), loss
        """

    def configure_optimizers(self):
        return torch.optim.LBFGS(self.parameters(), lr=self.learning_rate)
