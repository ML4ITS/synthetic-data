import torch


class LSTM(torch.nn.Module):
    def __init__(self, hidden_layers: int = 64):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.lstm1 = torch.nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = torch.nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = torch.nn.Linear(self.hidden_layers, 1)

    def forward(self, x_train: torch.Tensor, future: int = 0):
        outputs = []
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
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        return torch.cat(outputs, dim=1)
