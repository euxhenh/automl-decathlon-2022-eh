import torch
import torch.nn as nn


class MyGRU(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            output_dim,
            dropout_p=0,
    ):
        """Init GRU
        """
        super(MyGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p,
        )
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x should be (batch, seq_length, features)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.reshape(out.shape[0], -1)
