from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        activation: Optional[Literal["sig"]] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.memory_size = input_size
        self.activation = activation

        self.fc_z_0 = nn.Linear(input_size, input_size)
        self.fc_z_1 = nn.Linear(input_size, input_size)

        self.fc_r_0 = nn.Linear(input_size, input_size)
        self.fc_r_1 = nn.Linear(input_size, input_size)

        self.fc_h_0 = nn.Linear(input_size, input_size)
        self.fc_h_1 = nn.Linear(input_size, input_size)

        self.fc_action = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor, h_p: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        # Computing z.
        z_a = self.fc_z_0(x)
        z_b = self.fc_z_1(h_p)
        z = torch.sigmoid(z_a + z_b)

        # Computing r.
        r_a = self.fc_r_0(x)
        r_b = self.fc_r_1(h_p)
        r = torch.sigmoid(r_a + r_b)

        # Computing h_hat.
        h_a = self.fc_h_0(x)
        h_b = self.fc_h_1(h_p * r)
        h_hat = torch.tanh(h_a + h_b)

        # Computing h.
        h = (1 - z) * h_p + z * h_hat

        # Modification to compute action.
        action = self.fc_action(h)

        if self.activation == "sig":
            action = (torch.sigmoid(action) - 0.5) * 2

        return action, h

    def generate_memory(self) -> torch.Tensor:
        return torch.Tensor(torch.zeros(self.memory_size))


class GRUHidden(GRU):
    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        n_hidden: int = 8,
        activation: Optional[Literal["sig"]] = None,
    ) -> None:
        super().__init__(input_size, output_size, activation)
        self.memory_size = n_hidden

        self.fc_z_0 = nn.Linear(input_size, n_hidden)
        self.fc_z_1 = nn.Linear(n_hidden, n_hidden)

        self.fc_r_0 = nn.Linear(input_size, n_hidden)
        self.fc_r_1 = nn.Linear(n_hidden, n_hidden)

        self.fc_h_0 = nn.Linear(input_size, n_hidden)
        self.fc_h_1 = nn.Linear(n_hidden, n_hidden)

        self.fc_action = nn.Linear(n_hidden, output_size)
