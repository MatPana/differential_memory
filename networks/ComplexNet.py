from typing import Literal, Optional
import torch
import torch.nn as nn
import numpy as np


class ComplexNet(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        out_size: int = 1,
        hidden_size: int = 32,
        memory_size: int = 8,
        activation: Optional[Literal["sig"]] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.activation = activation
        self.memory_size = memory_size
        self.hidden_size = hidden_size

        self.fc_proc_0 = nn.Linear(input_size, hidden_size)
        self.relu_proc_0 = nn.ReLU()
        self.fc_query = nn.Linear(hidden_size, hidden_size)
        self.relu_read = nn.Tanh()

        self.fc_proc_1 = nn.Linear(memory_size + hidden_size, hidden_size)
        self.relu_proc_1 = nn.ReLU()
        self.fc_message = nn.Linear(hidden_size, memory_size)
        self.fc_address = nn.Linear(hidden_size, hidden_size)

        self.fc_act = nn.Linear(hidden_size, out_size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> tuple:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        proc_0 = self.fc_proc_0(x)
        query = self.fc_query(proc_0)
        read = memory @ query


        combined = torch.hstack((proc_0, read))
        proc_1 = self.fc_proc_1(combined)

        message = self.fc_message(proc_1)
        address = torch.softmax(self.fc_address(proc_1), dim=-1)

        write = torch.ger(message, address)
        new_memory = memory + write

        action = self.fc_act(proc_1)

        if self.activation == "sig":
            action = (torch.sigmoid(action) - 0.5) * 2

        return action, new_memory

    def generate_memory(self) -> torch.Tensor:
        return torch.Tensor(torch.randn(self.memory_size, self.hidden_size) * 0.05).to(
            self.device
        )
