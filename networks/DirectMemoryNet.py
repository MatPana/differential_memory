from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init


class DirectMemoryNet(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        memory_size: int = 4,
        hidden_size: int = 64,
        memory: Literal["default", "diag", "sums", "tanh", "mock"] = "default",
        activation: Optional[Literal["sig"]] = None,
        device: str = "cpu",
    ):
        super(DirectMemoryNet, self).__init__()
        self.memory = memory
        self.activation = activation
        self.device = device

        self.input_size = input_size
        self.output_size = output_size

        self.memory_size = memory_size
        self.hidden_size = hidden_size
        combined_size = self.memory_size + self.input_size

        # Additional hidden layer for memory processing
        self.fc_write_1 = nn.Linear(combined_size, hidden_size)
        self.fc_write_2 = nn.Linear(hidden_size, memory_size)

        self.fc_overwrite = nn.Linear(hidden_size, 1)

        if memory == "diag":
            # Initialize fc_write_1
            with torch.no_grad():
                # Identity matrix for the memory part
                init.eye_(
                    self.fc_write_1.weight[:memory_size, : memory_size + input_size]
                )
                # Xavier initialization for the remaining part
                init.xavier_uniform_(
                    self.fc_write_1.weight[memory_size:, : memory_size + input_size],
                    gain=init.calculate_gain("relu"),
                )
                init.zeros_(self.fc_write_1.bias)

            # Initialize weights and biases for fc_write_2
            with torch.no_grad():
                init.eye_(self.fc_write_2.weight)
                init.zeros_(self.fc_write_2.bias)

        self.fc_act = nn.Linear(hidden_size, output_size)

    def forward(
        self, observations: torch.Tensor, ext_memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if isinstance(observations, np.ndarray):
            observations = torch.tensor(observations, dtype=torch.float)

        # Clear the memory if is is mock memory version.
        if self.memory == "mock":
            ext_memory == self.generate_memory()

        combined = torch.hstack((ext_memory, observations))

        # Memory processing
        intermediate = f.relu(self.fc_write_1(combined))

        if self.memory == "sums":
            new_memory = ext_memory + self.fc_write_2(intermediate)
        elif self.memory == "tanh":
            # Tanh forget.
            overwrite = torch.sigmoid(self.fc_overwrite(intermediate))
            new_memory = overwrite * ext_memory + self.fc_write_2(intermediate)
        else:
            new_memory = self.fc_write_2(intermediate)

        action = self.fc_act(intermediate)

        if self.activation == "sig":
            action = (torch.sigmoid(action) - 0.5) * 2

        return action, new_memory

    def generate_memory(self) -> torch.Tensor:
        return torch.Tensor(torch.randn(self.memory_size) * 0.05).to(self.device)
