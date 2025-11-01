import math
import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    """
    Defines a fixed sinusoidal embedding layer for time.
    """
    def __init__(self, emb_dim):
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError(f"Embedding dimension {emb_dim} must be even.")
        self.emb_dim = emb_dim

    def forward(self, t):
        half_dim = self.emb_dim // 2
        exponent = torch.arange(half_dim, dtype=torch.float32, device=t.device) / (half_dim - 1)
        freqs = torch.exp(-math.log(10000.0) * exponent)
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class VelocityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, time_emd_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.proj_x = nn.Linear(input_dim, hidden_dim)
        self.proj_t = nn.Linear(time_emd_dim, hidden_dim)

    def forward(self, x, t):
        x = self.proj_x(x)
        t = self.proj_t(t)
        x = x + t
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x