import torch
from config import config
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons

#dataset
n_samples = config['n_samples']
x, _ = make_moons(n_samples, noise=0.05)

x_train = TensorDataset(torch.tensor(x, dtype=torch.float32))
train_loader = DataLoader(x_train, batch_size=32, shuffle=True, pin_memory=True)