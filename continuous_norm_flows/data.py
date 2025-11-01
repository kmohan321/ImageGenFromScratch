import torch
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
from config import config

n_samples = config['n_samples']
batch = config['batch']

X, _ = make_moons(n_samples=n_samples, noise=0.08)
X = X * 1.5           # scale for nicer visualization

train_dataset = TensorDataset(torch.tensor(X,dtype=torch.float32))
train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, pin_memory=True)
