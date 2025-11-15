import torch

config = {
  'n_samples': 1000,
  'input_dim': 2,
  'hidden_dim': 128,
  'out_dim': 2,
  'time_emd_dim': 32,
  'lr': 0.0001,
  'device': 'cuda' if torch.cuda.is_available() else 'cpu',
  'epoch': 10000,
  'n_steps': 100
}