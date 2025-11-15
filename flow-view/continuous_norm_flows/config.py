import torch

config ={
  'n_samples': 1000,
  'epochs': 100,
  'input_dim': 2,
  'hidden_dim': 64,
  'output_dim': 2,
  'lr': 1e-3,
  'device': 'cuda' if torch.cuda.is_available() else 'cpu',
  'batch': 32
}