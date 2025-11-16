import torch
config ={
  'input_channel': 1,
  'out_channel': 1,
  'latent_channel': 8,
  'channel_list': [32,64,128],
  'device': 'cuda' if torch.cuda.is_available() else 'cpu',
  'batch_size': 32,
  'lr': 1e-3,
  'epochs': 15
}
