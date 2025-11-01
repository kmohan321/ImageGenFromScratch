import torch
import torch.nn as nn
import torch.optim as optim
from model import VelocityNet, SinusoidalTimeEmbedding
from config import config
from data import train_loader
from tqdm import tqdm

epochs = config['epoch']
input_dim = config['input_dim']
hidden_dim = config['hidden_dim']
out_dim = config['out_dim']
time_emd_dim = config['time_emd_dim']
lr = config['lr']
device = config['device']

model = VelocityNet(input_dim, hidden_dim, out_dim,time_emd_dim).to(device)
emd_model = SinusoidalTimeEmbedding(time_emd_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#simple training steps
for epoch in range(epochs):
  running_loss = 0
  pbar = tqdm(train_loader, desc=f"{epoch}/{epochs} epochs")
  for batch in pbar:
    
    batch_size = batch[0].shape[0]
    batch_x1 = batch[0].to(device)
    t = torch.rand(size=(batch_size,), device=device)
    t_emd = emd_model(t)
    t = t.unsqueeze(1)
    batch_x0 = torch.randn_like(batch_x1)
    xt = (1 - t) * batch_x0 + t * batch_x1

    target = batch_x1 - batch_x0

    output = model(xt, t_emd)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    pbar.set_postfix({"Loss": f"{running_loss/len(train_loader):.4f}"})

print("saving the models...")
torch.save(model, 'saved_items/velocity_model.pth')
torch.save(emd_model, 'saved_items/embedding_model.pth')
