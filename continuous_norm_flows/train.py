import torch
import torch.optim as optim
from model import VelocityModel
from data import train_dataloader
from config import config
from tqdm import tqdm

epochs = config['epochs']
input_dim = config['input_dim']
hidden_dim = config['hidden_dim']
output_dim = config['output_dim']
lr = config['lr']
device = config['device']

model = VelocityModel(input_dim,hidden_dim,output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    running_loss = 0
    pbar = tqdm(train_dataloader, desc=f"{epoch}/{epochs}")
    for batch in pbar:
      optimizer.zero_grad()
      batch = batch[0].to(device)
      z_t, log_px = model(batch)
      loss = -log_px.mean()
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    
      pbar.set_postfix({"loss": f"{running_loss/len(train_dataloader):.4f}"})

print("saving the model...")
torch.save(model, "saved_items/model.pth")
