import torch
import torch.nn as nn
import torch.optim as optim
from data import train_loader, test_loader
from model import VAE
from tqdm import tqdm
import torchvision.utils as vutils
from config import config

device = config['device']
lr = config['lr']
epochs = config['epochs']

vae = VAE(cfg=config).to(device)
loss_fn = nn.L1Loss(reduction='sum')
optimizer = optim.Adam(vae.parameters(),lr=lr)
param = sum([p.numel() for p in vae.parameters()])
print(f"total numbers of parameters in vae: {param}")

vae.train()
print(f"Starting training on {device}...")

def save_reconstruction_preview(vae_model, 
                                test_loader, 
                                device, 
                                epoch_num,  
                                num_images=8):
    """
    Saves a grid of original vs. reconstructed images from one batch.
    """
    vae_model.eval()
    with torch.no_grad():
            image, _ = next(iter(test_loader))
            image = image.to(device)
            x_recon, _, _ = vae_model(image)

            originals = image[:num_images]
            recons = x_recon[:num_images]
            comparison = comparison = torch.cat([originals, recons])
            save_path = f"saved_items/epoch_{epoch_num:03d}.png"
            vutils.save_image(comparison, 
                              save_path, 
                              nrow=num_images,
                              normalize=True)
    vae_model.train()


for epoch in range(epochs):
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    total_loss_avg = 0.0
    mse_loss_avg = 0.0
    kl_loss_avg = 0.0
    
    
    for i, data in enumerate(loop):
        image, _ = data
        image = image.to(device)

        x_recon, mu, log_var = vae(image)
        mse_loss = loss_fn(x_recon, image)
        
        # kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=[1, 2, 3])
        # kl_loss  = torch.mean(kl_div)
        kl_loss  = torch.sum(kl_div)
        
        # kl_loss = beta * kl_loss
        total_loss = mse_loss + kl_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        total_loss_avg = (total_loss_avg * i + total_loss.item()) / (i + 1)
        mse_loss_avg = (mse_loss_avg * i + mse_loss.item()) / (i + 1)
        kl_loss_avg = (kl_loss_avg * i + kl_loss.item()) / (i + 1)

        loop.set_postfix(
            total_loss=f"{total_loss_avg:.4f}",
            mse_loss=f"{mse_loss_avg:.4f}",
            kl_loss=f"{kl_loss_avg:.4f}"
        )
        
    if (epoch + 1) % 5 == 0:
        
      save_reconstruction_preview(
          vae, 
          test_loader, 
          device,
          epoch + 1
      )
      
      print(f"\nSaved reconstruction preview for epoch {epoch+1}\n")

print("Training complete.")
torch.save(vae, f"saved_items/model.pth")
print("model saved successfully..")

