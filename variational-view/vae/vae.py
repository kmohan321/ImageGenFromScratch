import torch
import torchvision.utils as vutils
from config import config

print(f"loading the model")
vae = torch.load("saved_items/model.pth", weights_only=False)

device = config['device']
print("Generating images from random latent samples...")
vae.eval()

# Define your latent dimensions
latent_channels = 8  # Or whatever you used in your config
latent_h = 7
latent_w = 7
num_to_generate = 64 # Let's generate a full grid

with torch.no_grad():

    z = torch.randn(num_to_generate, latent_channels, latent_h, latent_w).to(device)
    
    z = vae.decoder_layer(z)
    for block in vae.dec_block:
      z = block(z)
    x_gen = vae.out_conv(z)
    
    vutils.save_image(x_gen, f"saved_items/gen.png", nrow=8, normalize=True)

print(f"Saved {num_to_generate} generated images")