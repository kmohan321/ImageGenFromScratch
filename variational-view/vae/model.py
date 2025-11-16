import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
  def __init__(self,
               input_dim: int,
               output_dim: int
               ):
    super().__init__()
    
    self.layer1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding = 1)
    self.layer2 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding = 1)
    
    self.down = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, stride=2)
    self.act = nn.SiLU()
    
    self.norm1 = nn.GroupNorm(input_dim//2, input_dim)
    self.norm2 = nn.GroupNorm(output_dim//2, output_dim)
    
    self.residual_link = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=2)
    
    
  def forward(self, x):
    residual = self.residual_link(x)
    x = self.act(self.norm1(self.layer1(x)))
    x = self.down(x)
    x = self.act(self.norm2(self.layer2(x)))
    x = x + residual
    return x 
  
class DecoderBlock(nn.Module):
  def __init__(self,
               input_dim: int,
               output_dim: int
               ):
    super().__init__()
    
    self.layer1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding = 1)
    self.layer2 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding = 1)
    
    self.up = nn.Upsample(scale_factor=2, mode='nearest')
    self.act = nn.SiLU()
    
    self.norm1 = nn.GroupNorm(input_dim//2, input_dim)
    self.norm2 = nn.GroupNorm(output_dim//2, output_dim)
    
    self.residual_upsample = nn.Upsample(scale_factor=2, mode='nearest')
    self.residual_conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)
    
    
  def forward(self, x):
    residual = self.residual_upsample(x)
    residual = self.residual_conv(residual)
    x = self.act(self.norm1(self.layer1(x)))
    x = self.up(x)
    x = self.act(self.norm2(self.layer2(x)))
    x = x + residual
    return x 

def reparameterize(mu, log_var):
    """
    Performs the reparameterization trick.
    z = mu + std * epsilon
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

class VAE(nn.Module):
  def __init__(self,
               cfg
               ):
    super().__init__()
    
    #encoder 
    self.input_conv = nn.Conv2d(cfg['input_channel'],cfg['channel_list'][0],kernel_size=3,padding=1)
    self.enc_block = nn.ModuleList([EncoderBlock(cfg['channel_list'][i],
                                                 cfg['channel_list'][i+1]
                                                 ) for i in range(len(cfg['channel_list'])-1)])
    self.bottlneck = nn.Conv2d(cfg['channel_list'][-1], 2*cfg['latent_channel'], kernel_size=3, padding=1)
    #decoder
    self.decoder_layer = nn.Conv2d(cfg['latent_channel'], cfg['channel_list'][-1],kernel_size=3,padding=1)
    reversed_channel = list(reversed(cfg['channel_list']))
    self.dec_block = nn.ModuleList([DecoderBlock(reversed_channel[i],
                                                 reversed_channel[i+1]) 
                                    for i in range(len(reversed_channel)-1)])
    self.out_conv = nn.Conv2d(cfg['channel_list'][0],cfg['out_channel'],kernel_size=3,padding=1)
    
    
  def forward(self, x):
    
    x = self.input_conv(x)
    for block in self.enc_block:
      x = block(x)
      
    x = self.bottlneck(x)
    sigma, log_var = torch.chunk(x,2,dim=1)
    
    x = reparameterize(sigma, log_var)
    
    x = self.decoder_layer(x)
    for block in self.dec_block:
      x = block(x)
    x = self.out_conv(x)
    return x, sigma, log_var