import torch
import matplotlib.pyplot as plt
from config import config

n_samples = config['n_samples']
device = config['device']

print("loading the model...")
model = torch.load("saved_items/model.pth", weights_only=False)
model = model.to(device)
model.eval()

z_base = torch.randn(n_samples, 2).to(device)
z_base = z_base.requires_grad_()

z_t, _ = model(z_base,t_start=1, t_end =0)  # from t=0 to t=1
z_transformed = z_t[-1].detach().cpu()
z_base = z_base.detach().cpu()

print("plotting the graph...")
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(z_base[:,0], z_base[:,1], s=5, color='skyblue')
plt.title("Base Gaussian Distribution")
plt.axis("equal")

plt.subplot(1,2,2)
plt.scatter(z_transformed[:,0], z_transformed[:,1], s=5, color='coral')
plt.title("Transformed (Moons) Distribution")
plt.axis("equal")

plt.savefig("saved_items/graph_cnf.png")