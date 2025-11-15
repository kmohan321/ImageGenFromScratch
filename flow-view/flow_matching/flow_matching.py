import torch
from config import config
import matplotlib.pyplot as plt
from data import x_train

device = config['device']
n_steps = config['n_steps']
n_generate = config['n_samples']
input_dim = config['input_dim']

print('loading the saved models...')
model = torch.load('saved_items/velocity_model.pth', weights_only=False)
emd_model = torch.load('saved_items/embedding_model.pth', weights_only=False)
model = model.to(device)
emd_model = emd_model.to(device)

print('generating the samples..')
@torch.no_grad()
def sample(model, time_emb_model, n_samples, data_dim, n_steps):
    """
    Generates samples using the trained velocity model with Euler's method.
    """
    print(f"Generating {n_samples} samples using {n_steps} steps...")
    model.eval()

    dt = 1.0 / n_steps
    x_t = torch.randn(n_samples, data_dim, device=device)

    for i in range(n_steps):
        t_val = i / n_steps
        t = torch.full((n_samples,), t_val,device=device)
        t_emb = time_emb_model(t)
        velocity = model(x_t, t_emb)
        x_t = x_t + velocity * dt

    return x_t.cpu().numpy()

generated_samples = sample(model, emd_model, n_generate, input_dim, n_steps)
original_data = x_train

#plot for generated samples and original data
plt.figure(figsize=(10, 6))
plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.5, label='Original Data')
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, label='Generated Samples', marker='x', color='red')
plt.title("Flow Matching: Original vs. Generated Samples")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.axis('equal')

plt.savefig('saved_items/generated_graph.png')