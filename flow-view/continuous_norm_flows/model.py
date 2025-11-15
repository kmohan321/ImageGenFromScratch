import torch
import torch.nn as nn
from torchdiffeq import odeint

class VelocityModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(input_dim + 1, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim)
    )

  def ode_solver(self, t, state):
    z_t, _ = state
    t = torch.ones(z_t.shape[0], 1, device = z_t.device) * t
    z_t_c = torch.cat([z_t, t], dim=1)

    dz_dt = self.model(z_t_c)

    trace = torch.zeros(z_t.shape[0],device=z_t.device)
    for i in range(z_t.shape[1]):
      grad = torch.autograd.grad(dz_dt[:,i].sum(), z_t, create_graph=True)
      grads = grad[0][:,i]
      trace += grads

    return dz_dt, -trace

  def forward(self, z, t_start = 0, t_end = 1, ode_steps = 50):

    # t_start = 0 -> x, t_end = 1 -> z (training)
    # t_start = 1 -> z, t_end = 0 -> x (inference)
    
    z = z.requires_grad_()
    grid = torch.linspace(t_start, t_end, ode_steps, device = z.device)
    logp_0 = torch.zeros(z.shape[0], device = z.device)
    state = (z, logp_0)

    z_t, delta_logp_t = odeint(self.ode_solver, state, grid, method='euler')
    z_0 = z_t[-1]
    log_z_0 = (-0.5* (z_0**2).sum(dim=1) - 0.5 * torch.log(torch.tensor(2 * torch.pi)) * z_t.shape[1])
    log_px = log_z_0 - delta_logp_t[-1]
    return z_t, log_px

