# Simple Continuous Normalizing Flows from Scratch

A minimal PyTorch implementation of a **Continuous Normalizing Flow (CNF)** for generative modeling on the 'make_moons' dataset. This code is designed to be a simple, easy-to-read demonstration of the core concepts.

## 📈 Example Result

After training, the model learns to transform a simple Gaussian distribution onto the data manifold, successfully capturing the "two moons" shape.

![Generated samples on the two-moons dataset](graph_cnf.png.png)

---

## 🧠 The Core Idea: The Change of Variables Formula in Continuous Time

This implementation trains a model $f_\theta(z, t)$ by treating the generative process as the solution to an Ordinary Differential Equation (ODE).

### 1. The "Ideal" (Governing) Equation

Our "ideal" goal is to learn a transformation $\Phi$ that maps a complex data distribution $p_X(x)$ to a simple base distribution $p_Z(z)$ (e.g., a Gaussian). The **Change of Variables** formula dictates the relationship between these densities:

$$
\log p_X(x) = \log p_Z(z) + \log \left| \det \left( \frac{\partial z}{\partial x} \right) \right|
$$

In a *Continuous* Normalizing Flow, this transformation is defined by an ODE, $\frac{dz_t}{dt} = f_\theta(z_t, t)$. The log-determinant term becomes the integral of the **trace of the Jacobian** of $f_\theta$:

$$
\log p_X(x) = \log p_Z(z_0) - \int_{0}^{T} \text{Tr}\left( \frac{\partial f_\theta(z_t, t)}{\partial z_t} \right) dt
$$

This is **intractable** to compute directly, as we need to solve the ODE and compute the trace integral simultaneously.

### 2. The "Practical" (Tractable) Loss

The "trick" is to use a **Neural ODE solver**. We define a single, augmented ODE that, when solved from $t=T$ (data $x$) down to $t=0$ (noise $z_0$), gives us both the latent vector and the log-determinant term.

This gives a new, **tractable** loss, the **Negative Log-Likelihood (NLL)**:

$$
\mathcal{L}_{\text{NLL}} = - \mathbb{E}_{x \sim p_X} \left[ \log p_X(x) \right]
$$

We minimize this loss by:
1.  Solving the ODE $x \rightarrow z_0$ to get the latent code and the log-det integral.
2.  Calculating $\log p_Z(z_0)$ (the log-prob of the code under the Gaussian).
3.  Summing them and backpropagating through the entire ODE solve.

The key theoretical insight is that the `torchdiffeq` library can backpropagate through the ODE solver, allowing us to optimize $f_\theta$ to minimize this NLL.

## ✨ Repo Features

* **Model (`ODEfunc`):** A simple Multi-Layer Perceptron (MLP) that defines the vector field $f_\theta(z, t)$.
* **Time Conditioning:** The network is conditioned on the current time $t$.
* **Neural ODE Solver:** Uses `torchdiffeq` to solve the augmented ODE, simultaneously finding $z_0$ and the log-determinant integral.
* **Jacobian Trace:** An efficient implementation (using the "sum trick") to compute the exact instantaneous trace $\text{Tr}(\frac{\partial f}{\partial z_t})$ needed for the loss.
* **Training:** A complete training loop for the `make_moons` dataset using the $\mathcal{L}_{\text{NLL}}$ objective.
* **Sampling:** An ODE solver to generate new samples by integrating from a base noise sample $z_0 \sim p_Z(z)$ at $t=0$ to $x = z_T$ at $t=T$.
* **Visualization:** A Matplotlib script to plot the generated samples against the original data.

## 📄 Reference

This code is an implementation of the ideas presented in:

* **Neural Ordinary Differential Equations:** [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366)
* **FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models:** [https://arxiv.org/abs/1810.01367](https://arxiv.org/abs/1810.01367)