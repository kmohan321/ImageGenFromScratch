# VAE: Deriving the Evidence Lower Bound (ELBO)

This document provides the step-by-step mathematical derivation for the **Evidence Lower Bound (ELBO)**, which is the objective function optimized in a Variational Autoencoder (VAE).

## 1. The Goal: Maximize Log-Likelihood

Our goal in any generative model is to model the true data distribution \( p(x) \). We do this by defining a model \( p_\theta(x) \) with parameters \( \theta \) and trying to maximize the log-likelihood of our data:

$$
\log p_\theta(x)
$$

To create a more powerful model, we introduce a latent variable \( z \). This "marginalizes" the likelihood over \( z \):

$$
p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, \mathrm{d}z
$$

As we've discussed, this integral is **intractable** because it requires integrating over all possible values of the high-dimensional latent space \( z \). This means we cannot optimize \( \log p_\theta(x) \) directly.

---

## 2. The "Variational" Trick: Introduce an Approximation

Since we can't compute \( p_\theta(x) \), we also can't compute the true posterior \( p_\theta(z \mid x) \) (the "ideal" encoder) because of Bayes' theorem:

$$
p_\theta(z \mid x) = \frac{p_\theta(x \mid z)\, p(z)}{p_\theta(x)} \quad \leftarrow \text{Intractable denominator}
$$

The core idea of variational inference is to *approximate* this intractable posterior \( p_\theta(z \mid x) \) with a simpler, learnable distribution. We call this our **Encoder**, \( q_\phi(z \mid x) \).

Our goal is to make this approximation as "close" as possible to the true posterior, using the **Kullback–Leibler divergence (KL)**.

---

## 3. The Derivation: From KL to ELBO

We start with the KL divergence:

$$
D_{KL}\!\left(q_\phi(z \mid x)\,\|\, p_\theta(z \mid x)\right)
$$

By definition:

$$
D_{KL}\!\left(q_\phi(z \mid x)\,\|\, p_\theta(z \mid x)\right)
= \mathbb{E}_{q_\phi(z \mid x)}\!\left[ \log \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} \right]
$$

$$
= \mathbb{E}_{q_\phi(z \mid x)}\!\left[ \log q_\phi(z \mid x) - \log p_\theta(z \mid x) \right]
$$

Apply Bayes’ identity:

$$
\log p_\theta(z \mid x) = \log p_\theta(x, z) - \log p_\theta(x)
$$

Substituting:

$$
= \mathbb{E}_{q_\phi(z \mid x)}\!\left[ \log q_\phi(z \mid x) - \left(\log p_\theta(x,z) - \log p_\theta(x)\right) \right]
$$

$$
= \mathbb{E}_{q_\phi(z \mid x)}\!\left[ \log q_\phi(z \mid x) - \log p_\theta(x,z) \right] + \log p_\theta(x)
$$

Thus:

$$
D_{KL}\!\left(q_\phi(z \mid x)\,\|\, p_\theta(z \mid x)\right)
= \mathbb{E}_{q_\phi(z \mid x)}\!\left[ \log q_\phi(z \mid x) - \log p_\theta(x,z) \right] + \log p_\theta(x)
$$

Rearranging for \( \log p_\theta(x) \):

$$
\log p_\theta(x)
= D_{KL}\!\left(q_\phi(z \mid x)\,\|\, p_\theta(z \mid x)\right)
- \mathbb{E}_{q_\phi(z \mid x)}\!\left[ \log q_\phi(z \mid x) - \log p_\theta(x,z) \right]
$$

Which becomes:

$$
\log p_\theta(x)
= D_{KL}\!\left(q_\phi(z \mid x)\,\|\, p_\theta(z \mid x)\right)
+ \mathbb{E}_{q_\phi(z \mid x)}\!\left[ \log \frac{p_\theta(x,z)}{q_\phi(z \mid x)} \right]
$$

---

## 4. Understanding the Result

So:

$$
\log p_\theta(x)
= \underbrace{D_{KL}\!\left(q_\phi(z \mid x)\,\|\, p_\theta(z \mid x)\right)}_{\text{KL gap } \ge 0}
+ \underbrace{\mathcal{L}(\phi,\theta)}_{\text{ELBO}}
$$

Where:

$$
\mathcal{L}(\phi,\theta)
= \mathbb{E}_{q_\phi(z \mid x)}\!\left[ \log \frac{p_\theta(x,z)}{q_\phi(z \mid x)} \right]
$$

---

## 5. Deconstructing the ELBO

Using:

$$
p_\theta(x,z) = p_\theta(x \mid z)\, p(z)
$$

We get:

$$
\mathcal{L}(\phi,\theta)
= \mathbb{E}_{q_\phi(z \mid x)}\!\left[ \log p_\theta(x \mid z) \right]
- D_{KL}\!\left(q_\phi(z \mid x)\,\|\, p(z)\right)
$$

---

## 6. Final Loss Function

Since we **maximize ELBO**, but code minimizes loss:

$$
\text{Loss} = -\mathcal{L}(\phi,\theta)
= -\mathbb{E}_{q_\phi(z \mid x)}\!\left[ \log p_\theta(x \mid z) \right]
+ D_{KL}\!\left(q_\phi(z \mid x)\,\|\, p(z)\right)
$$

---

## 7. Results

### Generated Images
<p align="center">
  <img src="gen.png" width="300"><br>
  <em>Figure 1: Generated Image</em>
</p>

### Reconstructed Images
<p align="center">
  <img src="epoch_015.png" width="300"><br>
  <em>Figure 2: Reconstructed Image</em>
</p>
