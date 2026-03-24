# Multi-Task Bayesian Optimization

An educational Jupyter notebook implementing **Multi-Task Bayesian Optimization** from first principles, based on:

> Swersky, K., Snoek, J., & Adams, R. P. (2013). **Multi-Task Bayesian Optimization**. *NeurIPS 2013*.
> https://papers.nips.cc/paper/2013/hash/f33ba15effa5c10e873bf3842afb46a6-Abstract.html

All math is implemented using **numpy and scipy only** — no GPy, GPflow, or sklearn GPs. Every equation in the paper is derived, explained, visualised, and connected to working code.

---

## Files

| File | Description |
|---|---|
| `multi_task_bayesian_optimization.ipynb` | Main educational notebook (21 cells, 7 sections) |
| `utils.py` | Pure numpy/scipy helper library (kernels, GP inference, acquisition functions, plotting) |

---

## Installation

```bash
pip install numpy scipy matplotlib ipywidgets jupyter
```

Tested with Python 3.9+. All dependencies are standard scientific Python — no specialist GP libraries required.

---

## Running the notebook

```bash
cd /Users/andersonchen/machine-learning-visuals
jupyter notebook multi_task_bayesian_optimization.ipynb
```

Or with JupyterLab:

```bash
jupyter lab multi_task_bayesian_optimization.ipynb
```

Run cells top-to-bottom on first execution. Sections can be re-run independently after the imports cell (Cell 2) has been executed.

**Note:** Section 6 (cost-sensitive acquisition) runs a Monte Carlo loop over the input domain and may take 1–3 minutes depending on your machine. Section 3 (BO loop visualization) generates 15 subplot rows and may produce a large figure.

---

## Notebook sections

| Section | Topic | Paper equations covered |
|---|---|---|
| 1 | Gaussian Processes from scratch — prior sampling, posterior update | (1), (2) |
| 2 | Kernels — RBF vs Matérn 5/2, interactive hyperparameter widget | kernel definition |
| 3 | Single-task Bayesian Optimization on Branin-Hoo | (4), (5) |
| 4 | Multi-Task GPs — ICM kernel, Kronecker structure, transfer learning demo | (3) |
| 5 | Cold-start experiment — warm-started MT-BO vs single-task BO | (1)–(3) combined |
| 6 | Cost-sensitive acquisition — MC entropy search, Eq. 9 | (8), (9) |
| 7 | Visual math reference — one plot per equation for all 9 equations | (1)–(9) |

---

## utils.py API reference

### Kernels

```python
rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0)
```
Squared exponential kernel. Returns an (n, m) kernel matrix.

```python
matern52_kernel(X1, X2, lengthscale=1.0, variance=1.0)
```
Matérn 5/2 kernel (used in the paper). Returns an (n, m) kernel matrix.

### GP inference

```python
gp_posterior(X_train, y_train, X_test, kernel_fn, noise=1e-4, **kernel_kwargs)
# Returns: mu (m,), sigma (m,), K_inv (n,n)
```
Computes GP posterior mean and standard deviation (paper equations 1 and 2). Uses Cholesky decomposition for numerical stability.

```python
sample_gp_prior(X, kernel_fn, n_samples=5, **kernel_kwargs)
# Returns: samples (n_samples, n)
```
Draws functions from a GP prior using the Cholesky factor: `f = L @ u`, `u ~ N(0, I)`.

### Acquisition functions

```python
expected_improvement(mu, sigma, y_best, xi=0.0)
# Returns: ei (m,)
```
Expected Improvement (paper equations 4 and 5). Minimisation convention.

### Test functions

```python
branin(x1, x2)           # Branin-Hoo benchmark, Section 4.1 of the paper
branin_shifted(x1, x2, shift=2.0)   # Shifted auxiliary task
```

### Multi-Task GP

```python
icm_kernel(X1, t1, X2, t2, kernel_fn, B, **kernel_kwargs)
# Returns: K (n, m)
```
Intrinsic Coregionalization Model kernel (paper equation 3): `k_MT = k_input(x,x') * B[t,t']`.

```python
mtgp_posterior(X_all, t_all, y_all, X_test, t_test, kernel_fn, B, noise=1e-4, **kernel_kwargs)
# Returns: mu (m,), sigma (m,)
```
Multi-task GP posterior using the ICM kernel.

### Plotting

```python
plot_gp(ax, X_test, mu, sigma, X_train=None, y_train=None, label='GP posterior', color='steelblue', title='')
plot_ei(ax, X_test, ei, x_next=None, color='darkorange')
```

---

## Mathematical background

The notebook covers these paper equations in depth:

| Eq. | Formula | Description |
|---|---|---|
| (1) | $\mu_* = K(X_*, X)[K(X,X) + \sigma_n^2 I]^{-1}\mathbf{y}$ | GP posterior mean |
| (2) | $\Sigma_* = K(X_*,X_*) - K(X_*,X)[K(X,X)+\sigma_n^2 I]^{-1}K(X,X_*)$ | GP posterior covariance |
| (3) | $k_\text{MT}((x,t),(x',t')) = k_\text{input}(x,x') \cdot B_{tt'}$ | ICM multi-task kernel |
| (4) | $\text{EI}(x) = (\mu(x)-f^*-\xi)\Phi(Z) + \sigma(x)\phi(Z)$ | Expected Improvement |
| (5) | $Z = (\mu(x) - f^* - \xi) / \sigma(x)$ | Standardised improvement |
| (6) | $\log p(\mathbf{y}\|X,\theta) = -\tfrac{1}{2}\mathbf{y}^T K^{-1}\mathbf{y} - \tfrac{1}{2}\log\|K\| - \tfrac{n}{2}\log 2\pi$ | Log marginal likelihood |
| (7) | Same as (6) with $K_\text{MT}$ | Multi-task log marginal likelihood |
| (8) | $\alpha_\text{ES}(x) = H[p(x^*\|\mathcal{D})] - \mathbb{E}_y[H[p(x^*\|\mathcal{D}\cup\{(x,y)\})]]$ | Entropy search acquisition |
| (9) | $\alpha_\text{cost}(x,t) = \alpha_\text{ES}(x,t) / c(t)$ | Cost-adjusted acquisition |

---

## Key concepts illustrated

- **GP prior vs posterior**: how observations collapse uncertainty
- **Kernel hyperparameters**: visual intuition for lengthscale and variance
- **Exploration vs exploitation**: the EI trade-off controlled by xi
- **Transfer learning**: how off-diagonal entries in B enable cross-task information sharing
- **Cold-start advantage**: MT-BO reaches better optima with fewer expensive evaluations
- **Cost-aware querying**: when to query the cheap vs expensive task

---

## Output files

After running the full notebook, two figures are saved to disk:

- `bo_loop.png` — 15-iteration BO loop with GP posteriors and EI curves
- `equations_reference.png` — 3x3 grid of visualisations for all 9 paper equations
