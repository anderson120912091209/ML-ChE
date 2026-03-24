"""
utils.py — Helper functions for Multi-Task Bayesian Optimization notebook.
All math references cite Swersky, Snoek, Adams (NeurIPS 2013).
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('seaborn-v0_8-whitegrid')

# ── Kernels ──────────────────────────────────────────────────────────────────

def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """
    Radial Basis Function (squared exponential) kernel.
    k(x,x') = variance * exp(-0.5 * ||x-x'||^2 / lengthscale^2)

    Args:
        X1: (n, d) array
        X2: (m, d) array
        lengthscale: length scale parameter l
        variance: signal variance sigma_f^2
    Returns:
        K: (n, m) kernel matrix
    """
    # Compute squared distances ||x-x'||^2
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]          # (n, m, d)
    sq_dist = np.sum(diff ** 2, axis=-1)              # (n, m)
    return variance * np.exp(-0.5 * sq_dist / lengthscale ** 2)


def matern52_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """
    Matern 5/2 kernel — the kernel used in the paper.
    k(x,x') = variance * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
    where r = ||x - x'||

    Paper reference: Section 2 (kernel choice)

    Args:
        X1: (n, d) array
        X2: (m, d) array
        lengthscale: length scale l
        variance: signal variance sigma_f^2
    Returns:
        K: (n, m) kernel matrix
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)
    r = np.sqrt(np.maximum(sq_dist, 0))              # Euclidean distance, clamp to avoid sqrt(neg)
    sqrt5 = np.sqrt(5.0)
    z = sqrt5 * r / lengthscale
    return variance * (1.0 + z + z**2 / 3.0) * np.exp(-z)


# ── GP Inference ─────────────────────────────────────────────────────────────

def gp_posterior(X_train, y_train, X_test, kernel_fn, noise=1e-4, **kernel_kwargs):
    """
    Compute GP posterior mean and variance.

    Paper equations (1) and (2):
      mu*(X*) = K(X*,X) [K(X,X) + sigma_n^2 I]^-1 y          ... (1)
      Sigma*(X*,X*) = K(X*,X*) - K(X*,X)[K(X,X)+sigma_n^2 I]^-1 K(X,X*)  ... (2)

    Args:
        X_train: (n, d) training inputs
        y_train: (n,) training targets
        X_test:  (m, d) test inputs
        kernel_fn: callable(X1, X2, **kwargs) -> (n,m) kernel matrix
        noise: observation noise variance sigma_n^2
        **kernel_kwargs: passed to kernel_fn
    Returns:
        mu: (m,) posterior mean
        sigma: (m,) posterior standard deviation
        K_inv: (n,n) inverse of noisy training kernel (for reuse)
    """
    n = len(X_train)

    # K(X,X) + sigma_n^2 * I  — noisy training covariance
    K_XX = kernel_fn(X_train, X_train, **kernel_kwargs)          # (n, n)
    K_XX_noisy = K_XX + noise * np.eye(n)                        # add noise on diagonal

    # K(X*,X) — cross-covariance between test and train
    K_sX = kernel_fn(X_test, X_train, **kernel_kwargs)           # (m, n)

    # K(X*,X*) — test covariance
    K_ss = kernel_fn(X_test, X_test, **kernel_kwargs)            # (m, m)

    # Cholesky decomposition for numerical stability
    L = np.linalg.cholesky(K_XX_noisy + 1e-9 * np.eye(n))       # L L^T = K_XX_noisy

    # Solve K_XX_noisy^-1 y via back-substitution (more stable than explicit inverse)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))   # K^-1 y

    # Equation (1): posterior mean
    mu = K_sX @ alpha                                             # (m,)

    # Solve K^-1 K(X,X*) via back-substitution
    v = np.linalg.solve(L, K_sX.T)                              # (n, m)

    # Equation (2): posterior variance (diagonal only)
    var = np.diag(K_ss) - np.sum(v**2, axis=0)                  # (m,)
    var = np.maximum(var, 1e-9)                                   # clamp negatives from numerics

    # K_inv for external use
    K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))

    return mu, np.sqrt(var), K_inv


def sample_gp_prior(X, kernel_fn, n_samples=5, **kernel_kwargs):
    """
    Draw samples from a GP prior f ~ GP(0, k).

    Args:
        X: (n,) or (n,1) input locations
        kernel_fn: kernel function
        n_samples: number of functions to sample
        **kernel_kwargs: kernel hyperparameters
    Returns:
        samples: (n_samples, n) array of function values
    """
    X = np.atleast_2d(X).T if X.ndim == 1 else X
    K = kernel_fn(X, X, **kernel_kwargs)
    K += 1e-9 * np.eye(len(X))                                   # jitter for numerical stability
    L = np.linalg.cholesky(K)                                    # Cholesky factor
    u = np.random.randn(len(X), n_samples)                       # standard normals
    return (L @ u).T                                             # (n_samples, n)


# ── Acquisition Functions ─────────────────────────────────────────────────────

def expected_improvement(mu, sigma, y_best, xi=0.0):
    """
    Expected Improvement acquisition function.

    Paper equations (4) and (5):
      EI(x) = (mu(x) - f* - xi) * Phi(Z) + sigma(x) * phi(Z)    ... (4)
      Z = (mu(x) - f* - xi) / sigma(x)                           ... (5)
    where Phi is the CDF and phi the PDF of standard normal.

    We MINIMISE f, so improvement = f* - f(x).

    Args:
        mu: (m,) posterior mean
        sigma: (m,) posterior std dev
        y_best: current best (minimum) observed value f*
        xi: exploration-exploitation trade-off parameter
    Returns:
        ei: (m,) expected improvement values
    """
    # Equation (5): standardised improvement
    Z = (y_best - mu - xi) / (sigma + 1e-9)                     # add eps to avoid div-by-zero

    # Equation (4): expected improvement
    ei = (y_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z) # (m,)
    ei[sigma < 1e-9] = 0.0                                       # no improvement where variance = 0
    return ei


# ── Test Functions ─────────────────────────────────────────────────────────────

def branin(x1, x2, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    """
    Branin-Hoo function — standard BO benchmark used in the paper (Section 4.1).
    Global minima at (-pi, 12.275), (pi, 2.275), (9.42478, 2.475), value ~= 0.397887.
    Domain: x1 in [-5,10], x2 in [0,15].
    """
    return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s


def branin_shifted(x1, x2, shift=2.0):
    """Shifted variant of Branin used as auxiliary task in Section 4.1."""
    return branin(x1 + shift, x2 + shift)


# ── Multi-Task GP ─────────────────────────────────────────────────────────────

def icm_kernel(X1, t1, X2, t2, kernel_fn, B, **kernel_kwargs):
    """
    Intrinsic Coregionalization Model (ICM) kernel.

    Paper equation (3):
      k_MT((x,t),(x',t')) = k_input(x,x') * k_task(t,t')

    The full multi-task covariance over all (x,t) pairs is:
      K_MT = K_input (kronecker) B
    where B is the task covariance matrix (positive semi-definite).

    Args:
        X1: (n,) input locations for first set
        t1: (n,) integer task indices for first set (0-indexed)
        X2: (m,) input locations for second set
        t2: (m,) integer task indices for second set
        kernel_fn: input space kernel
        B: (T, T) task covariance matrix (must be PSD)
        **kernel_kwargs: passed to kernel_fn
    Returns:
        K: (n, m) multi-task kernel matrix
    """
    X1 = np.atleast_2d(X1).T if np.array(X1).ndim == 1 else X1
    X2 = np.atleast_2d(X2).T if np.array(X2).ndim == 1 else X2

    K_input = kernel_fn(X1, X2, **kernel_kwargs)                 # input-space kernel (n, m)
    # Task kernel: B[t_i, t_j] for each pair
    K_task = B[np.array(t1)[:, None], np.array(t2)[None, :]]    # (n, m) task covariances

    return K_input * K_task                                      # element-wise: eq (3)


def mtgp_posterior(X_all, t_all, y_all, X_test, t_test, kernel_fn, B, noise=1e-4, **kernel_kwargs):
    """
    Multi-task GP posterior.  Same equations (1)+(2) but using ICM kernel.

    Args:
        X_all:  (n,) all training inputs
        t_all:  (n,) task indices for training data
        y_all:  (n,) training targets
        X_test: (m,) test inputs
        t_test: (m,) task indices for test points
        kernel_fn: input kernel
        B: (T,T) task covariance matrix
        noise: scalar noise variance
        **kernel_kwargs: kernel hyperparams
    Returns:
        mu: (m,) posterior mean
        sigma: (m,) posterior std dev
    """
    n = len(X_all)

    K_train = icm_kernel(X_all, t_all, X_all, t_all, kernel_fn, B, **kernel_kwargs)
    K_train += noise * np.eye(n)

    K_cross = icm_kernel(X_test, t_test, X_all, t_all, kernel_fn, B, **kernel_kwargs)
    K_test  = icm_kernel(X_test, t_test, X_test, t_test, kernel_fn, B, **kernel_kwargs)

    L = np.linalg.cholesky(K_train + 1e-9 * np.eye(n))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_all))

    mu = K_cross @ alpha
    v  = np.linalg.solve(L, K_cross.T)
    var = np.diag(K_test) - np.sum(v**2, axis=0)
    var = np.maximum(var, 1e-9)

    return mu, np.sqrt(var)


# ── Plotting Utilities ────────────────────────────────────────────────────────

def plot_gp(ax, X_test, mu, sigma, X_train=None, y_train=None,
            label='GP posterior', color='steelblue', title=''):
    """Plot GP mean +/- 2sigma confidence band."""
    ax.plot(X_test, mu, color=color, lw=2, label=f'{label} mean')
    ax.fill_between(X_test, mu - 2*sigma, mu + 2*sigma,
                    alpha=0.2, color=color, label='+-2sigma')
    if X_train is not None and y_train is not None:
        ax.scatter(X_train, y_train, c='black', zorder=5, s=40, label='Observations')
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)


def plot_ei(ax, X_test, ei, x_next=None, color='darkorange'):
    """Plot Expected Improvement curve."""
    ax.plot(X_test, ei, color=color, lw=2)
    ax.fill_between(X_test, 0, ei, alpha=0.3, color=color)
    if x_next is not None:
        ax.axvline(x_next, color='red', ls='--', lw=1.5, label=f'Next query: {x_next:.2f}')
        ax.legend(fontsize=9)
    ax.set_ylabel('EI', fontsize=11)
    ax.set_title('Expected Improvement', fontsize=13)
