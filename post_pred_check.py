"""=============================================================================
Reproduces Figure 5 in the blog post below.

Code by Gregory Gundersen (2020). See:
  http://gregorygundersen.com/blog/2020/02/04/bayesian-linear-regression/
============================================================================="""

from   collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from   scipy.stats import multivariate_normal, multivariate_t


np.random.seed(3)

# -----------------------------------------------------------------------------

def linear_bayes(X, y, mu0, Lambda0, a0, b0):
    """Estimate model parameters for Bayesian linear regression.

    See: https://en.wikipedia.org/wiki/Bayesian_linear_regression.
    """
    N = len(X)

    # Compute Lambda_n and mu_n.
    Lambda_n = X.T @ X + Lambda0
    mu_n     = (1./Lambda_n) * (Lambda0 * mu0 + X.T @ y)

    # Compute a_n and b_n.
    a_n = a0 + 0.5 * N
    A   = y.T @ y
    B   = mu0 * Lambda0 * mu0
    C   = mu_n * Lambda_n * mu_n
    b_n = b0 + 0.5 * (A + B - C)

    BLR = namedtuple('BLR', ['mu_n', 'Lambda_n', 'a_n', 'b_n'])
    return BLR(mu_n, Lambda_n, a_n, b_n)

# -----------------------------------------------------------------------------

beta = 0.5

fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(10, 5)

# Plot generating function.
X      = np.arange(-2, 202)
inds   = np.random.permutation(len(X))[:100]
X      = X[inds]
noise  = np.random.normal(size=len(X)) * 5
Y_true = beta * X + noise

model = linear_bayes(X, Y_true, mu0=0, Lambda0=0.2, a0=1, b0=1)

# Multivariate t mean.
X_  = X[:, np.newaxis]
mu  = X_ @ np.array([model.mu_n])

# Multivariate t shape.
L   = np.array([1./model.Lambda_n])[:, np.newaxis]
V   = (X_ @ L @ X_.T) + np.eye(len(X_))
Sig = (model.b_n / model.a_n) * V

# Multivariate t degrees of freedom.
nu  = int(2 * model.a_n)

Y = multivariate_t(loc=mu, shape=Sig, df=nu).rvs(size=1000)

ax0.hist(Y.mean(axis=1), bins=50, ls='dashed', lw=3, fc=(1, 0, 0, 0.4))
ax0.axvline(Y_true.mean(), color='b', ls='--')
ax0.tick_params(axis='both', which='major', labelsize=14)
ax0.set_title(r'$\mu_{y}$', fontsize=14, fontfamily='Arial', fontweight='bold')
ax0.set_ylabel('Frequency', fontsize=14)
ax0.set_xlabel(r'$\mathbf{y}$', fontsize=14)

ax1.hist(Y.var(axis=1), bins=50, ls='dashed', lw=3, fc=(1, 0, 0, 0.4))
ax1.axvline(Y_true.var(), color='b', ls='--')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_title(r'$\sigma^2_{y}$', fontsize=14, fontfamily='Arial', fontweight='bold')
ax1.set_xlabel(r'$\mathbf{y}$', fontsize=14)

plt.tight_layout()
plt.show()
