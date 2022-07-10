"""=============================================================================
Reproduces Figure 4 in the blog post below (Bishop's Figure 3.8).

Note: I resurrected this code a couple years after publishing the blog post.
      This figure uses a function `linear_bayes` rather than the class in
      `linbayes.py`. I suspect I started with this function before generalizing
      into a class and then didn't bother to update the code here.

Code by Gregory Gundersen (2020). See:
  http://gregorygundersen.com/blog/2020/02/04/bayesian-linear-regression/
============================================================================="""

from   collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from   scipy.stats import multivariate_normal

np.random.seed(20)


# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------

beta = 0.5

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True)
fig.set_size_inches(15, 5)

# Plot generating function.
X = np.arange(-2, 202)
Y = beta * X
ax0.plot(X, Y, c='b', ls='--', label=r'True function')
ax1.plot(X, Y, c='b', ls='--')
ax2.plot(X, Y, c='b', ls='--')

# Shuffle data.
inds  = np.random.permutation(len(X))[:30]
X     = X[inds]
noise = np.random.normal(size=len(X)) * 5
Y     = beta * X + noise

first = True
for i, ax in zip([2, 5, 30], [ax0, ax1, ax2]):

    # Plot data.
    X_ = X[:i]
    Y_ = Y[:i]
    ax.scatter(X_, Y_, c='b')

    # Fit model.
    model = linear_bayes(X_, Y_, mu0=0, Lambda0=1, a0=1, b0=1)

    # Plot posterior predictive.
    X_surf   = np.arange(-2, 202)[:, np.newaxis]
    Y_surf   = np.empty(204)
    std_surf = np.empty(204)
    # for i, x in enumerate(X_surf):

    # Multivariate t mean.
    mu  = X_surf @ np.array([model.mu_n])
    
    # Multivariate t shape.
    L   = np.array([1./model.Lambda_n])[:, np.newaxis]
    V   = (X_surf @ L @ X_surf.T)
    Sig = (model.b_n / model.a_n) * V

    # Multivariate t degrees of freedom.
    nu  = 2 * model.a_n

    # Predict mean.
    Y_surf = mu
    # Standard deviation is square root of multvariate t variance.
    var = (nu) / (nu - 2) * np.diag(Sig)
    std_surf = np.sqrt(var)

    label = r'Mean of $p(\mathbf{\hat{y}} \mid \mathbf{y})$'
    ax.plot(X_surf, Y_surf, c='r', label=label)
    ax.fill_between(
        X_surf.squeeze(),
        Y_surf - 2*std_surf,
        Y_surf + 2*std_surf,
        color='r',
        zorder=-1,
        alpha=0.1)

    ax.set_xlim(0, 200)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title(rf'$N = {len(X_)}$', fontsize=18, fontfamily='Arial',
                 fontweight='bold')
    ax.set_xlabel(r'$\mathbf{x}$', fontsize=18)

    if first:
        first = False
        ax.set_ylabel(r'$\mathbf{y}$', fontsize=18)

plt.tight_layout()
plt.show()
