"""=============================================================================
Reproduces Figure 2 in the blog post below.

Code by Gregory Gundersen (2020). See:
  http://gregorygundersen.com/blog/2020/02/04/bayesian-linear-regression/
============================================================================="""

import numpy as np
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
from   linbayes import BayesianLinearRegression
from   scipy.stats import invgamma, multivariate_normal
from   sklearn.linear_model import LinearRegression
from   sklearn.model_selection import train_test_split


np.random.seed(5)

# ------------------------------------------------------------------------------

def plot_model(ax, model, X, y, is_bayesian):

    if is_bayesian:
        x_surf = np.arange(-5, 5, 0.01)[:, np.newaxis]
        mu = model.mu_n
        cov = np.linalg.inv(model.Lambda_n)
        beta_pdf = multivariate_normal(mu, cov).pdf(x_surf)
        ax.plot(x_surf, beta_pdf, color='r', lw=2)
    else:
        ax.axvline(model.coef_, color='r', ls='--', label='OLS')
        txt = r'$\hat{\mathbf{\beta}} = %s$' % round(model.coef_[0], 2)
        ax.text(model.coef_ - 0.5, 2.65, txt, fontsize=12)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 3)
    ax.set_xticks([-1, 0, 1])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(rf'$N = {len(X)}$', fontsize=14, fontfamily='Arial',
                 fontweight='bold')


# ------------------------------------------------------------------------------

def make_regression(size):
    a_true    = 100
    b_true    = 1
    beta_true = 0.5
    var_true  = invgamma(a_true).rvs(size=1)
    var = np.repeat(var_true, size)
    X   = np.linspace(-1, 1, size*2)
    X   = np.random.choice(X, size=size, replace=False)
    Y   = multivariate_normal(beta_true * X, var).rvs()
    return X[:, np.newaxis], Y


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
fig.set_size_inches(12, 5)

X, y = make_regression(100)
first = True

for ax, n_samples in zip([ax1, ax2, ax3], [5, 20, 100]):

    X_, y_ = X[:n_samples], y[:n_samples]
    D    = 1
    mu0  = np.zeros(D)
    var0 = 0.5
    Lambda0 = 1./var0 * np.eye(D)
    a0  = 1
    b0  = 1
    mod = BayesianLinearRegression(
        dim=D, mu0=mu0, Lambda0=Lambda0, var0=var0, a0=a0, b0=b0)
    mod.fit(X_, y_)
    plot_model(ax, mod, X_, y_, True)

    mod = LinearRegression()
    mod.fit(X_, y_)
    plot_model(ax, mod, X_, y_, False)

    ax.set_xlabel(r'$\mathbf{\beta}$', fontsize=14)
    if first:
        ax.set_ylabel(
            r'$p(\mathbf{\beta} \mid \mathbf{X}, \mathbf{y}, \sigma^2)$',
            fontsize=14)
        first = False

plt.tight_layout()
plt.show()
