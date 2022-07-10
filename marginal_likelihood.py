"""=============================================================================
Reproduces Figure 3 in the blog post below.

Code by Gregory Gundersen (2020). See:
  http://gregorygundersen.com/blog/2020/02/04/bayesian-linear-regression/
============================================================================="""

import numpy as np
import matplotlib.pyplot as plt
from   linbayes import BayesianLinearRegression
from   scipy.stats import invgamma, multivariate_normal


np.random.seed(11)

# ------------------------------------------------------------------------------


def make_regression(a_true, size):
    b_true    = 1
    beta_true = 0.5
    
    var_true = invgamma(a_true).rvs(size=1)
    var = np.repeat(var_true, size)
    X = np.linspace(-1, 1, size*2)
    X = np.random.choice(X, size=size, replace=False)
    Y = multivariate_normal(beta_true * X, var).rvs()
    return X[:, np.newaxis], Y, beta_true, var_true, b_true


fig, axes = plt.subplots(1, 3)
fig.set_size_inches(15, 5)

first = True
for a_true, ax in zip([1, 5, 10], axes):

    X, Y, beta_true, var_true, b_true = make_regression(a_true, size=50)
    evids = []
    a0s   = np.arange(0, 20, 0.1)

    for a0 in a0s:

        D       = 1
        mu0     = np.zeros(D)
        Lambda0 = (1./var_true) * np.eye(D)

        model = BayesianLinearRegression(D, mu0, Lambda0, var_true, a0, b_true)
        model.fit(X, Y)
        evids.append(model.evidence())

    evids = np.array(evids)

    ax.plot(a0s, evids, c='r', lw=2)
    ax.axvline(a_true, c='b', ls='--')
    ax.set_xticks(range(0, 21, 2))
    ax.set_xlabel(r'$a_0$', fontsize=18)
    if first:
        ax.set_ylabel(r'$p(\mathbf{y} \mid \mathbf{\alpha})$', fontsize=18)
        first = False
    ax.set_yticks([])
    ax.set_xlim(0, 20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(r'$a_{true} = %s$' % a_true, fontsize=18)

plt.tight_layout()
plt.show()
