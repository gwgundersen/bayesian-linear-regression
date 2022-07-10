"""=============================================================================
Reproduces Figure 1 in the blog post below (Bishop's Figure 3.7).

Note: I resurrected this code a couple years after publishing the blog post.
      This figure does not match the blog post exactly, and I can't be bothered
      to figure out why.

Code by Gregory Gundersen (2020). See:
  http://gregorygundersen.com/blog/2020/02/04/bayesian-linear-regression/
============================================================================="""

import matplotlib.pyplot as plt
from   matplotlib import cm
import numpy as np
from   numpy.linalg import inv
from   sklearn.linear_model import LinearRegression

from   linbayes import BayesianLinearRegression


np.random.seed(52)

# ------------------------------------------------------------------------------

def get_surface(z_function):
    X      = np.linspace(-1, 1, 100)
    Y      = np.linspace(-1, 1, 100)
    XX, YY = np.meshgrid(X, Y)
    grid   = np.vstack((YY.flatten(), XX.flatten())).T
    ZZ     = z_function(grid)
    ZZ     = ZZ.reshape(XX.shape)
    return XX, YY, ZZ


def plot_density(ax, z_function):
    XX, YY, ZZ = get_surface(z_function)
    ax.contourf(XX, YY, ZZ, cmap=cm.jet, levels=1000)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zorder(-1)


def plot_betas(ax, model):
    beta = model.sample_beta(size=6)
    X = np.linspace(-1, 1, 100)
    for b in beta:
        Y = X * b[0] + b[1]
        ax.plot(X, Y, c='r', zorder=0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)


def plot_data(ax, X, y):
    ax.scatter(X[:, 0], y, c='b', zorder=1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])


def make_regression(size=50):
    X = np.linspace(-1, 1, size*2)
    X = np.random.choice(X, size=size, replace=False)
    beta0 = -0.7
    beta1 = 0.5
    noise = np.random.random(size=size) * 0.1
    y = beta1 * X + beta0 + noise
    # For the bias term.
    X = np.hstack((X[:, np.newaxis], np.ones(len(X))[:, np.newaxis]))
    return X, y


# ------------------------------------------------------------------------------

X, y = make_regression(size=30)

# Examine the model on three different data set sizes.
X1, y1 = X[:1, :], y[:1]
X2, y2 = X[:5, :], y[:5]
X3, y3 = X, y

D       = X.shape[1]
mu0     = np.zeros(D)
Lambda0 = np.eye(D)
var0    = 0.1
a0      = 3.0
b0      = 1.0

fig, row_axes = plt.subplots(2, 4, sharex=True, sharey=True)
fig.set_size_inches(20, 7.5)

model = BayesianLinearRegression(dim=D, mu0=mu0, Lambda0=Lambda0, var0=var0,
                                 a0=a0, b0=b0)
axes1 = row_axes.T[0]
plot_density(axes1[0], model.prior)
plot_betas(axes1[1], model)
for ax in axes1:
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

for X_, y_, axes in zip([X1, X2, X3], [y1, y2, y3], row_axes.T[1:]):

    ax1, ax2 = axes
    model.fit(X_, y_)

    plot_betas(ax2, model)
    plot_data(ax2, X_, y_)
    plot_density(ax1, model.posterior)

for axes in row_axes.T:
    for ax in axes:
        ax.set(aspect='equal')
        ax.tick_params(axis='both', which='major', labelsize=18)

for i, ax in enumerate(row_axes[0]):
    if i == 0:
        ax.set_ylabel(r'$\beta_1$', fontsize=18)
        ax.yaxis.labelpad = -5
    ax.set_xlabel(r'$\beta_0$', fontsize=18)
    ax.xaxis.labelpad = 10

for i, ax in enumerate(row_axes[1]):
    if i == 0:
        ax.set_ylabel(r'$y$', fontsize=18)
        ax.yaxis.labelpad = -5
    ax.set_xlabel(r'$x$', fontsize=18)
    ax.xaxis.labelpad = 5


pad = 5
rows = ['prior or\nposterior', 'data\nspace']
xys = [(-1, 0.5), (-1, 0.5)]
for ax, row, xy in zip(row_axes[:, 0], rows, xys):
    ax.annotate(row, xy=xy, xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                ha='right', va='center', fontsize=18)

plt.tight_layout()
plt.gcf().subplots_adjust(left=0.13)
plt.show()
