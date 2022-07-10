"""=============================================================================
Bayesian linear regression.

Code by Gregory Gundersen (2020). See:
  http://gregorygundersen.com/blog/2020/02/04/bayesian-linear-regression/
============================================================================="""

import numpy as np
from   numpy import (exp,
                     log,
                     sqrt)
from   numpy.linalg import (inv,
                            slogdet)
from   scipy.special import loggamma, gammaln
from   scipy.stats import multivariate_normal


# ------------------------------------------------------------------------------

class BayesianLinearRegression:

    def __init__(self, dim, mu0, Lambda0, var0, a0, b0):
        """Initialize Bayesian linear regression with a normal prior on the
        weights and an inverse–gamma prior on the variance.
        """
        assert(mu0.size == dim)
        assert(Lambda0.shape == (dim, dim))
        assert(np.isscalar(a0))
        assert(np.isscalar(b0))

        self.var0    = var0
        self.dim     = dim
        self.mu0     = mu0
        self.Lambda0 = Lambda0
        self.a0      = a0
        self.b0      = b0
        self.fitted  = False

    def fit(self, X, y):
        """Fit model parameters.
        """
        self.N, D = X.shape
        assert(D == self.dim)

        # Compute Lambda_n and mu_n.
        Lambda_n = X.T @ X + self.Lambda0
        mu_n     = inv(Lambda_n) @ (self.Lambda0 @ self.mu0 + X.T @ y)

        # Compute a_n and b_n.
        a_n = self.a0 + 0.5 * self.N
        A   = y.T @ y
        B   = self.mu0.T @ self.Lambda0 @ self.mu0
        C   = mu_n.T @ Lambda_n @ mu_n
        b_n = self.b0 + 0.5 * (A + B - C)

        self.mu_n     = mu_n
        self.Lambda_n = Lambda_n
        self.a_n      = a_n
        self.b_n      = b_n
        self.fitted   = True
        return self

    def posterior(self, X):
        """Evaluate functional form of posterior.
        """
        l = self.likelihood(X)
        p = self.prior(X)
        e = self.evidence()
        return (l*p)/e

    def likelihood(self, X):
        """Evaluate likelihood.
        """
        assert self.fitted
        cov = self.var0 * inv(self.Lambda_n)
        return multivariate_normal(self.mu_n, cov).pdf(X)

    def prior(self, X):
        """Evaluate functional form of prior.
        """
        mu = np.zeros(self.mu0.shape)
        cov = self.var0 * inv(self.Lambda0)
        A = multivariate_normal(self.mu0, cov).logpdf(X)
        B = invgamma_logpdf(self.a0, self.b0, self.var0)
        return exp(A + B)

    def sample_beta(self, size=1):
        """Sample linear model's coefficients ("betas").
        """
        if self.fitted:
            mu  = self.mu_n
            cov = self.var0 * inv(self.Lambda_n)
        else:
            mu  = self.mu0
            cov = self.var0 * inv(self.Lambda0)
        return multivariate_normal(mu, cov).rvs(size=size)

    def predict(self, X):
        """Predict model's posterior mean given new observations.
        """
        assert self.fitted
        _, D = X.shape
        assert(D == self.dim)
        # Predict the mean of the multivariate t-distribution.
        return X @ self.mu_n

    def evidence(self):
        """Compute marginal likelihood ("evidence") of model.
        """
        assert self.fitted
        norm     = (-self.N/2.) * log(2*np.pi)
        logdet, sign = slogdet(self.Lambda0)
        A = sign * logdet
        logdet, sign = slogdet(self.Lambda_n)
        B = sign * logdet
        L_term   = 0.5 * (A - B)
        b_a_term = self.a0 * log(self.b0) - self.a_n * log(self.b_n)
        g_term   = gammaln(self.a_n) - gammaln(self.a0)
        log_evid = norm + L_term + b_a_term + g_term
        return exp(log_evid)


# ------------------------------------------------------------------------------

def invgamma_logpdf(a, b, x):
    return (a*log(b)) - gammaln(a) + ((a+1)*log(1./x)) - (b/x)
