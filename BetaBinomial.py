import numpy as np

from sympy import loggamma as lg, exp
from sympy import symbols
from sympy.stats import density, E, Beta, Binomial
from sympy.stats import BetaBinomial as BetaBn
from sympy.stats import sample as sym_sample
from sympy.plotting import plot as sym_plot


class BetaBinomial():
    """Bayesian Beta Binomial model

    a, b: hyper parameters. definite positive integers
    n: number of samples
    r: number of trials per sample (const across samples)
    k: number of positive trials per sample

    samples: 2d numpy array [row=samples, cols=features]


    References
    ==========

    [1] https://en.wikipedia.org/wiki/Beta-binomial_distribution
    [2] Quintana, F.A. and Tam, W.K., 1996. Bayesian estimation of
        beta-binomial models by simulating posterior densities.
        Journal of the Chilean Statistical Society, 13(1-2), pp.43-56.
    """

    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b

    def update_model(self, samples):
        # update model hyper-parameters with new data
        k, r, n = self._get_k_r_n_from_samples(samples)
        self.a += k.sum()
        self.b += n*r - k.sum()

    def gen_samples(self, n, r, a=None, b=None):
        if not a: a = self.a
        if not b: b = self.b

        samples = np.empty((0, r))

        for _ in range(n):
            p = Beta('p', a, b)
            B = Binomial('B', 1, sym_sample(p))
            sample = []
            for _ in range(r):
                # TODO: change to sample a complete vector at once
                sample.append(sym_sample(B))
            samples = np.append(samples, [sample], axis=0)
        return samples

    def estimate_priors(self, samples):
        k, r, n = self._get_k_r_n_from_samples(samples)
        m1 = float(np.sum(k) / n)
        m2 = float(np.dot(k, k) / n)
        denom = (n * (m2/m1 - m1 - 1) + m1)
        a_hat = (r*m1 - m2) / denom
        b_hat = (r - m1) * (r - m2/m1) / denom
        return a_hat, b_hat

    def _get_k_r_n_from_samples(self, samples):
        k = samples.sum(axis=1)
        r = samples.shape[1]
        n = samples.shape[0]
        return k, r, n

    def expectation(self, n=1):
        X = BetaBn('X', n, self.a, self.b)
        return E(X).evalf()

    def calc_compound_distribution(self, n_trials):
        """calculate f(k | r,a,b) using symbolic expressions

        redundant, given BetaBn is available in SymPy
        """
        a, b, r, k = symbols('a b r k')
        log_fun = (lg(r + 1) + lg(k + a) + lg(r - k + b) + lg(a + b)) - \
                  (lg(k + 1) + lg(r - k + 1) + lg(r + a + b) + lg(a) + lg(b))

        fun = exp(log_fun)
        return fun

    def plot_posterior_pmf(self, n=10):
        # note: a "continuous pdf" is displayed, instead of
        # a proper probability mass function
        x = symbols("x")
        X = BetaBn('X', n, self.a, self.b)
        sym_plot(density(X)(x),
                 xlim=[0, n],
                 title='BetaBinomial pmf\nn={}, a={}, b={}, mean={:.2f}'.format(
                     n, self.a, self.a, self.expectation(n))
                 )
