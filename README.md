# Bayesian Beta Binomial model

A Python implementation of Bayesian Beta Binomial model.
Heavy-load computation done with SymPy.

The model allows you to generate samples with an underlying Beta Binomial distribution.
Generated samples have the shape n_samples * r_trials. Each sample has r_trials of Bernoulli(p), where p is drawn from Beta(a,b) distribution. (a,b) are the hyper-parameters of the model.

The hyper-parameters can be estimated from samples. Alternatively, new samples can be fed to the model so the hyper-parameters will be updated according to the new evidence.

Use examples.py to run a quick test using the model.


    References
    ==========

    [1] https://en.wikipedia.org/wiki/Beta-binomial_distribution
    [2] Quintana, F.A. and Tam, W.K., 1996. Bayesian estimation of
        beta-binomial models by simulating posterior densities.
        Journal of the Chilean Statistical Society, 13(1-2), pp.43-56.
