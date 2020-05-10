from BetaBinomial import BetaBinomial

# create a model
a = 10
b = 20
model = BetaBinomial(a=a, b=b)

# draw samples from the model
samples = model.gen_samples(n=30, r=30)
print('Samples drawn from a={}, b={} beta binomial model: '.format(a, b))
print(samples)

# estimate underlying hyper-params of the samples
a_hat, b_hat = model.estimate_priors(samples)
print('Estimated parameters: a={:.2f}, b={:.2f}'.format(a_hat, b_hat))

# now lets learn from differently modeled data and see how it affects the model
new_samples = model.gen_samples(n=40, r=30, a=50, b=10)
model.update_model(new_samples)
print('Updated priors: a={}, b={}'.format(model.a, model.b))

# visualize the resulting pmf
model.plot_posterior_pmf()
