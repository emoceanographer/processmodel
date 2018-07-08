## import dependencies
import numpy as np
import matplotlib
matplotlib.use('agg') # python install issue fix github.com/matplotlib/matplotlib.issues/9017
import matplotlib.pyplot as plt

import pymc3 as pm
from scipy import optimize

# initialize random number generator
np.random.seed(123)

# true parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# size of dataset
size = 100

# predicator variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

# apparently makes plots, but I am not sure if these displayed?
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
axes[0].scatter(X1, Y)
axes[1].scatter(X2, Y)
axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set

# builds our model
basic_model = pm.Model() # creates a container for the model

with basic_model: # everything in here is added to the model behind the screens

	# priors for unknown parameters
	alpha = pm.Normal('alpha', mu=0, sd=10) # first arg is name of RV; match the name of the var 
	# it is assigned to; the rest are the hyperparamters of the model (Beta, Exponential, Categorical)
	beta = pm.Normal('beta', mu=0, sd=10, shape=2)
	sigma = pm.HalfNormal('sigma', sd=1)

	# Expected values of outcome
	mu = alpha + beta[0]*X1 + beta[1]*X2
		# RVS can be added, subtracted, divided, multiplied, and indexed into to creat new RVs

	# Likelihood (sampling distribution) of observations
	Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y) # the Y says it is an observed stochastic
		# this is apparently a sampling distribution of outcomes
	
# runs a maximum a posteriori methods
map_estimate = pm.find_MAP(model=basic_model, fmin=optimize.fmin_powell)
print(map_estimate)

# runs an MCMC
with basic_model:
	# draw 500 posterior samples
	trace = pm.sample(500)
print(pm.summary(trace))