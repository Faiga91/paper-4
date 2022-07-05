import numpy as np
#import pymc3 as pm
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import multiprocessing as mp 

import logging
logger = logging.getLogger("pymc3")
logger.propagate = False

""" 
def estimate_posterior_probability(Temperature_value, C):

    K = len(C) # The number of outcomes 
    alphas = 0.1 * np.ones(K) # hyperparameters (initially all equal)
    ctx = mp.get_context('fork')
    with pm.Model() as model:
        # Parameters of the Multinomial are from a Dirichlet
        parameters = pm.Dirichlet('parameters', a=alphas, shape=K)
        # Observed data is from a Multinomial distribution
        observed_data = pm.Multinomial(
            'observed_data', n=C.sum(), p=parameters, shape=K, observed=C)

    with model:
        # Sample from the posterior
        trace = pm.sample(draws=1000, chains=2, tune=500, 
                        discard_tuned_samples=True, cores=1)

    trace_df = pd.DataFrame(trace['parameters'], columns = Temperature_value)
    pvals = trace_df.iloc[:, :K].mean(axis = 0)
    return pvals
"""

def get_metrics(X, X_r):
    X_mean = X.values.mean()
    NMAE = mean_absolute_error(X, X_r, multioutput="uniform_average") / X_mean 
    RMSE = mean_squared_error(X,X_r, squared=False) 
    return (NMAE, RMSE)