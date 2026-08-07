import numpy as np

def gaussian_samples(X, n_samples, sigma):
    """Sample from the empirical multivariate Gaussian distribution"""
    n_features = X.shape[-1]
    mu = X
    samples = np.random.normal(mu, sigma, size=(n_samples, n_features))
    return samples