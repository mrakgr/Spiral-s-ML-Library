import numpy as np

def sample_int_lognormal(mean, sigma, n):
    mu = np.log(mean) - 0.5 * sigma**2
    x = np.random.lognormal(mu, sigma, size=n)
    f = np.floor(x)
    r = np.random.rand(n)
    return f + (r < (x - f))

def sample_int_lognormal_my(mean, sigma, n):
    mu = np.log(mean) - 0.5 * sigma**2
    x = np.exp(np.random.normal(mu, sigma, size=n))
    f = np.floor(x)
    r = np.random.rand(n)
    return f + (r < (x - f))

# Example:
samples = sample_int_lognormal_my(mean=10, sigma=0.6, n=100)
print(np.mean(samples)) # Should be close to 20
print(samples) # Should be close to 20
