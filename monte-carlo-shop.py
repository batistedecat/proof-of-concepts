import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 100000

samples = rng.uniform(0,1,n)
mu = np.mean(samples)
sigma = np.std(samples)

hist = np.histogram(samples, bins = 10)

print(mu, sigma)


