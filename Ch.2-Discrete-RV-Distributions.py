import numpy as np
import matplotlib.pyplot as plt

# Importing the necessary modules from SciPy for probability distributions
from scipy import stats
from scipy.stats import binom           # Binomial distribution
from scipy.stats import poisson         # Poisson distribution

# Version information for loaded libraries
print("Numpy version =", np.version.version)
import scipy
print("Scipy version =", scipy.__version__)

#------ Binomial Distribution Parameters ------
n = 15  # Number of trials
p = 0.2  # Probability of success in each trial

# Range of possible values from 0 to n
r_values = np.arange(0, n+1, 1)

# Probability mass function (PMF) and cumulative distribution function (CDF) for the values
pmf = binom.pmf(r_values, n, p)
cdf = binom.cdf(r_values, n, p)

# Rounding the PMF and CDF values for better readability
round_pmf = [round(num, 3) for num in pmf]
round_cdf = [round(num, 3) for num in cdf]
print("values =", r_values, "\n")
print("pmf =", round_pmf, "\n")
print("cdf =", round_cdf, "\n")

# Mean, variance, and standard deviation for the binomial distribution
mean = binom.mean(n, p)
var  = binom.var(n, p)
std  = binom.std(n, p)
print("mean = %.2f, variance = %.2f, sd = %.2f \n" % (mean, var, std))

# Plotting the PMF and CDF
fig = plt.figure(figsize=(12,4))
plt.subplot(121)
plt.bar(r_values, pmf)
plt.title("pmf of Bin(%d, %.1f)" % (n, p))

plt.subplot(122)
plt.bar(r_values, cdf)
plt.title("cdf of Bin(%d, %.1f)" % (n, p))
plt.show()

# Example calculations using the CDF
Q1_cdf = cdf[8]
Q2_cdf = cdf[8] - cdf[7]
Q3_cdf = 1 - cdf[7]
Q4_cdf = cdf[7] - cdf[3]

print("cdf[8] = %.3f" % cdf[8])
print("cdf[7] = %.3f" % cdf[7])
print("cdf[3] = %.3f" % cdf[3])
print("Q1 by cdf = %.3f" % Q1_cdf)
print("Q2 by cdf = %.3f" % Q2_cdf)
print("Q3 by cdf = %.3f" % Q3_cdf)
print("Q4 by cdf = %.3f" % Q4_cdf, "\n")

Q2_pmf = pmf[8]
Q4_pmf = pmf[4] + pmf[5] + pmf[6] + pmf[7]
print("Q2 by pmf = %.3f" % Q2_pmf)
print("Q4 by pmf = %.3f" % Q4_pmf)

#------ Poisson Distribution Parameters ------
mu = 4.5  # Mean number of successes

#------ Some possible values 0-10 (no upper bound for Poisson)
n = 10
r_values = np.arange(0, n+1, 1)

# PMF and CDF calculations for Poisson distribution
pmf = poisson.pmf(r_values, mu)
cdf = poisson.cdf(r_values, mu)

# Rounding off PMF and CDF values
round_pmf = [round(num, 4) for num in pmf]
round_cdf = [round(num, 4) for num in cdf]
print("pmf =", round_pmf, "\n")
print("cdf =", round_cdf, "\n")

# Mean, variance, and standard deviation for the Poisson distribution
mean = poisson.mean(mu)
var  = poisson.var(mu)
std  = poisson.std(mu)
print("mean = %.2f, variance = %.2f, sd = %.2f \n" % (mean, var, std))

# Plotting PMF and CDF for Poisson distribution
fig = plt.figure(figsize=(12,4))
plt.subplot(121)
plt.bar(r_values, pmf)
plt.title("pmf of Poi(%.1f)" % mu)

plt.subplot(122)
plt.bar(r_values, cdf)
plt.title("cdf of Poi(%.1f)" % mu)
plt.show()
