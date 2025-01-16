# Import necessary libraries for numerical computation, data visualization, and statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Print versions of libraries for verification of compatibility
print("Numpy version =", np.version.version)
print("Seaborn version =", sns.__version__)

# Import the scipy library and its version
import scipy
print("Scipy version =", scipy.__version__)

# Import statistical functions and distributions from scipy
from scipy import stats
from scipy.stats import uniform         # Uniform distribution
from scipy.stats import norm            # Normal distribution
from scipy.stats import expon           # Exponential distribution

# Import sympy for symbolic mathematics and print its version
import sympy
print("Sympy version =", sympy.__version__)

# Import specific functions and symbols from sympy
from sympy import *

# Declare symbolic variables and functions
x  = Symbol('x')          # Symbolic variable x
y  = Symbol('y')          # Symbolic variable y
fy = Function('fy')(y)    # Symbolic function fy defined in terms of y
fy = (1/8) + (3*y/8)      # Define fy as a linear function of y

# Define integrals of fy for cumulative distribution functions (CDFs)
cdf_x_nobound = Integral(fy, y)            # General integral of fy with no bounds
cdf_x_bound   = Integral(fy, (y, 0, x))    # Integral of fy from 0 to x

# Pretty print the bounded CDF integral
pprint(cdf_x_bound, use_unicode=True)       

# Solve the integral for the bounded CDF
cdf_x_bound.doit()

# Substitute values into the solved integral and store in Fx
Fx = cdf_x_bound.doit()

# Function to compute percentile values of a given distribution
def getPercentileValues(dist, loc, scale):
    r_values = np.linspace( dist.ppf(0.01, loc, scale),  # Percentile 1
                            dist.ppf(0.99, loc, scale),  # Percentile 99
                            100                          # 100 values evenly spaced
                          )
    return r_values

# Substitute specific values into the CDF and calculate differences
upper = Fx.subs(x, 1.5)  # Substitute x = 1.5
lower = Fx.subs(x, 1)    # Substitute x = 1
print("F(1.5) - F(1) = %.4f" % (upper-lower))

# Calculate the integral of fy over a specific range (1 to 1.5)
result = Integral(fy, (y, 1, 1.5)).doit()
print("Result = %.4f" % result)

# Define parameters for a uniform distribution
A = 0          # Lower bound of the uniform distribution
B = 360        # Upper bound of the uniform distribution
loc = A        # Location parameter for uniform distribution
scale = B - A  # Scale (range) of the uniform distribution

# Calculate and print probabilities using the uniform CDF
cdf_90  = uniform.cdf(90, loc, scale)    # P(X <= 90)
cdf_180 = uniform.cdf(180, loc, scale)   # P(X <= 180)
Q1 = cdf_180 - cdf_90                    # Probability between 90 and 180
print("P(X <= 90)  = %.3f" % cdf_90)
print("P(X <= 180) = %.3f" % cdf_180)
print("Q1 = %.3f \n" % Q1)

# Calculate additional probabilities for different ranges
cdf_0   = uniform.cdf(0, loc, scale)     # P(X <= 0)
cdf_270 = uniform.cdf(270, loc, scale)   # P(X <= 270)
cdf_360 = uniform.cdf(360, loc, scale)   # P(X <= 360)
Q2 = (cdf_90 - cdf_0) + (cdf_360 - cdf_270)  # Combined probabilities
print("P(X <= 0)   = %.3f" % cdf_0)
print("P(X <= 90)  = %.3f" % cdf_90)
print("P(X <= 270) = %.3f" % cdf_270)
print("P(X <= 360) = %.3f" % cdf_360)
print("Q2 = %.3f \n" % Q2)

# Calculate probabilities using the normal distribution
Q1 = norm.cdf(1.25)                      # P(Z <= 1.25)
print("Q1 = %.4f" % Q1)

Q2 = norm.cdf(-1.25)                     # P(Z <= -1.25)
print("Q2 = %.4f" % Q2)

Q3 = norm.cdf(1.25) - norm.cdf(-0.38)    # P(-0.38 <= Z <= 1.25)
print("Q3 = %.4f \n" % Q3)

# Find Z values corresponding to specific percentiles
Q4 = norm.ppf(0.05)                      # Z at 5th percentile
print("Q4 = %.4f" % Q4)

Q5 = norm.ppf(0.95)                      # Z at 95th percentile
print("Q5 = %.4f" % Q5)

# Define functions to convert between X and Z for a normal distribution
def getZfromX(x, loc, scale):
    z = (x - loc)/scale                  # Formula to convert X to Z
    return z

def getXfromZ(z, loc, scale):
    x = z * scale + loc                  # Formula to convert Z to X
    return x

# Compute PDF and CDF for uniform distribution values
r_values = np.arange(-100, 500, 1)       # Generate range of values
pdf = uniform.pdf(r_values, loc, scale) # Calculate PDF values
cdf = uniform.cdf(r_values, loc, scale) # Calculate CDF values

# Compute mean, variance, and standard deviation
mean = uniform.mean(loc, scale)
var  = uniform.var(loc, scale)
std  = uniform.std(loc, scale)
print("mean = %.2f, variance = %.2f, sd = %.2f \n" % (mean, var, std))

# Plot PDF and CDF for the uniform distribution
fig = plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(r_values, pdf)
plt.title("pdf of U(%d, %d)" % (A, B))

plt.subplot(122)
plt.plot(r_values, cdf)
plt.title("cdf of U(%d, %d)" % (A, B))
plt.show()

# Additional calculations for normal distribution (both X and Z)
loc = 1.25
scale = 0.46
xlower = 1
xupper = 1.75
Q1_x = norm.cdf(xupper, loc, scale) - norm.cdf(xlower, loc, scale)
print("Q1 by X = %.4f" % Q1_x)

# Manual Z-table lookup equivalent calculations
zlower = getZfromX(xlower, loc, scale)
zupper = getZfromX(xupper, loc, scale)
Q1_z = norm.cdf(zupper) - norm.cdf(zlower)
print("Q1 by Z = %.4f, zlower = %.2f, zupper = %.2f" % 
      (Q1_z, zlower, zupper))

# Exponential distribution calculations
loc = 0
scale = 6
lmd = 1/scale                           # Lambda (rate parameter)
print("lambda = %.4f" % lmd)

# Compute mean, variance, and standard deviation
mean = expon.mean(loc, scale)
var = expon.var(loc, scale)
std = expon.std(loc, scale)
print("mean = %.2f, variance = %.2f, sd = %.2f \n" % (mean, var, std))

# Exponential distribution probabilities
Q1 = expon.cdf(10, loc, scale)
Q2 = expon.cdf(10, loc, scale) - expon.cdf(5, loc, scale)
print("Q1 = %.4f" % Q1)
print("Q2 = %.4f" % Q2, "\n")

# Plot PDF and CDF for exponential distribution
r_values = getPercentileValues(expon, loc, scale)
fig = plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(r_values, expon.pdf(r_values, loc, scale))
plt.title("pdf of Exp(%.4f)" % lmd)

plt.subplot(122)
plt.plot(r_values, expon.cdf(r_values, loc, scale))
plt.title("cdf of Exp(%.4f)" % lmd)
plt.show()
