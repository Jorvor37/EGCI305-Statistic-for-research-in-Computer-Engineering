import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Numpy version =", np.version.version)
print("Seaborn version =", sns.__version__)

import scipy
print("Scipy version =", scipy.__version__)

from scipy import stats
from scipy.stats import norm            # Normal distribution
from scipy.stats import expon           # Exponential distribution


import sympy
print("Sympy version =", sympy.__version__)

from sympy import *

# Import essential libraries for numerical operations, plotting, and advanced mathematical functions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sympy
from scipy import stats
from scipy.stats import norm, expon
from sympy import *

# Print versions of the libraries to ensure compatibility
print("Numpy version =", np.version.version)
print("Seaborn version =", sns.__version__)
print("Scipy version =", scipy.__version__)
print("Sympy version =", sympy.__version__)



#----- Declare variable & function symbols
x   = Symbol('x')
y   = Symbol('y')
fx  = Function('fx')(x)
fy  = Function('fy')(y)
fxy = Function('fxy')(x, y)
fxy = (6/5) * (x + y**2)

fx      = Integral(fxy, (y, 0, 1/4))       # hold x fixed, integral by dy
prob_xy = Integral(fx,  (x, 0, 1/4))       # integral by dx
pprint(prob_xy, use_unicode=True)
prob_xy.doit()



# Declare symbolic variables and functions for symbolic computation using SymPy
x = Symbol('x')
y = Symbol('y')
fx = Function('fx')(x)
fy = Function('fy')(y)
fxy = Function('fxy')(x, y)
fxy = (6/5) * (x + y**2)  # Define the joint probability density function (PDF)

# Calculate the marginal probability density function (PDF) for X
fx = Integral(fxy, (y, 0, 1/4))  # Integrate over y, holding x fixed
prob_xy = Integral(fx, (x, 0, 1/4))  # Integrate the result over x
pprint(prob_xy, use_unicode=True)
prob_xy.doit()  # Evaluate the integral to get the probability



#------ Marginal f(y) : hold y fixed, integrate all fxy over x
fy = Integral(fxy, (x, 0, 1))
pprint(fy, use_unicode=True)
fy.doit()



# Calculate the marginal probability density function (PDF) for Y
fy = Integral(fxy, (x, 0, 1))  # Integrate over x, holding y fixed
pprint(fy, use_unicode=True)
fy.doit()  # Evaluate the integral to get the marginal PDF for Y



#------ P(1/4 < Y < 3/4) : integrate fy with bounds
prob_y = Integral(fy.doit(), (y, 1/4, 3/4))
prob_y.doit()



# Calculate the probability that Y is between 1/4 and 3/4
prob_y = Integral(fy.doit(), (y, 1/4, 3/4))  # Integrate the marginal PDF of Y over the interval
prob_y.doit()  # Evaluate the integral to get the probability