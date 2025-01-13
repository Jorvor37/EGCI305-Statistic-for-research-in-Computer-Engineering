# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import pandas as pd

# Printing library versions
print("Numpy version =", np.version.version)
print("Seaborn version =", sns.__version__)
print("Scipy version =", scipy.__version__)

# Data initialization and basic operations
A = np.array([5.9, 7.2, 7.3, 6.3, 8.1, 6.8, 7.0, 7.6, 6.8, 
              6.5, 7.0, 6.3, 7.9, 9.0, 8.2, 8.7, 7.8, 9.7, 
              7.4, 7.7, 9.7, 7.8, 7.7, 11.6, 11.3, 11.8, 10.7])
print("Data =", A, "\n")
print("Sort =", np.sort(A), "\n")
print("Size =", A.size, "\n")

# Statistical description using SciPy
result = stats.describe(A)
print(result)

# Visualization setup and plots
fig = plt.figure(figsize=(10,8))    # Set figure size
ax = plt.subplot(221)               # Create a subplot in a 2x2 grid, position 1
sns.histplot(A).set_title("Histogram")    # Plot histogram

ax = plt.subplot(222)               # Subplot in position 2
sns.kdeplot(A).set_title("KDE")           # Plot Kernel Density Estimate

ax = plt.subplot(223)               # Subplot in position 3
sns.histplot(A, kde=True).set_title("Histogram with KDE")  # Histogram with KDE

ax = plt.subplot(224)               # Subplot in position 4
sns.boxplot(x=A, width=0.2).set_title("Boxplot")           # Boxplot

plt.savefig("flexural_plots.png", bbox_inches="tight")  # Save figure to file
plt.show()                                               # Display figure

# Descriptive statistics using pandas
print(pd.DataFrame(A).describe().transpose())
