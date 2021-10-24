# Raafi Rahman
# Stat 72401 HW #3

# Libraries ================================================

from numpy.random.mtrand import logistic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression

# Question 8 ==============================================

# Part A ---------------------------------------------------
print('\nPart A ---------------------------------------------------\n')

np.random.seed(1)

# The 'rnorm' function has default mu = 0, and sigma = 1
X = np.random.normal(0, 1, size = 100)
Y = X - 2*(X**2) + np.random.normal(0, 1, size = 100)

print('n is the number of observations. P is the order. n = 100 and p = 2.')


# Part B ---------------------------------------------------
print('\nPart B ---------------------------------------------------\n')

plt.scatter(X,Y)
plt.show()

print('''The data seems to trend along a negative quadratic curve 
with most of the data concentrated around 0, from -1 to 1.
This is because we used mu = 0 and sigma = 1''')


# Part C ---------------------------------------------------
print('\nPart C ---------------------------------------------------\n')

# random seed
np.random.seed()

# The 'rnorm' function has default mu = 0, and sigma = 1
X = np.random.normal(0, 1, size = 100)

# i
Y = X - 2*(X**2) + np.random.normal(0, 1, size = 100)

# ii
Y = X - 2*(X**2) + np.random.normal(0, 1, size = 100)

# iii
Y = X - 2*(X**2) + np.random.normal(0, 1, size = 100)

# iv
Y = X - 2*(X**2) + np.random.normal(0, 1, size = 100)
