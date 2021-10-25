# Raafi Rahman
# Stat 72401 HW #3

# Libraries ================================================

from numpy.random.mtrand import logistic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

# Question 8 ==============================================

# Part A ---------------------------------------------------
print('\nPart A ---------------------------------------------------\n')

np.random.seed(1)

# The 'rnorm' function has default mu = 0, and sigma = 1
X = np.random.normal(0, 1, size=100)
Y = X - 2*(X**2) + np.random.normal(0, 1, size=100)

print('n is the number of observations. P is the order. n = 100 and p = 2.')

print('Equation is: Y = 1X - 2X^2 + epsilon')

# Part B ---------------------------------------------------
print('\nPart B ---------------------------------------------------\n')

plt.scatter(X, Y)
plt.show()

print('''The data seems to trend along a negative quadratic curve
with most of the data concentrated around 0, from -1 to 1.
This is because we used mu = 0 and sigma = 1''')


# Part C ---------------------------------------------------
print('\nPart C ---------------------------------------------------\n')

# random seed
np.random.seed()

# The 'rnorm' function has default mu = 0, and sigma = 1
X = np.random.normal(0, 1, size=100)
y = X - 2*(X**2) + np.random.normal(0, 1, size=100)

df = pd.DataFrame({"X": X, "X2": X**2, "X3": X**3, "X4": X**4, "y": y})

x = np.linspace(-3, 3, 100)

loo = LeaveOneOut()

# i
runningSum = 0
iteration = 0

for train, test in loo.split(df):
    Xtrain, Xtest = X[train], X[test]
    ytrain, ytest = y[train], y[test]

    polyRegress = PolynomialFeatures(degree=1)
    xTrainPoly = polyRegress.fit_transform(Xtrain.reshape(-1, 1))
    xTestPoly = polyRegress.fit_transform(Xtest.reshape(-1, 1))

    linRegress = LinearRegression()
    linRegress.fit(xTrainPoly, ytrain)

    runningSum += mean_squared_error(ytest, linRegress.predict(xTestPoly))
    iteration += 1

print("i: MSE =", runningSum/iteration)

# ii

runningSum = 0
iteration = 0

for train, test in loo.split(df):
    Xtrain, Xtest = X[train], X[test]
    ytrain, ytest = y[train], y[test]

    polyRegress = PolynomialFeatures(degree=2)
    xTrainPoly = polyRegress.fit_transform(Xtrain.reshape(-1, 1))
    xTestPoly = polyRegress.fit_transform(Xtest.reshape(-1, 1))

    linRegress = LinearRegression()
    linRegress.fit(xTrainPoly, ytrain)

    runningSum += mean_squared_error(ytest, linRegress.predict(xTestPoly))
    iteration += 1

print("ii: MSE =", runningSum/iteration)

# iii

runningSum = 0
iteration = 0

for train, test in loo.split(df):
    Xtrain, Xtest = X[train], X[test]
    ytrain, ytest = y[train], y[test]

    polyRegress = PolynomialFeatures(degree=3)
    xTrainPoly = polyRegress.fit_transform(Xtrain.reshape(-1, 1))
    xTestPoly = polyRegress.fit_transform(Xtest.reshape(-1, 1))

    linRegress = LinearRegression()
    linRegress.fit(xTrainPoly, ytrain)

    runningSum += mean_squared_error(ytest, linRegress.predict(xTestPoly))
    iteration += 1

print("iii: MSE =", runningSum/iteration)

# iv

runningSum = 0
iteration = 0

for train, test in loo.split(df):
    Xtrain, Xtest = X[train], X[test]
    ytrain, ytest = y[train], y[test]

    polyRegress = PolynomialFeatures(degree=4)
    xTrainPoly = polyRegress.fit_transform(Xtrain.reshape(-1, 1))
    xTestPoly = polyRegress.fit_transform(Xtest.reshape(-1, 1))

    linRegress = LinearRegression()
    linRegress.fit(xTrainPoly, ytrain)

    runningSum += mean_squared_error(ytest, linRegress.predict(xTestPoly))
    iteration += 1

print("iv: MSE =", runningSum/iteration)


# Part D ---------------------------------------------------
print('\nPart D ---------------------------------------------------\n')

# Set a new random seed
np.random.seed()

# The 'rnorm' function has default mu = 0, and sigma = 1
X = np.random.normal(0, 1, size=100)
y = X - 2*(X**2) + np.random.normal(0, 1, size=100)

df = pd.DataFrame({"X": X, "X2": X**2, "X3": X**3, "X4": X**4, "y": y})

x = np.linspace(-3, 3, 100)

loo = LeaveOneOut()

# i
runningSum = 0
iteration = 0

for train, test in loo.split(df):
    Xtrain, Xtest = X[train], X[test]
    ytrain, ytest = y[train], y[test]

    polyRegress = PolynomialFeatures(degree=1)
    xTrainPoly = polyRegress.fit_transform(Xtrain.reshape(-1, 1))
    xTestPoly = polyRegress.fit_transform(Xtest.reshape(-1, 1))

    linRegress = LinearRegression()
    linRegress.fit(xTrainPoly, ytrain)

    runningSum += mean_squared_error(ytest, linRegress.predict(xTestPoly))
    iteration += 1

print("i: MSE =", runningSum/iteration)

# ii

runningSum = 0
iteration = 0

for train, test in loo.split(df):
    Xtrain, Xtest = X[train], X[test]
    ytrain, ytest = y[train], y[test]

    polyRegress = PolynomialFeatures(degree=2)
    xTrainPoly = polyRegress.fit_transform(Xtrain.reshape(-1, 1))
    xTestPoly = polyRegress.fit_transform(Xtest.reshape(-1, 1))

    linRegress = LinearRegression()
    linRegress.fit(xTrainPoly, ytrain)

    runningSum += mean_squared_error(ytest, linRegress.predict(xTestPoly))
    iteration += 1

print("ii: MSE =", runningSum/iteration)

# iii

runningSum = 0
iteration = 0

for train, test in loo.split(df):
    Xtrain, Xtest = X[train], X[test]
    ytrain, ytest = y[train], y[test]

    polyRegress = PolynomialFeatures(degree=3)
    xTrainPoly = polyRegress.fit_transform(Xtrain.reshape(-1, 1))
    xTestPoly = polyRegress.fit_transform(Xtest.reshape(-1, 1))

    linRegress = LinearRegression()
    linRegress.fit(xTrainPoly, ytrain)

    runningSum += mean_squared_error(ytest, linRegress.predict(xTestPoly))
    iteration += 1

print("iii: MSE =", runningSum/iteration)

# iv

runningSum = 0
iteration = 0

for train, test in loo.split(df):
    Xtrain, Xtest = X[train], X[test]
    ytrain, ytest = y[train], y[test]

    polyRegress = PolynomialFeatures(degree=4)
    xTrainPoly = polyRegress.fit_transform(Xtrain.reshape(-1, 1))
    xTestPoly = polyRegress.fit_transform(Xtest.reshape(-1, 1))

    linRegress = LinearRegression()
    linRegress.fit(xTrainPoly, ytrain)

    runningSum += mean_squared_error(ytest, linRegress.predict(xTestPoly))
    iteration += 1

print("iv: MSE =", runningSum/iteration)

print('''\nThe results are not the same because using a different seed
gives us a different dataset that will fit our model differently.''')

# Part E ---------------------------------------------------
print('\nPart E ---------------------------------------------------\n')

print('''As expected, the 2nd degree polynomial, on average, gives us
the best results because the way our data was constructed follows a
quadtratic trend.''')


# Part F ---------------------------------------------------
print('\nPart F ---------------------------------------------------\n')

# Set a new random seed
np.random.seed()

# The 'rnorm' function has default mu = 0, and sigma = 1
X = np.random.normal(0, 1, size=100)
y = X - 2*(X**2) + np.random.normal(0, 1, size=100)

df = pd.DataFrame({"X": X, "X2": X**2, "X3": X**3, "X4": X**4, "y": y})

x = np.linspace(-3, 3, 100)

loo = LeaveOneOut()

# i
runningSum = 0
iteration = 0

polyRegress = PolynomialFeatures(degree=1)
xPoly = polyRegress.fit_transform(X.reshape(-1, 1))
xPoly = polyRegress.fit_transform(X.reshape(-1, 1))

model = sm.OLS(y, xPoly).fit()

runningSum += mean_squared_error(y, model.predict(xPoly))
iteration += 1

for i in range(2):
    if model.pvalues[i] >= 0.05:
        print("pvalue", i, ":", model.pvalues[i])

print("\ni: MSE =", runningSum/iteration, "\n")

# ii

runningSum = 0
iteration = 0

polyRegress = PolynomialFeatures(degree=2)
xPoly = polyRegress.fit_transform(X.reshape(-1, 1))
xPoly = polyRegress.fit_transform(X.reshape(-1, 1))

model = sm.OLS(y, xPoly).fit()

runningSum += mean_squared_error(y, model.predict(xPoly))
iteration += 1

for i in range(3):
    if model.pvalues[i] >= 0.05:
        print("pvalue", i, ":", model.pvalues[i])

print("\nii: MSE =", runningSum/iteration, "\n")

# iii

runningSum = 0
iteration = 0

polyRegress = PolynomialFeatures(degree=3)
xPoly = polyRegress.fit_transform(X.reshape(-1, 1))
xPoly = polyRegress.fit_transform(X.reshape(-1, 1))

model = sm.OLS(y, xPoly).fit()

runningSum += mean_squared_error(y, model.predict(xPoly))
iteration += 1

for i in range(4):
    if model.pvalues[i] >= 0.05:
        print("pvalue", i, ":", model.pvalues[i])

print("\niii: MSE =", runningSum/iteration, "\n")

# iv

runningSum = 0
iteration = 0

polyRegress = PolynomialFeatures(degree=4)
xPoly = polyRegress.fit_transform(X.reshape(-1, 1))
xPoly = polyRegress.fit_transform(X.reshape(-1, 1))

model = sm.OLS(y, xPoly).fit()

runningSum += mean_squared_error(y, model.predict(xPoly))
iteration += 1

for i in range(5):
    if model.pvalues[i] >= 0.05:
        print("pvalue", i, ":", model.pvalues[i])

print("\niv: MSE =", runningSum/iteration, "\n")

print('''When we run a standard polynomial regression without LOOCV,
we see that the p values are significant for all coefficients except
the first or second degree term.''')
