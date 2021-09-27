# Raafi Rahman
# Stat 72401 HW #1

# Libraries ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from seaborn import categorical

import sklearn.linear_model as linMod
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm
import statsmodels.formula.api as smf  # more convenient

import os

# Question 5 ===============================================


# Question 10 ==============================================

# Laptop Path
# UP_DIR = "C:\\Users\\Rafha\\OneDrive\\Documents\\GitHub\\STAT72401\\Datasets"

# Desktop Path
UP_DIR = "C:\\Users\\Rafha\\Desktop\\Code\\STAT72401\\Datasets"

csvPath = os.path.join(UP_DIR, 'Carseats.csv')
CarData = pd.read_csv(csvPath)

print(CarData)

regress = linMod.LinearRegression()


# Part A ---------------------------------------------------

print("\nPART A ---------------------------------------------------")

X = CarData[['Price', 'Urban', 'US']]  # Inputs (Price, Urban, US)
y = CarData['Sales']

# Create dummy variables
X = pd.get_dummies(data=X, drop_first=True)

regress.fit(X, y)

print("Regression coefficients:", regress.coef_)
print("Regression intercept:", regress.intercept_)


# Part B ---------------------------------------------------

print("\nPART B ---------------------------------------------------")

print("""We get...
Regression coefficients: [-0.05445885 -0.02191615  1.2005727 ]
Regression intercept: 13.043468936764892

This means the following...
        - A 1 unit increase in 'Price' causes a -0.05445885 change in 'Sales'
        - An answer of 'Yes' to 'Urban' causes a -0.02191615 change in 'Sales'
        - An answer of 'Yes' to 'US' causes a 1.2005727 change in 'Sales'
        - All variables at 0, 'Sales' would be 13.043468936764892""")


# Part C ---------------------------------------------------

print("\nPART C ---------------------------------------------------")

print("""Equation of model...

y_hat = 13.043468936764892 + (-0.05445885)(Price) + (-0.02191615)(Urban) + (1.2005727)(US),

Urban = 1 if 'Yes', else 0
US = 1 if 'Yes', else 0""")


# Part D ---------------------------------------------------

print("\nPART D ---------------------------------------------------")

print("""We can reject the null hypothesis H_0: Beta_1 = 0 and 
H_0: Beta_3 = 0""")


# Part E ---------------------------------------------------

print("\nPART E ---------------------------------------------------")

regress2 = linMod.LinearRegression()

X2 = CarData[['Price', 'US']]  # Inputs (Price, Urban, US)
y2 = CarData['Sales']

# Create dummy variables
X2 = pd.get_dummies(data=X2, drop_first=True)

regress2.fit(X2, y2)

print("Regression coefficients:", regress2.coef_)
print("Regression intercept:", regress2.intercept_)


# Part F ---------------------------------------------------

print("\nPART F ---------------------------------------------------")

print("R^2 for (a):", regress.score(X, y))
print("R^2 for (e):", regress2.score(X2, y2))

print("""Both models fit the data similarly. They both yield low R^2 values 
meaning they do not fit the data well""")


# Part G ---------------------------------------------------

print("\nPART G ---------------------------------------------------")

print("I don't think sklearn has a built in way to get confidence intervals :(")


# Part H ---------------------------------------------------

print("\nPART H ---------------------------------------------------")

print("Not sure how to check using sklearn")
