# Raafi Rahman
# Stat 72401 HW #2

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
import statsmodels.formula.api as smf

import os

# Question 10 ==============================================

# Laptop Path
UP_DIR = "C:\\Users\\Rafha\\OneDrive\\Documents\\GitHub\\STAT72401\\Datasets"

# Desktop Path
# UP_DIR = "C:\\Users\\Rafha\\Desktop\\Code\\STAT72401\\Datasets"

csvPath = os.path.join(UP_DIR, 'Weekly.csv')
weeklyRaw = pd.read_csv(csvPath)

# Direction_Up = 1, else 0. Drop index column
weeklyData = pd.get_dummies(data=weeklyRaw, drop_first=True).drop(columns="Unnamed: 0")

print("Data head:")
print(weeklyData.head())


# Part A ---------------------------------------------------

print("""\n(A) Produce some numerical and graphical summaries of the Weekly data.
    Do there appear to be any patterns?\n""")

# Statistical facts
print(weeklyData.describe(), "\n")

# Pairwise correlation
print(weeklyData.corr())

# Pairplot
sns.pairplot(weeklyData)
plt.show()