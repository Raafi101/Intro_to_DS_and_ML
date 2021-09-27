# Raafi Rahman
# Stat 724 HW #1

# Libraries ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from seaborn import categorical

from sklearn.preprocessing import scale, LabelEncoder, OneHotEncoder
import sklearn.linear_model as linMod
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm
import statsmodels.formula.api as smf  # more convenient

import os

# Question 5 ===============================================


# Question 10 ==============================================

# Laptop Path
#UP_DIR = "C:\\Users\\Rafha\\OneDrive\\Documents\\GitHub\\STAT72401\\Datasets"

# Desktop Path
UP_DIR = "C:\\Users\\Rafha\\Desktop\\Code\\STAT72401\\Datasets"

csvPath = os.path.join(UP_DIR, 'Carseats.csv')
CarData = pd.read_csv(csvPath)
print(CarData.info())
print(CarData.head())

regress = linMod.LinearRegression()

X = CarData[['Price']]  # Inputs (Price, Urban, US)
y = CarData['Sales']

preX = CarData[['Urban', 'US']]

lableEncoderObject = LabelEncoder()
onehotencoder = OneHotEncoder()

urban = lableEncoderObject.fit_transform(preX['Urban'])
urban = onehotencoder.fit_transform(urban).toarray()

us = lableEncoderObject.fit_transform(preX['US'])
us = onehotencoder.fit_transform(us).toarray()

print(us, urban)

'''

regress.fit(X, y)

print(regress.coef_)
print(regress.intercept_)

# Create Plot
Price = np.arange(0, 200)
Income = np.arange(0, 200)
Age = np.arange(0, 100)

B1, B2 = np.meshgrid(Price, Income, indexing='xy')
Z = np.zeros((Income.size, Price.size))

for (i,j),v in np.ndenumerate(Z):
        Z[i,j] =(regress.intercept_ + B1[i,j]*regress.coef_[0] + B2[i,j]*regress.coef_[1])

fig = plt.figure(figsize=(10,6))
fig.suptitle('Regression: Sales on Price + Income', fontsize=20)

ax = axes3d.Axes3D(fig)

ax.plot_surface(B1, B2, Z, rstride=10, cstride=5, alpha=0.4, color='b')
ax.scatter3D(CarData.Price, CarData.Income, CarData.Sales, c='r')

ax.set_xlabel('Price')
ax.set_xlim(0,200)
ax.set_ylabel('Income')
ax.set_ylim(ymin=0)
ax.set_zlabel('Sales')

plt.style.use('seaborn')

plt.show()
# a

'''
