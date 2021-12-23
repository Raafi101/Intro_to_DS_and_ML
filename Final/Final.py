# Raafi Rahman
# Stat 72401 Final


# Libraries ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import sklearn.linear_model as linMod
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import sklearn.tree as tree
from sklearn.tree import DecisionTreeRegressor

from scipy import stats


# Import and clean data ===================================================

# Laptop Path (For my personal machine)
# UP_DIR = "C:\\Users\\Rafha\\OneDrive\\Documents\\GitHub\\STAT72401\\Final"

# Desktop Path (For my personal machine)
UP_DIR = "C:\\Users\\Rafha\\Desktop\\Code\\STAT72401\\Final"

# Open data
csvPath = os.path.join(UP_DIR, 'Houses.csv')
housesRaw = pd.read_csv(csvPath)

# Copy and filter housesRaw and print
housesClean = housesRaw[['property_type', 'purpose', 'bedrooms',
                        'baths', 'Total_Area', 'city', 'province_name',
                         'latitude', 'longitude', 'price']].rename(
    columns={'Total_Area': 'area', 'property_type': 'type', 'province_name': 'province'})

housesClean = housesClean[housesClean.baths != 0]
housesClean = housesClean[housesClean.bedrooms != 0]
housesClean = housesClean[housesClean.area != 0]
housesClean = housesClean[housesClean.purpose == 'For Sale']

housesClean = housesClean.drop(['purpose'], axis=1)

# Remove Outliers
housesClean = housesClean[(
    np.abs(stats.zscore(housesClean['area'])) < 3)]
housesClean = housesClean[(
    np.abs(stats.zscore(housesClean['bedrooms'])) < 3)]
housesClean = housesClean[(
    np.abs(stats.zscore(housesClean['baths'])) < 3)]
housesClean = housesClean[(
    np.abs(stats.zscore(housesClean['price'])) < 3)]

colors = {'Karachi': 'red', 'Lahore': 'blue', 'Islamabad': 'green',
          'Faisalabad': 'yellow', 'Rawalpindi': 'purple'}

print(housesClean)

# Visualize data ==========================================================

# Pairplot (divided into 3 for easy viewing)
sns.set(style='white', font_scale=1, color_codes=True)

maxPrice = housesClean['price'].max()
upperY = maxPrice*1.25

"""
pair = sns.pairplot(housesClean.sample(10000),
                    plot_kws={'alpha': 0.1},
                    x_vars=['type', 'bedrooms', 'baths', 'area'],
                    y_vars=['price'],
                    hue='city')
for axis in pair.fig.axes:
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45)
plt.show()

pair = sns.pairplot(housesClean.sample(10000),
                    plot_kws={'alpha': 0.1},
                    x_vars=['city', 'province', 'latitude', 'longitude'],
                    y_vars=['price'],
                    hue='city')
for axis in pair.fig.axes:
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45)
plt.show()


# Bedrooms x Price
housesClean.plot(x='bedrooms',
                 y='price',
                 kind='scatter',
                 alpha=.1,
                 c=housesClean['city'].map(colors))
plt.xlim([0, 15])
plt.ylim([0, upperY])
plt.title('Bedrooms x Price')
plt.show()

# Bathrooms x Price
housesClean.plot(x='baths',
                 y='price',
                 kind='scatter',
                 alpha=.1,
                 c=housesClean['city'].map(colors))
plt.xlim([0, 15])
plt.ylim([0, upperY])
plt.title('Bathrooms x Price')
plt.show()


# Area x Price
housesClean.plot(x='area',
                 y='price',
                 kind='scatter',
                 alpha=.1,
                 c=housesClean['city'].map(colors))
plt.xlim([0, 100000])
plt.ylim([0, upperY])
sns.regplot(housesClean.area, housesClean.price, order=1,
            ci=None, scatter=False, color='black')
plt.title('Area x Price')
plt.show()


# City x Price
housesClean.plot(x='city',
                 y='price',
                 kind='scatter',
                 alpha=.1,
                 c=housesClean['city'].map(colors))
plt.ylim([0, upperY])
plt.title('City x Price')
plt.show()

# Type x Price
housesClean.plot(x='type',
                 y='price',
                 kind='scatter',
                 alpha=.1,
                 c=housesClean['city'].map(colors))
plt.ylim([0, upperY])
plt.title('Type x Price')
plt.xticks(rotation=45)
plt.show()

# Province x Price
housesClean.plot(x='province',
                 y='price',
                 kind='scatter',
                 alpha=.1,
                 c=housesClean['city'].map(colors))
plt.ylim([0, upperY])
plt.title('Province x Price')
plt.show()
"""

housesClean = pd.get_dummies(
    data=housesClean, drop_first=True).rename(
    columns={'type_Lower Portion': 'type_Lower_Portion',
             'type_Upper Portion': 'type_Upper_Portion',
             'purpose_For Sale': 'purpose_For_Sale'})

print(housesClean)

"""
plt.figure(figsize=(15, 10))
sns.heatmap(housesClean.corr(), annot=True)
plt.show()
"""

# Analysis ================================================================

X = housesClean.drop(columns='price')
y = housesClean['price']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.2)

# Multiple Linear Regression ----------------------------------------------
print("Multiple Linear Regression")
linRegress = linMod.LinearRegression()
linRegress.fit(Xtrain, ytrain)

print("Linear R2:", r2_score(ytest, linRegress.predict(Xtest)))
print(linRegress.coef_)
print(linRegress.intercept_)

# PCA
pca = PCA(n_components=3)
XtrainPCA = pca.fit_transform(Xtrain)
XtestPCA = pca.fit_transform(Xtest)

# Polynomial Regression deg=2
print("Polynomial Regression deg=2")
poly = PolynomialFeatures(degree=2)
XtrainP = poly.fit_transform(XtrainPCA)
XtestP = poly.fit_transform(XtestPCA)
polyRegress = linMod.LinearRegression().fit(XtrainP, ytrain)

print("Quad R2:", r2_score(ytest, polyRegress.predict(XtestP)))
print(polyRegress.coef_)
print(polyRegress.intercept_)

# K Nearest Neighbors Regression ------------------------------------------
KNN = KNeighborsRegressor()
KNN.fit(Xtrain, ytrain)

print("K-Nearest R2:", r2_score(ytest, KNN.predict(Xtest)))

# Test different K values in K Nearest Neighbors

# Minkowski metric
"""
rmse_val = []
BestK = 0
BestR2 = 0

for K in range(1, 20):
    KNN = KNeighborsRegressor(n_neighbors=K)
    KNN.fit(Xtrain, ytrain)
    error = np.sqrt(mean_squared_error(
        ytest, KNN.predict(Xtest)))  # calculate rmse
    rmse_val.append(error)  # store rmse values

    if r2_score(ytest, KNN.predict(Xtest)) > BestR2:  # Update best K value so far
        BestR2 = r2_score(ytest, KNN.predict(Xtest))
        BestK = K

print('Best K value is', BestK)
print('Best R2 score is', BestR2)

curve = pd.DataFrame(rmse_val)  # elbow curve
curve.plot()
plt.show()
"""

# Manhattan metric
"""
rmse_val = []
BestK = 0
BestR2 = 0

for K in range(1, 20):
    KNN = KNeighborsRegressor(n_neighbors=K, metric='manhattan')
    KNN.fit(Xtrain, ytrain)
    error = np.sqrt(mean_squared_error(
        ytest, KNN.predict(Xtest)))  # calculate rmse
    rmse_val.append(error)  # store rmse values

    if r2_score(ytest, KNN.predict(Xtest)) > BestR2:  # Update best K value so far
        BestR2 = r2_score(ytest, KNN.predict(Xtest))
        BestK = K

print('Best K value is', BestK)
print('Best R2 score is', BestR2)

curve = pd.DataFrame(rmse_val)  # elbow curve
curve.plot()
plt.show()
"""


# Decision Tree Regression ------------------------------------------------

# Decision Tree Regression
treeRegress = DecisionTreeRegressor()
treeRegress.fit(Xtrain, ytrain)

print("Decision Tree R2:", r2_score(ytest, treeRegress.predict(Xtest)))

# Random Forest Regression
RFRegress = RandomForestRegressor()
RFRegress.fit(Xtrain, ytrain)

print("Random Forest R2:", r2_score(ytest, RFRegress.predict(Xtest)))
