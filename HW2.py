# Raafi Rahman
# Stat 72401 HW #2

# Libraries ================================================

from numpy.random.mtrand import logistic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from seaborn import categorical

import sklearn.linear_model as linMod
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

import os

# Question 10 ==============================================

# Laptop Path
# UP_DIR = "C:\\Users\\Rafha\\OneDrive\\Documents\\GitHub\\STAT72401\\Datasets"

# Desktop Path
UP_DIR = "C:\\Users\\Rafha\\Desktop\\Code\\STAT72401\\Datasets"

csvPath = os.path.join(UP_DIR, 'Weekly.csv')
weeklyRaw = pd.read_csv(csvPath)

# Direction_Up = 1, else 0. Drop index column
weeklyData = pd.get_dummies(
    data=weeklyRaw, drop_first=True).drop(columns="Unnamed: 0")

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
# Un-quote to see graph
# sns.pairplot(weeklyData)
# plt.show()

print("""As the year increases, so does volume.
If today is positive then direction is 'up' and vice-versa.
Everything else seems uniform.""")


# Part B ---------------------------------------------------

# Scikit Learn
X = weeklyData[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
y = weeklyData['Direction_Up']

LogRegress = linMod.LogisticRegression()

Model = LogRegress.fit(X, y)

# Statsmodels
yStat, XStat = dmatrices('Direction_Up ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume',
                         data=weeklyData, return_type='dataframe')

ModelStats = sm.Logit(yStat, XStat).fit()

# Model Summary
print(ModelStats.summary())

print('Lag2 has a p-value of 0.03, implying statistical significance')

# Part C ---------------------------------------------------

ConfuseMatrix = confusion_matrix(y, LogRegress.predict(X))
print(ConfuseMatrix)

# Confusion Matrix:
# [[ 54 430]
#  [ 48 557]]

print('''54 "down" days and 557 "up" days were correctly classified.
The model resulted in 430 false ups and 48 false downs.''')

# Part D ---------------------------------------------------

trainData = weeklyData[(weeklyData['Year'] >= 1990) &
                       (weeklyData['Year'] <= 2008)]
testData = weeklyData[(weeklyData['Year'] >= 2009) &
                      (weeklyData['Year'] <= 2010)]

XTrain = trainData['Lag2']
XTrain = XTrain.values.reshape(np.shape(XTrain)[0], 1)

yTrain = trainData['Direction_Up']

XTest = testData['Lag2']
XTest = XTest.values.reshape(np.shape(XTest)[0], 1)

yTest = testData['Direction_Up']

ModelD = LogRegress.fit(XTrain, yTrain)

print(confusion_matrix(yTest, LogRegress.predict(XTest)))

print('Accuracy of Logistic Regression: ', LogRegress.score(XTest, yTest))

# Part E ---------------------------------------------------

LDA = LinearDiscriminantAnalysis()
LDA.fit(XTrain, yTrain)

print(confusion_matrix(yTest, LDA.predict(XTest)))

print('Accuracy of Linear Discriminant Analysis: ', LDA.score(XTest, yTest))

# Part F ---------------------------------------------------

QDA = QuadraticDiscriminantAnalysis()
QDA.fit(XTrain, yTrain)

print(confusion_matrix(yTest, QDA.predict(XTest)))

print('Accuracy of Quadratric Discriminant Analysis: ', QDA.score(XTest, yTest))

# Part G ---------------------------------------------------

# K NEAREST NEIGHBORS

LDA = LinearDiscriminantAnalysis()
LDA.fit(XTrain, yTrain)

print(confusion_matrix(yTest, LDA.predict(XTest)))

print('Accuracy of Linear Discriminant Analysis: ', LDA.score(XTest, yTest))

# Part H ---------------------------------------------------


# Part I ---------------------------------------------------
