# Raafi Rahman
# Stat 72401 HW #4

# Libraries ================================================

import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier

import os

# Problem 2 ================================================

UP_DIR = "C:\\Users\\Rafha\\Desktop\\Code\\STAT72401\\Datasets"

csvPath = os.path.join(UP_DIR, 'OJ.csv')
OJ = pd.read_csv(csvPath)

OJData = pd.get_dummies(
    data=OJ, drop_first=True).drop(columns="Unnamed: 0")

print(OJData)

Target = OJData['Purchase_MM']

BoostScore = 0
BoostMSE = 0

BagScore = 0
BagMSE = 0

RanForScore = 0
RanForMSE = 0

LogRegScore = 0
LogRegMSE = 0

# Test 100 times and take average Score and MSE
for iterations in range(100):
    XTrain, XTest, YTrain, YTest = train_test_split(
        OJData.drop(columns='Purchase_MM'), Target, test_size=0.2)

    # Boosting -------------------------------------------------

    model = GradientBoostingClassifier()
    model.fit(XTrain, YTrain)

    # Model Score
    BoostScore += model.score(XTest, YTest)

    # Mean Squared Error
    predictions = model.predict(XTest)
    BoostMSE += mean_squared_error(YTest, predictions)

    # Bagging --------------------------------------------------

    model = BaggingClassifier()
    model.fit(XTrain, YTrain)

    # Model Score
    BagScore += model.score(XTest, YTest)

    # Mean Squared Error
    predictions = model.predict(XTest)
    BagMSE += mean_squared_error(YTest, predictions)

    # Random Forest --------------------------------------------

    model = RandomForestClassifier()
    model.fit(XTrain, YTrain)

    # Model Score
    RanForScore += model.score(XTest, YTest)

    # Mean Squared Error
    predictions = model.predict(XTest)
    RanForMSE += mean_squared_error(YTest, predictions)

    # Logistic Regression --------------------------------------

    model = LogisticRegression(max_iter=1000)
    model.fit(XTrain, YTrain)

    # Model Score
    LogRegScore += model.score(XTest, YTest)

    # Mean Squared Error
    predictions = model.predict(XTest)
    LogRegMSE += mean_squared_error(YTest, predictions)

print("\nAverage Boosting Score:", BoostScore/100)
print("Average Boosting MSE:", BoostMSE/100)

print("\nAverage Bagging Score:", BagScore/100)
print("Average Bagging MSE:", BagMSE/100)

print("\nAverage Random Forest Score:", RanForScore/100)
print("Average Random Forest MSE:", RanForMSE/100)

print("\nAverage Logistic Regression Score:", LogRegScore/100)
print("Average Logistic Regression MSE:", LogRegMSE/100)

print("""\nOut of Boosting, Bagging, and Random Forest, 
Boosting performs the best. When each model was ran 100 times, boosting
had an average score of about .82 and an average MSE of about .18.
Logistic Regression had an average score of about .83 and an average
MSE of about .17. It worked surprisingly well, outperforming all other models.""")
