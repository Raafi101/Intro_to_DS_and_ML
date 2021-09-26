# Raafi Rahman
# Stat 724 HW #1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm
import statsmodels.formula.api as smf # more convenient

import os

UP_DIR = "C:\\Users\\Rafha\\OneDrive\\Documents\\GitHub\\STAT72401\\Datasets"

plt.style.use('seaborn-white')

# Question 5 ===============================================




# Question 10 ==============================================

csv = os.path.join(UP_DIR, 'Carseats.csv')
CarData = pd.read_csv(csv)
print(CarData.info())
print(CarData.head())

plt.figure(figsize=(5, 5))
sns.regplot(CarData.Income, CarData.Price, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.xlim(0,150)
plt.ylim(ymin=0)
plt.show()

# a

