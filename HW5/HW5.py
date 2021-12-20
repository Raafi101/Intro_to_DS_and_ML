# Raafi Rahman
# Stat 72401 HW #5

# Libraries ================================================

from numpy.random.mtrand import logistic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

import os

# Question 10 ==============================================

"""
(a) Generate a simulated data set with 20 observations in each of
three classes (i.e. 60 observations total), and 50 variables.
Hint: There are a number of functions in R that you can use to
generate data. One example is the rnorm() function; runif() is
another option. Be sure to add a mean shift to the observations
in each class so that there are three distinct classes.
"""

# Generate data normally
class1 = np.append(np.random.normal(-1, .5, (20, 50)),
                   np.full((20, 1), 0), axis=1)
class2 = np.append(np.random.normal(0, .5, (20, 50)),
                   np.full((20, 1), 2), axis=1)
class3 = np.append(np.random.normal(1, .5, (20, 50)),
                   np.full((20, 1), 1), axis=1)

df = pd.DataFrame(np.vstack((class1, class2, class3)))
print(df)

"""
(b) Perform PCA on the 60 observations and plot the first two prin-
cipal component score vectors. Use a different color to indicate
the observations in each of the three classes. If the three classes
appear separated in this plot, then continue on to part (c). If
not, then return to part (a) and modify the simulation so that
there is greater separation between the three classes. Do not
continue to part (c) until the three classes show at least some
separation in the first two principal component score vectors.
"""

# Reduce data to two most important principal components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df)
dfPCA = pd.DataFrame(principalComponents, columns=['PC1', 'PC2'])
dfPCA['labels'] = df[[50]]
fig = plt.figure(figsize=(10, 10))

plt.plot(dfPCA[:20]['PC1'], dfPCA[:20]['PC2'], "r^")
plt.plot(dfPCA[20:40]['PC1'], dfPCA[20:40]['PC2'], "gs")
plt.plot(dfPCA[40:59]['PC1'], dfPCA[40:59]['PC2'], "bo")

plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

"""
(c) Perform K-means clustering of the observations with K = 3.
How well do the clusters that you obtained in K-means cluster-
ing compare to the true class labels?

Hint: You can use the table() function in R to compare the true
class labels to the class labels obtained by clustering. Be careful
how you interpret the results: K-means clustering will arbitrarily
number the clusters, so you cannot simply check whether the true
class labels and clustering labels are the same.
"""

print("\nC ====================================")

# Perform K-means clustering with K = 3 on RAW DATA
model = KMeans(n_clusters=3)
model.fit(df)

predictions = model.predict(df)

print(predictions)
print(df[[50]])

print(accuracy_score(model.labels_, df[[50]]))

"""
Although the labels are not correctly given when the K-means model runs,
all predictions should be accurate.
"""

"""
(d) Perform K-means clustering with K = 2. Describe your results.
"""

print("\nD ====================================")

# Perform K-means clustering with K = 2 on RAW DATA
model = KMeans(n_clusters=2)
model.fit(df)

predictions = model.predict(df)

print(predictions)
print(df[[50]])

print(accuracy_score(model.labels_, df[[50]]))

"""
(e) Now perform K-means clustering with K = 4, and describe your
results.
"""

print("\nE ====================================")

# Perform K-means clustering with K = 4 on RAW DATA
model = KMeans(n_clusters=4)
model.fit(df)

predictions = model.predict(df)

print(predictions)
print(df[[50]])

print(accuracy_score(model.labels_, df[[50]]))

"""
(f) Now perform K-means clustering with K = 3 on the first two
principal component score vectors, rather than on the raw data.
That is, perform K-means clustering on the 60 × 2 matrix of
which the first column is the first principal component score
vector, and the second column is the second principal component
score vector. Comment on the results.
"""

print("\nF ====================================")

# Perform K-means clustering with K = 3 on PCA DATA
model = KMeans(n_clusters=3)
model.fit(dfPCA[['PC1', 'PC2']])

predictions = model.predict(dfPCA[['PC1', 'PC2']])

print(predictions)
print(dfPCA[['labels']])

print(accuracy_score(model.labels_, dfPCA[['labels']]))

"""
(g) Using the scale() function, perform K-means clustering with
K = 3 on the data after scaling each variable to have standard
deviation one. How do these results compare to those obtained
in (b)? Explain.
"""

# Question 13 ==============================================

"""
13. On the book website, www.statlearning.com, there is a gene expres-
sion data set (Ch12Ex13.csv) that consists of 40 tissue samples with
measurements on 1,000 genes. The first 20 samples are from healthy
patients, while the second 20 are from a diseased group.

(a) Load in the data using read.csv(). You will need to select
header = F.
"""

# Laptop Path (For my personal machine)
# UP_DIR = "C:\\Users\\Rafha\\OneDrive\\Documents\\GitHub\\STAT72401\\Datasets"

# Desktop Path (For my personal machine)
UP_DIR = "C:\\Users\\Rafha\\Desktop\\Code\\STAT72401\\Datasets"

csvPath = os.path.join(UP_DIR, 'Ch12Ex13.csv')
geneRaw = pd.read_csv(csvPath)

print(geneRaw.head())

"""
(b) Apply hierarchical clustering to the samples using correlation-
based distance, and plot the dendrogram. Do the genes separate
the samples into the two groups? Do your results depend on the
type of linkage used?
"""

# Use hierarchical clustering on data
model = linkage(geneRaw.T)  # Transpose to fix the graph
fig = plt.figure(figsize=(10, 10))
den = dendrogram(model)
plt.show()

"""
It is clear from the dendrogram that the genes seperate into 2
seperate groups
"""

"""
(c) Your collaborator wants to know which genes differ the most
across the two groups. Suggest a way to answer this question,
and apply it here.
"""
