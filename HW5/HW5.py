# Raafi Rahman
# Stat 72401 HW #5

# Libraries ================================================

from numpy.random.mtrand import logistic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale,StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,accuracy_score

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

class1 = np.append(np.random.normal(-1, 1, (20, 50)), np.full((20, 1), 1), axis=1)
class2 = np.append(np.random.normal(0, 1, (20, 50)), np.full((20, 1), 2), axis=1)
class3 = np.append(np.random.normal(1, 1, (20, 50)), np.full((20, 1), 3), axis=1)

df = pd.DataFrame(np.vstack((class1,class2,class3)))
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

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df.iloc[:,0:50])
df_pca = pd.DataFrame(principalComponents, columns=['PC1', 'PC2'])
df_pca['labels'] = df[[50]]
df_pca.labels = df_pca.labels.astype(int)
fig = plt.figure(figsize=(15, 8))
sns.scatterplot(x="PC1", y="PC2", hue="labels", palette={1:'green', 2:'red', 3:'blue'}, data=df_pca)
plt.grid()
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
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
(d) Perform K-means clustering with K = 2. Describe your results.
(e) Now perform K-means clustering with K = 4, and describe your
results.
(f) Now perform K-means clustering with K = 3 on the first two
principal component score vectors, rather than on the raw data.
That is, perform K-means clustering on the 60 × 2 matrix of
which the first column is the first principal component score
vector, and the second column is the second principal component
score vector. Comment on the results.

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

(b) Apply hierarchical clustering to the samples using correlation-
based distance, and plot the dendrogram. Do the genes separate

the samples into the two groups? Do your results depend on the
type of linkage used?
(c) Your collaborator wants to know which genes differ the most
across the two groups. Suggest a way to answer this question,
and apply it here.
"""