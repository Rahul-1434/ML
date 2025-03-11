from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Class']
dataset = pd.read_csv("8-dataset.csv", names=names)

dataset['Class'] = dataset['Class'].str.strip().str.lower()

print("Unique class labels:", dataset['Class'].unique())

label = {'setosa': 0, 'versicolor': 1, 'virginica': 2}

y = [label[c] for c in dataset['Class']]

X = dataset.iloc[:, :-1]

X, y = shuffle(X, y, random_state=42)

colormap = np.array(['red', 'lime', 'black'])

plt.figure(figsize=(14, 7))

plt.subplot(1, 3, 1)
plt.title('Real')
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y])

model = KMeans(n_clusters=3, random_state=0).fit(X)
plt.subplot(1, 3, 2)
plt.title('KMeans')
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_])

kmeans_accuracy = metrics.accuracy_score(y, model.labels_)
kmeans_confusion_matrix = metrics.confusion_matrix(y, model.labels_)
print('The accuracy score of KMeans: ', kmeans_accuracy)
print('The Confusion matrix of KMeans:\n', kmeans_confusion_matrix)

gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
y_cluster_gmm = gmm.predict(X)
plt.subplot(1, 3, 3)
plt.title('GMM Classification')
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm])

gmm_accuracy = metrics.accuracy_score(y, y_cluster_gmm)
gmm_confusion_matrix = metrics.confusion_matrix(y, y_cluster_gmm)
print('The accuracy score of GMM: ', gmm_accuracy)
print('The Confusion matrix of GMM:\n', gmm_confusion_matrix)

plt.show()
