#programa que agrupa usuarios según su edad y gasto mensual y muestra esos grupos en una gráfica.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

X = np.array([
    [18, 200],
    [22, 250],
    [25, 300],
    [30, 500],
    [35, 550],
    [40, 600],
    [45, 650],
    [50, 700]
])

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.xlabel("Edades")
plt.ylabel("Gastos")
plt.title("Clustering de usuarios con K-Means y PCA")
plt.show()
