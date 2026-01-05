#programa que agrupa productos y ventas

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = np.array([
    [10, 50],
    [12, 40],
    [15, 60],
    [20, 80],
    [25, 100],
    [30, 90],
    [35, 120],
    [40, 150]
])

X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

plt.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis', s=100)
plt.xlabel("Precio")
plt.ylabel("Unidades vendidas")
plt.title("Segmentación de productos por precio y ventas")
plt.show()

