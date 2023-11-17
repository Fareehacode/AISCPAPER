import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# Loading FESEM image data and material labels
# Make sure 'X' contains your image data, and 'y' contains the corresponding material labels
fesemds = datasets.load_fesemds()
X = fesemds.data  

# Features (FESEM images)
y = fesemds.target  

# Material labels
# We can reduce the dimensionality of the data for visualization (PCA)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Set the number of clusters
n_clusters = 3

#Create and fit the K-Means model
kmeans =KMeans(n_clusters=n_clusters, random_state=0)  kmeans.fit(X)

#Get cluster assignments for the data points
cluster_labels = kmeans.labels_
plt.figure(figsize=(12, 6))

# Original data (FESEM images)
plt.subplot(1, 2, 1)
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y,cmap='viridis')
plt.title("Original Data")

# K-Means clustering result
splt.subplot(1, 2, 2)
plt.scatter(X_reduced[:, 0],X_reduced[:,1],c=cluster_labels, cmap='viridis')
plt.title("K-Means Clustering")
plt.show()

# Evaluate the clustering quality using the Adjusted Rand Index (ARI)
ari = adjusted_rand_score(y, cluster_labels)
print(f"Adjusted Rand Index: {ari}")

