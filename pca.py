import numpy as np
from sklearn import datasets
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()

# Get feature names and target names
features = iris.get("feature_names")
target_names = iris.target_names
print(f"Features: {features}")
print(f"Species Names: {target_names}")

# Standardizing the data
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris.data)

# Compute covariance matrix
covariance_matrix = np.cov(iris_scaled, rowvar=False)

# Perform Eigen decomposition of the covariance matrix
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# Sort the eigenvalues in descending order and reorder the eigenvectors accordingly
sorted_indices = np.argsort(eigen_values)[::-1]
sorted_eigenvalues = eigen_values[sorted_indices]
sorted_eigenvectors = eigen_vectors[:, sorted_indices]

# Select the top 'n' principal components
n_components = 2
eigenvector_subset = sorted_eigenvectors[:, :n_components]

# Project the data onto the top 'n' components (reduced dimensionality)
iris_reduced = np.dot(iris_scaled, eigenvector_subset)

# Reconstruct the original data from the reduced dimensionality data
iris_reconstructed = np.dot(iris_reduced, eigenvector_subset.T) + np.mean(iris.data, axis=0)

# Compute cosine similarity between the original data and reconstructed data
cosine_sim = np.mean(cosine_similarity(iris.data, iris_reconstructed))

# Output the average cosine similarity
print(f"\n\nAverage Cosine Similarity between Original data and Reconstructed data = {cosine_sim:.4f} \n\n")

# Plot explained variance ratio
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues / np.sum(sorted_eigenvalues), alpha=0.7)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.show(block=False)  # Display in a separate window

# Visualize the reduced data in 2D
plt.figure(figsize=(8, 6))
plt.scatter(iris_reduced[:, 0], iris_reduced[:, 1], c=iris.target, cmap='viridis')
plt.colorbar(label='Species')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Iris Dataset Reduced to 2D')
plt.show()  # This will display the scatter plot in another window
