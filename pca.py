from sklearn import datasets
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

iris = datasets.load_iris()

f = iris.get("feature_names")
print(f" Features: {f}")

# Access the target values (species labels)
targets = iris.target

# Access the target names (species names)
target_names = iris.target_names

# Print the target names
print(f" Species Names: {target_names}")

iris_meaned = iris.data - np.mean(iris.data, axis=0)
# print(iris_meaned)

covariance_matrix = np.cov(iris_meaned, rowvar=False)
# print(c)

eigen_value, eigen_vector = np.linalg.eig(covariance_matrix)
# print(eigen_value)
# print(eigen_vector)

# indices that would sort the eigenvalues in descending order
sorted_index = np.argsort(eigen_value)[::-1]
# sorts the eigenvalues in descending order
sorted_eig_value = eigen_value[sorted_index]
# sorts the eigenvectors according to the sorted eigenvalues
sorted_eig_vector = eigen_vector[:,sorted_index]
# print(sorted_eigen_value)
# print(sorted_eigen_vector)


# sets the number of principal components to keep
n = 2
eig_vec_subset = sorted_eig_vector[:,0:n]
# print(eig_vec_subset)


# Reduced dimensional representation of the Iris dataset after applying PCA.
iris_reduced = np.dot(eig_vec_subset.transpose(), iris_meaned.transpose()).transpose()
# print(iris_reduced)

# Reconstruct the original data from the reduced data
iris_reconstructed = np.dot(iris_reduced, eig_vec_subset.T) + np.mean(iris.data, axis=0)
# print(iris_reconstructed)

# Compute cosine similarity between original and reconstructed data
cos_sim = np.mean(cosine_similarity(iris.data, iris_reconstructed))

print(f"\n\nAverage Cosine Similarity: {cos_sim:.4f}")
