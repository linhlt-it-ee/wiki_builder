from typing import List

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances

def get_kmean_matrix(embeddings: List, n_clusters: int = 100, is_gmm: bool = False):
    if not is_gmm:
        model = KMeans(n_clusters=n_clusters).fit(embeddings)
        centers = model.cluster_centers_
    else:
        model = GaussianMixture(n_components=n_clusters, covariance_type="full",
                              random_state=0).fit(embeddings)
        centers = model.means_
   
    return centers

"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
y = model.predict(embeddings)
X = PCA(2).fit_transform(np.vstack((embeddings, centers)))
plt.scatter(X[:-n_clusters, 0], X[:-n_clusters, 1], c=y, cmap="viridis")
plt.scatter(X[-n_clusters:, 0], X[-n_clusters:, 1], c="black", s=100, alpha=0.5)
plt.savefig("cluster.jpg")
exit()
"""

if __name__ == "__main__":
    corpus = []     # bert embedding
    kmean_matrix, centers, cluster_assignments = get_kmean_matrix(corpus, num_cluster_list=[72])
    # cosine_scores = np.array(torch.cdist(torch.tensor(centers), torch.tensor(corpus)))
    # centers_idx = [np.argmin(x) for x in cosine_scores]
