from typing import List

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances

def get_kmean_matrix(embeddings: List, n_clusters: int = 100, is_gmm: bool = False):
    if not is_gmm:
        clustering_model = KMeans(n_clusters=n_clusters).fit(embeddings)
        # cluster_distance = clustering_model.transform(embeddings).min(axis=1)
        # cluster_assignment = clustering_model.labels_
        centers = clustering_model.cluster_centers_
    else:
        gmm = GaussianMixture(n_components=n_clusters,
                              covariance_type="full",
                              random_state=0).fit(embeddings)
        # cluster_assignment = gmm.predict(corpus_embeddings)
        centers = gmm.means_

    return centers

if __name__ == "__main__":
    corpus = []     # bert embedding
    kmean_matrix, centers, cluster_assignments = get_kmean_matrix(corpus, num_cluster_list=[72])
    # cosine_scores = np.array(torch.cdist(torch.tensor(centers), torch.tensor(corpus)))
    # centers_idx = [np.argmin(x) for x in cosine_scores]
