from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import operator
import copy
from collections import OrderedDict
from sklearn.mixture import GaussianMixture

def get_kmean_matrix(corpus=[], num_cluster_list=[2],is_gmm=False):
    cluster_dict = {}
    range_n_clusters = num_cluster_list  # [2, 3, 4, 5, 6]
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(18, 7)
    # print(corpus[0])
    # if not corpus:
    #     corpus = samples
    highest_cluster = {"cluster_num": range_n_clusters[0], "score": 0}
    # for text in corpus:
    #     print(text)
    corpus_embeddings = corpus  # embedder.encode(corpus)

    for i, num_clusters in enumerate(range_n_clusters):
        # Perform kmean clustering
        if not is_gmm:
            clustering_model = KMeans(n_clusters=num_clusters)
            cluster_distance_matrix = clustering_model.fit_transform(corpus_embeddings)
            cluster_assignment = clustering_model.labels_
            centers = clustering_model.cluster_centers_
        else:
            gmm = GaussianMixture(3,
                                  covariance_type='full',
                                  random_state=0).fit(corpus_embeddings)

        return cluster_assignment, centers
        # center_idx = [np.where(corpus_embeddings==x) for x in centers]
        # print(center_idx)
        # print(centers)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        # silhouette_avg = silhouette_score(corpus_embeddings, cluster_labels)

        # Compute the silhouette scores for each sample
        # sample_silhouette_values = silhouette_samples(corpus_embeddings, cluster_labels)
        # print("cluster label",cluster_distance_matrix[0])
        # silhouette_matrix=np.zeros((len(corpus_embeddings),num_clusters))
        # silhouette_matrix=[[ sample_silhouette_values[ii] if ii==cluster_assignment[jj] else y for ii,y in enumerate(x)] for jj,x in enumerate(silhouette_matrix)]

if __name__ == "__main__":
    corpus=[]#bert embedding
    kmean_matrix,centers,cluster_assignments=get_kmean_matrix(corpus,num_cluster_list=[72])
        # cosine_scores =np.array(torch.cdist(torch.tensor(centers), torch.tensor(corpus)))
        # centers_idx=[np.argmin(x) for x in cosine_scores]

