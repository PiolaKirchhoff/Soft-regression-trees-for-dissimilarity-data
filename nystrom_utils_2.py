# -*- coding: utf-8 -*-
"""Prototype selection for branch-node optimization in DSRTs.

This module implements Random Prototype Selection (RPS) and k-Medoids
Prototype Selection (kMPS) for Distance-based Soft Regression Trees,
as described in Section 4.2 of the paper. RPS samples observations
uniformly, while kMPS selects medoids from a k-medoids partition of
the distance matrix. Alternative selection strategies are also included.
"""
from logger import *
from sklearn.utils import check_random_state
from scipy.linalg import eigvalsh
import numpy as np
import math
from kmedoids import fasterpam

def is_positive_semidefinite(matrix, tol=1e-10):
    """Check whether a symmetric matrix is positive semidefinite.

    Parameters
    ----------
    matrix : numpy.ndarray of shape (n, n)
        Matrix to be checked.
    tol : float, default=1e-10
        Numerical tolerance for slightly negative eigenvalues.

    Returns
    -------
    bool
        True if every eigenvalue is at least -tol, False otherwise.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues >= -tol)


def nearest_psd(matrix):
    """Replace negative eigenvalues to obtain a positive semidefinite matrix.

    The matrix is reconstructed from its eigenvectors and the eigenvalues
    after replacing negative eigenvalues with zero. The input is assumed
    to be symmetric.

    Parameters
    ----------
    matrix : numpy.ndarray of shape (n, n)
        Matrix to be processed.

    Returns
    -------
    psd_matrix : numpy.ndarray of shape (n, n)
        Matrix reconstructed using nonnegative eigenvalues.
    """

    eigvals, eigvecs = np.linalg.eigh(matrix)

    # Replace negative eigenvalues with zero
    eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)

    # Reconstruct the matrix
    psd_matrix = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    return psd_matrix

# The following three functions implement alternative k-medoids rules,
# not the medoid-based kMPS strategy reported in the paper.

# k_med_type=1: minimize the sum of distances to other clusters.
def select_points_min_sum_external_distance(K, labels, n_clusters):
    """Select one representative point from each cluster.

    For each cluster, choose the observation with the smallest sum of
    distances to all observations outside that cluster.

    Parameters
    ----------
    K : numpy.ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    labels : array-like of shape (n_samples,)
        Cluster labels returned by the k-medoids algorithm.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    numpy.ndarray of shape (n_clusters,)
        Indices of the selected observations.
    """
    selected_indices = []
    for k in range(n_clusters):
        cluster_indices = np.where(labels == k)[0]
        other_indices = np.where(labels != k)[0]
        sum_dists = []
        for idx in cluster_indices:
            dist_to_others = K[idx, other_indices]
            sum_dists.append(dist_to_others.sum())
        min_idx_local = np.argmin(sum_dists)
        min_idx_global = cluster_indices[min_idx_local]
        selected_indices.append(int(min_idx_global))
    return np.array(selected_indices)


# k_med_type=2: minimize summed distances to each other cluster.
def select_points_per_target_cluster(K, labels, n_clusters):
    """Select points based on distances to each other cluster.

    For each source cluster, identify a point whose sum of distances to
    each different target cluster is smallest. Repeated indices are then
    removed, so the number of selected points can vary.

    Parameters
    ----------
    K : numpy.ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    labels : array-like of shape (n_samples,)
        Cluster labels returned by the k-medoids algorithm.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    numpy.ndarray
        Sorted unique indices of the selected observations.
    """
    selected_points = []
    for k in range(n_clusters):
        cluster_k = np.where(labels == k)[0]
        for j in range(n_clusters):
            if j == k:
                continue
            cluster_j = np.where(labels == j)[0]
            sums = [(idx, K[idx, cluster_j].sum()) for idx in cluster_k]
            best_idx, _ = min(sums, key=lambda x: x[1])
            selected_points.append(int(best_idx))
    # Remove repeated indices
    return np.unique(selected_points)

# k_med_type=3: minimize the distance to the nearest point in each other cluster.
def select_points_min_min_distance_to_each_other_cluster(K, labels, n_clusters):
    """Select points using their minimum distances to other clusters.

    For each source cluster and every other target cluster, choose the
    source observation with the smallest distance to any observation in
    the target cluster. Repeated indices are then removed.

    Parameters
    ----------
    K : numpy.ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    labels : array-like of shape (n_samples,)
        Cluster labels returned by the k-medoids algorithm.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    numpy.ndarray
        Sorted unique indices of the selected observations.
    """
    selected_points = []
    for k in range(n_clusters):
        cluster_k = np.where(labels == k)[0]
        for j in range(n_clusters):
            if j == k:
                continue
            cluster_j = np.where(labels == j)[0]
            min_dists = []
            for idx in cluster_k:
                min_dist = K[idx, cluster_j].min()
                min_dists.append((idx, min_dist))
            best_idx, _ = min(min_dists, key=lambda x: x[1])
            selected_points.append(int(best_idx))
    # Remove repeated indices
    return np.unique(selected_points)

class Nystrom():
    """Select training observations for distance-based branch-node routing.

    The class returns prototype indices for the routing functions of
    Distance-based Soft Regression Trees (DSRTs). Despite its historical
    name, the selection used by the DSRT experiments does not construct
    a Nyström approximation of the full matrix.

    Available sampling options include uniform, k-medoids-based,
    divide-and-conquer ridge leverage, and recursive ridge leverage.
    """
    def __init__(self, n_components = 100, sampling = "uniform", ridge_leverage_lambda = 1,
                 random_state = None,type_matrix ='gower',  # For ridge leverage sampling, convert Gower distances to a kernel with exp(-G)
                 k_med_type = 0
    ):
        """Configure the prototype selection method.

       Parameters
       ----------
       n_components : int, default=100
           Requested number of selected observations.
       sampling : str, default='uniform'
           Selection strategy: 'uniform', 'kmediod', 'DAC', or 'recursive'.
           The spelling 'kmediod' is the identifier used by the code.
       ridge_leverage_lambda : float, default=1
           Parameter stored for divide-and-conquer ridge leverage sampling.
       random_state : int, RandomState, or None, default=None
           Random state used for uniform and k-medoids selection.
       type_matrix : str, default='gower'
           Type of input matrix. For ridge leverage methods, 'gower'
           triggers a conversion of distances to kernel similarities.
       k_med_type : int, default=0
           k-medoids selection rule: 0 uses the medoids (kMPS in the paper),
           while 1, 2, and 3 use alternative cluster-based rules.
       """

        self.type_matrix = type_matrix 
        self.n_components = n_components
        self.random_state = random_state
        self.sampling = sampling
        self.ridge_leverage_lambda = ridge_leverage_lambda
        self.sigma = 0.5 # Scale used to convert Gower distances to kernel similarities
        self.k_med_type = k_med_type
        
    def fit(self, K, l_idx):
        """Select prototype indices from the provided matrix.

        For uniform sampling, indices are chosen at random. For k-medoids,
        indices are obtained from the medoids or one of the alternative
        cluster-based rules. The other strategies use ridge leverage scores.
        This method returns indices only; it does not return a kernel matrix.

        Parameters
        ----------
        K : numpy.ndarray of shape (n_samples, n_samples)
            Distance matrix for uniform or k-medoids selection. The ridge
            leverage strategies handle the input according to type_matrix.
        l_idx : numpy.ndarray of shape (n_samples,)
            Original observation indices corresponding to the rows of K.

        Returns
        -------
        list
            Original observation indices obtained from l_idx at the selected
            positions.

        Notes
        -----
        The selected positions within K are also stored in index_selected,
        and the corresponding original indices in actual_indices.
        """

        
        rnd = check_random_state(self.random_state)
        n_samples = K.shape[0]

        # Restrict the number of selected prototypes to the dataset size
        if self.n_components > n_samples:
            n_components = n_samples
            msg = ("n_components > n_samples. This is not possible.\n"
                   "n_components was set to n_samples, which results"
                   " in inefficient evaluation of the full kernel.")
            logger_warning(msg)
            
        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)

        if self.sampling == "uniform":
            inds = rnd.permutation(n_samples)
            index_involved = inds[:n_components]
        elif self.sampling == "kmediod":
            
            result = fasterpam(K, n_components, random_state=self.random_state)
            labels = result.labels
            
            if self.k_med_type == 0:
                # kMPS in the paper: use the medoids as prototypes
                index_involved = result.medoids
            elif self.k_med_type == 1:
                index_involved = select_points_min_sum_external_distance(K, labels, n_components)
            elif self.k_med_type == 2:
                index_involved = select_points_per_target_cluster(K, labels, n_components)
            elif self.k_med_type == 3:
                index_involved = select_points_min_min_distance_to_each_other_cluster(K, labels, n_components)
        elif self.sampling == "DAC":
            
            approx_leverage_scores = self.DAC_ridge_leverage(K, self.ridge_leverage_lambda, n_components)

            
            # Sample according to normalized ridge leverage scores
            p = approx_leverage_scores/np.sum(approx_leverage_scores)  
            index_involved = np.random.choice(n_samples, size=n_components, replace=False, p=p)  
        elif self.sampling == "recursive":
            if self.type_matrix == 'gower':
                K = np.exp(-K/self.sigma)   
                
                if not is_positive_semidefinite(K):
                    K = nearest_psd(K)
            else:
                
                if not is_positive_semidefinite(K):
                    K_S = nearest_psd(K)
            index_involved = self.recursive_ridge_leverage(K, n_components)
        else:
            msg = "{self.sampling} is not a valid sampling strategy for Nyström method."
            logger_error(msg)
            raise ValueError(msg)
            
        self.index_selected = index_involved
        
        self.actual_indices = l_idx[index_involved]
        return self.actual_indices.tolist() # Return original observation indices

    def DAC_ridge_leverage(self, K, n_components, lambda_,):
        """Approximate ridge leverage scores using divide and conquer.

        The input observations are shuffled and processed in smaller blocks.
        Ridge leverage scores are approximated for each block separately.
        For Gower distances, each block is converted to kernel similarities
        before computing its scores.

        Parameters
        ----------
        K : numpy.ndarray of shape (n_samples, n_samples)
            Input distance or kernel matrix.
        n_components : int
            Number of observations processed in each block.
        lambda_ : float
            Regularization parameter, required to be nonnegative by the check.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Approximate ridge leverage scores.

        References
        ----------
        Farah Cherfaoui, Hachem Kadri, Liva Ralaivola.  Scalable ridge Leverage score sampling for the 
        Nyström method. ICASSP, May 2022, Singapour, Singapore.
        Based on the implementation of the authors: https://github.com/DAC-paper/Divide_And_Conquer_leverage_score_approximation 
        """
        
        if lambda_ < 0:
            msg = "'lambda_' parameter of DAC_ridge_leverage have to be positive."
            logger_error(msg)
            raise ValueError(msg)

        n = K.shape[0]
        ind = np.arange(n)
        np.random.shuffle(ind)
        approximated_ls = np.zeros((n))

        for l in range(0, math.ceil(n/n_components)):
            # Sample a subset of data
            true_sample_size = min(n_components, n - l*n_components)
            temp_ind = ind[l*n_components: l*n_components + true_sample_size]
            
            # compute the kernel matrix using the subset of selected data
            if self.type_matrix == 'gower':
                K_S = np.exp(-K[np.ix_(temp_ind,temp_ind)]/self.sigma)   
                
                if not is_positive_semidefinite(K_S):
                    K_S = nearest_psd(K_S)
            else:
                K_S = K[np.ix_(temp_ind,temp_ind)] 
                
                if not is_positive_semidefinite(K_S):
                    K_S = nearest_psd(K_S)
            # compute the approximated leverage score by inverting the small matrix
            approximated_ls[temp_ind] = np.sum(K_S * np.linalg.inv(K_S + lambda_ * np.eye(true_sample_size)) , axis = 1)

        return approximated_ls

    def recursive_ridge_leverage(self,
                                 K,
                                 n_components,
    ):
        """Select indices using recursive ridge leverage sampling.

        The algorithm works through increasingly large subsets of the input
        matrix and samples observations according to estimated leverage scores.

        Parameters
        ----------
        K : numpy.ndarray of shape (n_samples, n_samples)
            Input kernel matrix.
        n_components : int
            Requested number of selected observations.

        Returns
        -------
        numpy.ndarray
            Selected row indices relative to the input matrix.

        References
        ----------
        Cameron Musco, Christopher Musco. Recursive Sampling for the Nyström Method. NIPS 2017.
        https://doi.org/10.48550/arXiv.1605.07583
        Based on the implementation of Axel Vanraes: https://github.com/axelv/recursive-nystrom
        """

        n_samples = K.shape[0]
        n_oversample = np.log(n_components)
        k = np.ceil(n_components / (4 * n_oversample)).astype(np.integer)
        n_levels = np.ceil(np.log(n_samples / n_components) / np.log(2)).astype(np.integer)
        perm = np.random.permutation(n_samples)

        # set up sizes for recursive levels
        size_list = [n_samples]
        for l in range(1, n_levels+1):
            size_list += [np.ceil(size_list[l - 1] / 2).astype(np.integer)]

        # indices of poitns selected at previous level of recursion
        # at the base level it's just a uniform sample of ~ n_component points
        sample = np.arange(size_list[-1])
        indices = perm[sample]
        weights = np.ones((indices.shape[0],))

        # we need the diagonal of the whole kernel matrix
        k_diag = np.ones((n_samples,1))

        # Main recursion, unrolled for efficiency
        for l in reversed(range(n_levels)):
            # indices of current uniform sample
            current_indices = perm[:size_list[l]]
            # build sampled kernel

            # all rows and sampled columns
            KS = K[np.ix_(current_indices, indices)]
            SKS = KS[sample, :] # sampled rows and sampled columns

            # optimal lambda for taking O(k log(k)) samples
            if k >= SKS.shape[0]:
                # for the rare chance we take less than k samples in a round
                lmbda = 10e-6
                # don't set to exactly 0 to avoid stability issues
            else:
                # eigenvalues equal roughly the number of points per cluster, maybe this should scale with n?
                # can be interpret as the zoom level
                #lmbda = (np.sum(np.diag(SKS) * (weights ** 2))
                #        - np.sum(eigvalsh(SKS * weights[:,None] * weights[None,:], eigvals=(SKS.shape[0]-k, SKS.shape[0]-1))))/k
                lmbda = (np.sum(np.diag(SKS) * (weights ** 2)) 
         - np.sum(eigvalsh(SKS * weights[:, None] * weights[None, :])[-k:])) / k
            lmbda = np.maximum(lmbda, 1e-7)

            # compute and sample by lambda ridge leverage scores
            R = np.linalg.inv(SKS + np.diag(lmbda * weights ** (-2)))
            R = np.matmul(KS, R)
            #R = np.linalg.lstsq((SKS + np.diag(lmbda * weights ** (-2))).T,KS.T)[0].T
            if l != 0:
                # max(0, . ) helps avoid numerical issues, unnecessary in theory
                leverage_score = np.minimum(1.0, n_oversample * (1 / lmbda) * np.maximum(+0.0, (
                        k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
                # on intermediate levels, we independently sample each column
                # by its leverage score. the sample size is n_components in expectation
                sample = np.where(np.random.uniform(size=size_list[l]) < leverage_score)[0]
                # with very low probability, we could accidentally sample no
                # columns. In this case, just take a fixed size uniform sample
                if sample.size == 0:
                    leverage_score[:] = n_components / size_list[l]
                    sample = np.random.choice(size_list[l], size=n_components, replace=False)
                weights = np.sqrt(1. / leverage_score[sample])

            else:
                leverage_score = np.minimum(1.0, (1 / lmbda) * np.maximum(+0.0, (
                        k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
                p = leverage_score/leverage_score.sum()

                sample = np.random.choice(n_samples, size=n_components, replace=False, p=p)
            indices = perm[sample]

        return indices

