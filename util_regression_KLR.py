# -*- coding: utf-8 -*-
"""
Utility functions for Distance-based Soft Regression Trees (DSRTs).

This module provides functions for computing routing probabilities,
handling tree structures, and initializing branch nodes through
clustering.
"""

import numba
import math
import numpy as np
from numba import vectorize, jit, int64, float64, int32
from numba import types
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
from kmedoids import fasterpam


@vectorize([float64(float64,float64)],nopython=True)
def numba_log_part(x,C):
    """Compute the logarithmic term used to calculate a branch-node intercept.

    Parameters
    ----------
    x : float or array-like of float
        Coefficient value(s) for which the logarithmic term is evaluated.
    C : float
        Upper bound used in the logarithmic ratio.

    Returns
    -------
    float or numpy.ndarray
        Value(s) of log(x / (C - x)). No clipping is applied.
    """
    return np.log(x/(C-x))


@jit(float64(float64[:],float64[:],float64[:],float64,int64),nopython=True,fastmath=True)
def b_calc_snello(alp,y,f_K,C,le):
    """Compute a branch-node intercept from precomputed distance contributions.

    The calculation accumulates contributions from the first selected
    observations of each class, using an equal number from both classes (in this context the classes associated to branch nodes are the routing directions).

    Parameters
    ----------
    alp : numpy.ndarray
        Nonnegative coefficient values, one per observation.
    y : numpy.ndarray
        Binary labels with values -1 or 1 (left or right branch).
    f_K : numpy.ndarray
        Precomputed distance-based contributions, one per observation.
    C : float
        Upper bound used in the logarithmic term.
    le : int
        Number of observations used to determine the class counts.

    Returns
    -------
    float
        Intercept calculated from the selected observations.

    Notes
    -----
    The caller must provide consistent array lengths and enough observations
    from both classes; the function does not check these conditions.
    """

    b = 0
    log_part = numba_log_part(alp,C)

    s = (le - abs(np.sum(y)))//2 
    s_pos = s
    s_neg = s

    i = 0
    while (s_pos > 0 or s_neg > 0):
        
        
        if y[i] > 0 and s_pos > 0:
            b += log_part[i]*y[i] + f_K[i]
            s_pos -= 1
        elif y[i] < 0 and s_neg > 0:
            b += log_part[i]*y[i] + f_K[i]
            s_neg -= 1
        i+=1

    return b/(2*s)

@jit(float64(float64[:],float64[:],float64[:],float64[:,:],float64,int64),nopython=True,fastmath=True)
def b_calc(alp,y,ay,K,C,le):
    """Compute a branch-node intercept from a full coefficient-distance product.

    This is the original matrix-input counterpart of ``b_calc_snello``.
    It forms ``K.dot(ay)`` internally and averages contributions from
    an equal number of observations with positive and negative labels.

    Parameters
    ----------
    alp : numpy.ndarray of shape (n_samples,)
        Nonnegative coefficients used in the logarithmic term.
    y : numpy.ndarray of shape (n_samples,)
        Branch-direction labels, encoded as -1 and 1.
    ay : numpy.ndarray of shape (n_samples,)
        Signed routing coefficients multiplied by the matrix ``K``.
    K : numpy.ndarray of shape (n_samples, n_samples)
        Pairwise distance or similarity matrix used for branch routing.
    C : float
        Coefficient-bound parameter in ``log(alp / (C - alp))``.
    le : int
        Number of observations used to determine the class counts.

    Returns
    -------
    float
        Branch-node intercept computed from the selected contributions.

    Notes
    -----
    The function assumes valid, matching array dimensions and nonempty
    positive and negative label groups; it does not clip coefficients or
    guard against division by zero when the class-count term vanishes.
    The original numerical operations and Numba settings are preserved.
    """

    b = 0
    f_K = K.dot(ay)
    log_part = numba_log_part(alp,C)

    s = (le - abs(np.sum(y)))//2 
    s_pos = s
    s_neg = s

    i = 0
    while (s_pos > 0 or s_neg > 0):
        
        
        if y[i] > 0 and s_pos > 0:
            b += log_part[i]*y[i] + f_K[i]
            s_pos -= 1
        elif y[i] < 0 and s_neg > 0:
            b += log_part[i]*y[i] + f_K[i]
            s_neg -= 1
        i+=1

    return b/(2*s)


@jit(float64(float64[:],float64[:],float64,float64,int64),nopython=True,fastmath=True)
def Pr_it(K_i,ay,b,value,l):
    """Compute the routing probability for one observation at a branch node.

    Parameters
    ----------
    K_i : numpy.ndarray
        Distances between the observation and the selected training points.
    ay : numpy.ndarray
        Coefficients associated with those training points.
    b : float
        Branch-node intercept.
    value : float
        Parameter controlling the steepness of the logistic function.
    l : int
        Normalization factor used in the weighted distance sum.

    Returns
    -------
    float
        Routing probability, with an explicit safeguard for large
        positive values of the logistic exponent.
    """

    val = -value*(K_i.dot(ay)/l-b)
    
    if val > 700:
        return  0.
    else:
        return 1 / (1 + math.exp(val))
    
    
def erase_first(left,right):
    """Take the next node from two ordered lists of branch-node ancestors.

    The node with the smaller first index is selected. Its direction is
    -1 when taken from the left list and 1 when taken from the right list.

    Parameters
    ----------
    left : list of int
        Ordered ancestor indices associated with left branches.
    right : list of int
        Ordered ancestor indices associated with right branches.

    Returns
    -------
    list of int
        Selected node index and corresponding direction.

    Notes
    -----
    The selected element is removed from its input list. At least one of
    the two lists must be nonempty.
    """
    # -1 indicates a left branch; 1 indicates a right branch.
    direction = 0
    nodo = -1
    
    if not right:
        nodo = left[0]
        left.pop(0)
        direction = -1
        return [nodo,direction]
    if not left:
        nodo = right[0]
        right.pop(0)
        direction = 1    
        return [nodo,direction]
    if right[0] > left[0]:
        nodo = left[0]
        left.pop(0)
        direction = -1
    else:
        nodo = right[0]
        right.pop(0)
        direction = 1
    return [nodo,direction]

def alt_descendant(n,left=True,right=True):
    """Return the immediate children of a node in a binary tree.

    Parameters
    ----------
    n : int
        Zero-based index of the parent node.
    left : bool, default=True
        Whether to include the left child.
    right : bool, default=True
        Whether to include the right child.

    Returns
    -------
    list of int
        Zero-based indices of the requested children, left first.
    """
    result = []
    t = n+1
    if left:
        result.append(2*t)
    if right:
        result.append(2*t+1)
    return [i-1 for i in result]
 
    
def alt_anchesters(n, left=True, right=True):
    """Find ancestors reached through left and/or right branches.

    Parameters
    ----------
    n : int
        Zero-based index of the starting node.
    left : bool, default=True
        Whether to include ancestors reached through a left branch.
    right : bool, default=True
        Whether to include ancestors reached through a right branch.

    Returns
    -------
    list of int
        Zero-based indices of the selected ancestors, ordered from the
        closest ancestor toward the root.
    """
    result = []
    c = n+1
    while c != 1:
        a = math.floor(c/2)
        if left and c%2==0:
            result.append(a)
        if right and c%2!=0:
            result.append(a)
        c = a
    return [i-1 for i in result]

@jit(int64[:](int32,int32,int32),nopython=True)
def alt_anchesters_numba(n, left=1, right=1):
    """Find selected ancestors in a Numba-compatible representation.

    Parameters
    ----------
    n : int
        Zero-based index of the starting node.
    left : int, default=1
        Whether to include ancestors reached through a left branch.
    right : int, default=1
        Whether to include ancestors reached through a right branch.

    Returns
    -------
    numpy.ndarray of int64
        Selected ancestor indices ordered from the root toward the
        starting node.
    """
    result = []
    c = n+1
    while c != 1:
        a = math.floor(c/2)
        if left and c%2==0:
            result.append(a)
        if right and c%2!=0:
            result.append(a)
        c = a
    return np.array([i-1 for i in result][::-1])

@jit(int64[:](int64,int64,int64),nopython=True)
def alt_descendant_numba(n,left=1,right=1):
    """Return the immediate children of a node as an integer array.

    Parameters
    ----------
    n : int
        Zero-based index of the parent node.
    left : int, default=1
        Whether to include the left child.
    right : int, default=1
        Whether to include the right child.

    Returns
    -------
    numpy.ndarray of int64
        Zero-based indices of the requested children, left first.
    """
    result = []
    t = n+1
    if left:
        result.append(2*t-1)
    if right:
        result.append(2*t)
    return np.array(result)


def flat_list(l):
    """Flatten one level of a collection of lists.

    Parameters
    ----------
    l : iterable of iterables
        Groups of elements to concatenate.

    Returns
    -------
    list
        Elements of all groups, in their original order.
    """

    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list

@jit(int64[:](types.ListType(int64),types.ListType(int64)),nopython=True)
def erase_first_numba(left,right):
    """Take the next ancestor from two Numba typed lists.

    The node with the smaller first index is selected. Its direction is
    -1 for the left list and 1 for the right list.

    Parameters
    ----------
    left : numba.typed.List
        Ordered ancestor indices associated with left branches.
    right : numba.typed.List
        Ordered ancestor indices associated with right branches.

    Returns
    -------
    numpy.ndarray of int64
        Selected node index and corresponding direction.

    Notes
    -----
    The selected element is removed from its input list. At least one list
    must be nonempty.
    """
    # -1 indicates a left branch; 1 indicates a right branch.
    direction = 0
    nodo = -1
    
    if not right:
        nodo = left[0]
        left.pop(0)
        direction = -1
        return np.array([nodo,direction])
    if not left:
        nodo = right[0]
        right.pop(0)
        direction = 1    
        return np.array([nodo,direction])
    if right[0] > left[0]:
        nodo = left[0]
        left.pop(0)
        direction = -1
    else:
        nodo = right[0]
        right.pop(0)
        direction = 1
    return np.array([nodo,direction])

def subtree(index,depth,list_index):
    """Append all descendants of a node down to a given depth.

    Parameters
    ----------
    index : int
        Zero-based index of the subtree root.
    depth : int
        Number of descendant levels to traverse.
    list_index : list of int
        List to which descendant indices are appended in depth-first order.

    Returns
    -------
    None
        Descendant indices are added directly to ``list_index``.
    """

    if depth == 0:
        return 
    lista = alt_descendant(index)
    list_index.append(lista[0])
    list_index.append(lista[1])
    subtree(lista[0],depth-1,list_index)
    subtree(lista[1],depth-1,list_index)
    

# the difference with the one above is that here we hanlde directly the distance matrix REMINDER: SpectralClustering handles only similarity matrix  
def dict_clustering_new(distance_matrix,depth,assign_labels,seed):
    """Build a binary hierarchy of clusters using k-medoids on a matrix.

    At each level of the tree, the observations in every current node are partitioned
    into two clusters using their pairwise matrix entries.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Square pairwise matrix supplied to k-medoids as distances.
    depth : int
        Number of binary partitioning levels.
    assign_labels : str
        Accepted for compatibility; not used by the current implementation.
    seed : int
        Random state passed to k-medoids.

    Returns
    -------
    dict or int
        Dictionary mapping each tree level to its node-to-observation-index
        mapping; -1 if an exception occurs during clustering.
    """
    d = depth
    it = 0
    lis = [0]

    
    
    dic = {0:list(range(distance_matrix.shape[0]))}
    mega_dic = {}
    mega_dic[it] = dic
    try:
        while(d >= 1):
            
            temp_l = []
            
            new_dic = {}
            
            for i in lis:
                
                index_involved = dic[i]
                
                clust_method = fasterpam(distance_matrix[np.ix_(index_involved,index_involved)], 2, random_state=seed)
                temp_label = clust_method.labels
                
                
                left,right = alt_descendant(i)
                new_dic[left] = []
                new_dic[right] = []
                temp_l.append(left)
                temp_l.append(right)
                for k,j in enumerate(dic[i]):
                    if temp_label[k] == 0:
                        new_dic[left].append(j)
                    else:
                        new_dic[right].append(j)
                        
            it += 1
                 
            lis = temp_l.copy()
            dic = new_dic.copy()
            mega_dic[it] = new_dic.copy()
            d -= 1
    except:

        return -1
    return mega_dic



# The input matrix is also used to compute the silhouette score.
def clustering_init_new(distance_matrix,depth,seed,assign_labels='kmeans'):
    """Initialize a clustering tree and evaluate its leaf partition.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Square pairwise matrix passed to clustering and interpreted as
        distances when computing the silhouette score.
    depth : int
        Number of binary partitioning levels.
    seed : int
        Random state passed to k-medoids.
    assign_labels : str, default='kmeans'
        Forwarded to ``dict_clustering_new``; currently unused there.

    Returns
    -------
    tuple
        Cluster hierarchy and silhouette score, or (-1, -1) if the
        underlying clustering function returns -1.
    """
    lab = np.zeros(distance_matrix.shape[0])
    

    dic = dict_clustering_new(distance_matrix,depth,assign_labels,seed)
    if dic == -1:
        return -1,-1

    for i in dic[depth]:
        lab[dic[depth][i]] = i

    silhouette_avg = silhouette_score(distance_matrix, lab, metric='precomputed') # se la matrice è di distanze!
    
    return dic,silhouette_avg