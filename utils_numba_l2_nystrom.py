# -*- coding: utf-8 -*-
"""Numba-accelerated objectives and gradients for DSRT branch-node updates.

This module provides the objective functions and gradients used in the
Branch Node (BN) Step of Distance-based Soft Regression Trees (DSRTs).
It includes serial and parallel evaluations for root-node and subtree
optimization, with support for prototype-selected routing coefficients.
"""
import numpy as np
from numba import jit, float64, int64
from numba.types import DictType, ListType
from numba import types
from numba.typed import Dict, List
from numba import prange

def precompute_tree_structures_final(depth):
    """Build Numba-compatible tree structures used during the BN Step.

    Nodes use zero-based indices in a complete binary tree (i.e. the root node index is zero).
    The resulting mappings identify the leaves and the branch nodes on each root-to-leaf
    path, including the branches reached by taking a left or right turn.

    Parameters
    ----------
    depth : int
        Depth of the complete binary tree.

    Returns
    -------
    leaf_node : numba.typed.Dict
        Map from each leaf-node index to its position (0 to 2**depth - 1).
    estr : numpy.ndarray of int64, shape (2,)
        Indices of the first and last leaf nodes.
    BF_in_NL : numba.typed.Dict
        Branch-node ancestors of each leaf, ordered from root to leaf.
    BF_in_NL_L : numba.typed.Dict
        Ancestors at which the path to each leaf turns left.
    BF_in_NL_R : numba.typed.Dict
        Ancestors at which the path to each leaf turns right.
    """
    n_leaf = 2 ** depth
    n_branches = n_leaf - 1

    leaf_node = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i, j in enumerate(range(n_branches, 2 * n_branches + 1)):
        leaf_node[j] = i

    estr = np.array([min(leaf_node), max(leaf_node)], dtype=np.int64)

    list_type = types.ListType(types.int64)
    BF_in_NL = Dict.empty(key_type=types.int64, value_type=list_type)
    BF_in_NL_L = Dict.empty(key_type=types.int64, value_type=list_type)
    BF_in_NL_R = Dict.empty(key_type=types.int64, value_type=list_type)

    for t in leaf_node:
        # Left-turn ancestors
        result_l = List.empty_list(types.int64)
        c = t + 1
        while c != 1:
            a = c // 2
            if c % 2 == 0:
                result_l.append(a - 1)
            c = a
        BF_in_NL_L[t] = List(result_l[::-1]) if len(result_l) > 0 else List.empty_list(types.int64)

        # Right-turn ancestors
        result_r = List.empty_list(types.int64)
        c = t + 1
        while c != 1:
            a = c // 2
            if c % 2 != 0:
                result_r.append(a - 1)
            c = a
        BF_in_NL_R[t] = List(result_r[::-1]) if len(result_r) > 0 else List.empty_list(types.int64)

        # All ancestors
        result_both = List.empty_list(types.int64)
        c = t + 1
        while c != 1:
            a = c // 2
            result_both.append(a - 1)
            c = a
        BF_in_NL[t] = List(result_both[::-1]) if len(result_both) > 0 else List.empty_list(types.int64)

    return leaf_node, estr, BF_in_NL, BF_in_NL_L, BF_in_NL_R

simple_fun_dec = float64(
    float64[:],         # var
    float64[:,:],       # sq_resp
    float64[:, :],      # K
    int64,              # len_node
    int64[:],         # list_tree
    DictType(int64, int64),                     # leaf_node
    int64[:],                                   # estr
    DictType(int64, ListType(int64)),            # BF_in_NL
    DictType(int64, ListType(int64)),           # BF_in_NL_L
    DictType(int64, ListType(int64))            # BF_in_NL_R
)

simple_fun_dec_val = float64(
    float64[:],         # var
    float64[:,:],       # sq_resp
    float64[:, :],      # K
    int64,              # len_node
    int64[:],         # list_tree
    DictType(int64, int64),                     # leaf_node
    int64[:],                                   # estr
    DictType(int64, ListType(int64)),            # BF_in_NL
    DictType(int64, ListType(int64)),           # BF_in_NL_L
    DictType(int64, ListType(int64)) ,           # BF_in_NL_R
    float64,             #val sigmoide
    int64,                # number_samples
    float64              # l2 penalty
)

grad_simple_fun_dec = float64[:](
    float64[:],         # var
    float64[:,:],       # sq_resp
    float64[:, :],      # K
    int64,              # len_node
    int64[:],         # list_tree
    DictType(int64, int64),                     # leaf_node
    int64[:],                                   # estr
    DictType(int64, ListType(int64)),            # BF_in_NL
    DictType(int64, ListType(int64)),           # BF_in_NL_L
    DictType(int64, ListType(int64)),            # BF_in_NL_R
    float64              # l2 penalty
)

sig_fun = float64(
    float64[:],         # var
    float64[:],         # b_node
    float64[:,:],       # sq_resp
    float64[:, :],      # K
    float64[:,:],       # alp_y_node
    int64[:, :],        # index_node
    int64[:],           # len_node
    int64[:],         # list_tree
    DictType(int64, int64),                      # leaf_node
    int64[:],                                     # estr
    DictType(int64, ListType(int64)),            # BF_in_NL
    DictType(int64, ListType(int64)),            # BF_in_NL_L
    DictType(int64, ListType(int64)),            # BF_in_NL_R
    float64              # l2 penalty
)

sig_fun_val = float64(
    float64[:],         # var
    float64[:],         # b_node
    float64[:,:],       # sq_resp
    float64[:, :],      # K
    float64[:,:],       # alp_y_node
    int64[:, :],        # index_node
    int64[:],           # len_node
    int64[:],         # list_tree
    DictType(int64, int64),                      # leaf_node
    int64[:],                                     # estr
    DictType(int64, ListType(int64)),            # BF_in_NL
    DictType(int64, ListType(int64)),            # BF_in_NL_L
    DictType(int64, ListType(int64)),          # BF_in_NL_R
    float64,             #val sigmoide
    float64              # l2 penalty
)

grad_sig_fun = float64[:](
    float64[:],         # var
    float64[:],         # b_node
    float64[:,:],       # sq_resp
    float64[:, :],      # K
    float64[:,:],       # alp_y_node
    int64[:, :],        # index_node
    int64[:],           # len_node
    float64[:],         # list_tree
    DictType(int64, int64),                      # leaf_node
    int64[:],                                     # estr
    DictType(int64, ListType(int64)),            # BF_in_NL
    DictType(int64, ListType(int64)),            # BF_in_NL_L
    DictType(int64, ListType(int64)),           # BF_in_NL_R
    float64              # l2 penalty
)


grad_simple_fun_dec_val = float64[:](
    float64[:],         # var
    float64[:,:],       # sq_resp
    float64[:, :],      # K
    int64,              # len_node
    int64[:],         # list_tree
    DictType(int64, int64),                     # leaf_node
    int64[:],                                   # estr
    DictType(int64, ListType(int64)),            # BF_in_NL
    DictType(int64, ListType(int64)),           # BF_in_NL_L
    DictType(int64, ListType(int64)),            # BF_in_NL_R
    float64,                                      # val sigmoide
    int64,                               # l_samples
    float64              # l2 penalty
)

grad_sig_fun_val = float64[:](
    float64[:],         # var
    float64[:],         # b_node
    float64[:,:],       # sq_resp
    float64[:, :],      # K
    float64[:,:],       # alp_y_node
    int64[:, :],        # index_node
    int64[:],           # len_node
    int64[:],         # list_tree
    DictType(int64, int64),                      # leaf_node
    int64[:],                                    # estr
    DictType(int64, ListType(int64)),            # BF_in_NL
    DictType(int64, ListType(int64)),            # BF_in_NL_L
    DictType(int64, ListType(int64)),            # BF_in_NL_R
    float64,                                      # val sigmoide
    float64              # l2 penalty
)



@jit(sig_fun_val,nopython=True,fastmath=True)
def simple_root_val(var,b_node,sq_resp,K,alp_y_node,index_node,len_node,list_tree,leaf_node,estr,BF_in_NL,BF_in_NL_L,BF_in_NL_R,value,l2_penalty):
    """Evaluate the regularized root-node objective (serial version).

    Only the root node coefficients and intercept are optimized; all
    other branch-node parameters are held fixed. The loss is assembled
    from squared leaf prediction errors weighted by routing probabilities.

    Parameters
    ----------
    var : numpy.ndarray
        Active root coefficients, followed by the root intercept.
    b_node : numpy.ndarray
        Intercepts of the other branch nodes, in node-index order.
    sq_resp : numpy.ndarray, shape (n_leaves, n_samples)
        Squared prediction errors, by leaf and training observation.
    K : numpy.ndarray, shape (n_samples, n_samples)
        Training dissimilarity matrix.
    alp_y_node : numpy.ndarray
        Coefficients for all branch nodes; non-root rows remain fixed.
    index_node : numpy.ndarray of int64
        Observation indices associated with each branch node.
    len_node : numpy.ndarray of int64
        Number of active indices at each branch node.
    list_tree : numpy.ndarray of int64
        Tree depth, total training observations, and active root variables.
    leaf_node, BF_in_NL, BF_in_NL_L, BF_in_NL_R : numba.typed.Dict
        Leaf indices and root-to-leaf path information.
    estr : numpy.ndarray
        First and last leaf-node indices.
    value : float
        Steepness parameter of the routing logistic function.
    l2_penalty : float
        Weight applied to the squared root coefficients, if positive.

    Returns
    -------
    float
        Weighted sum of squared errors plus the L2 penalty.
    """
    depth = int(list_tree[0])
    number_samples = int(list_tree[1])
    n_alp_var = int(list_tree[2])
    alp_y = np.ascontiguousarray(var[:-1])
    b_0 = var[-1]
    
    
    KK = np.ascontiguousarray(K) 
    
    output = 0
    
    for i in range(number_samples):
        single_output = 0
        for tl in leaf_node:
            
            mini_output = 1
            if tl != estr[1]:
                for t in BF_in_NL_L[tl]:
                    idx = index_node[t][:len_node[t]]
                    if t == 0:

                        val = -value*(KK[i,idx].dot(alp_y)/number_samples-b_0)
                        if val > 700:
                            mini_output *=  0.
                        else:
                            mini_output *= 1 / (1 + np.exp(val))

                    else:
                        val = -value*(KK[i,idx].dot(alp_y_node[t][idx])/number_samples-b_node[t-1])
                        if val > 700:
                            mini_output *=  0.
                        else:
                            mini_output *= 1 / (1 + np.exp(val))
                            
                        
            if tl != estr[0]:
                for tr in BF_in_NL_R[tl]:
                    idx = index_node[tr][:len_node[tr]]
                    if tr == 0:
                        val = -value*(KK[i,idx].dot(alp_y)/number_samples-b_0)
                        if val > 700:
                            mini_output *=  1.
                        else:
                            mini_output *= 1. - (1 / (1 + np.exp(val)))
                        

                    else:
                        val = -value*(KK[i,idx].dot(alp_y_node[tr][idx])/number_samples-b_node[tr-1])
                        if val > 700:
                            mini_output *=  1.
                        else:
                            mini_output *= 1. - (1 / (1 + np.exp(val)))

            single_output += mini_output*sq_resp[leaf_node[tl]][i]
        output += single_output
    if l2_penalty > 0.:
        
        output += l2_penalty*sum(alp_y**2)
    
    return output#/number_samples


@jit(grad_sig_fun_val,nopython=True,fastmath=True)
def grad_simple_root_val(var,b_node,sq_resp,K,alp_y_node,index_node,len_node,list_tree,leaf_node,estr,BF_in_NL,BF_in_NL_L,BF_in_NL_R,value,l2_penalty):
    
    """Evaluate the root-node objective gradient (serial version).

    Parameters
    ----------
    var : numpy.ndarray
        Active root coefficients, followed by the root intercept.
    b_node : numpy.ndarray
        Intercepts of the other branch nodes.
    sq_resp : numpy.ndarray
        Squared leaf prediction errors for each training observation.
    K : numpy.ndarray
        Training dissimilarity matrix.
    alp_y_node : numpy.ndarray
        Current coefficients for all branch nodes.
    index_node : numpy.ndarray of int64
        Active observation indices at each branch node.
    len_node : numpy.ndarray of int64
        Number of active indices per node.
    list_tree : numpy.ndarray of int64
        Tree depth, training sample count, and active root variable count.
    leaf_node, BF_in_NL, BF_in_NL_L, BF_in_NL_R : numba.typed.Dict
        Leaf mapping and path information.
    estr : numpy.ndarray
        First and last leaf-node indices.
    value : float
        Steepness of the routing logistic function.
    l2_penalty : float
        L2 regularization weight, if positive.

    Returns
    -------
    numpy.ndarray
        Gradient entries for the active root coefficients and intercept.
    """

    depth = int(list_tree[0])
    number_samples = int(list_tree[1])
    n_alp_var = int(list_tree[2])
    alp_y = np.ascontiguousarray(var[:-1])
    b_0 = var[-1]
    
    
    KK = np.ascontiguousarray(K) 
    grad = np.zeros(int(n_alp_var) + 1)
    
    if l2_penalty > 0.:
        grad[:int(number_samples)] = 2*l2_penalty*alp_y
    
    for i in range(number_samples):

        for tl in leaf_node:
            
            mini_output = 1
            if tl != estr[1]:
                for t in BF_in_NL_L[tl]:
                    idx = index_node[t][:len_node[t]]
                    if t == 0:
                        idx_0 = idx
                        val = -value*(KK[i,idx].dot(alp_y)/number_samples-b_0)
                        if val > 700:
                            mini_output *=  0.
                        else:
                            mini_output *= -np.exp(val) / (1 + np.exp(val))**2

                    else:
                        val = -value*(KK[i,idx].dot(alp_y_node[t][idx])/number_samples-b_node[t-1])
                        if val > 700:
                            mini_output *=  0.
                        else:
                            mini_output *= 1 / (1 + np.exp(val))
                            
                        
            if tl != estr[0]:
                for tr in BF_in_NL_R[tl]:
                    idx = index_node[tr][:len_node[tr]]
                    if tr == 0:
                        idx_0 = idx
                        val = -value*(KK[i,idx].dot(alp_y)/number_samples-b_0)
                        if val > 700:
                            mini_output *=  0.
                        else:
                            mini_output *= np.exp(val) / (1 + np.exp(val))**2
                        

                    else:
                        val = -value*(KK[i,idx].dot(alp_y_node[tr][idx])/number_samples-b_node[tr-1])
                        if val > 700:
                            mini_output *=  1.
                        else:
                            mini_output *= 1. - (1 / (1 + np.exp(val)))
            
            single_output = value*mini_output*sq_resp[leaf_node[tl]][i]/number_samples
            grad[:number_samples] += -single_output*KK[i,idx_0]
            grad[-1] += single_output*number_samples 

        
    return grad



@jit(grad_sig_fun_val, nopython=True, fastmath=True, parallel=True)
def grad_simple_root_parallel_val(var, b_node, sq_resp, K, alp_y_node, index_node, len_node,
                     list_tree, leaf_node, estr,BF_in_NL, BF_in_NL_L, BF_in_NL_R,value,l2_penalty):
    """Evaluate the root-node objective gradient in parallel.

    The implementation accumulates contributions for different training
    observations separately and sums them after the parallel loop.

    Parameters
    ----------
    var : numpy.ndarray
        Active root coefficients, followed by the root intercept.
    b_node : numpy.ndarray
        Intercepts of the other branch nodes.
    sq_resp : numpy.ndarray
        Squared prediction errors for all leaves and observations.
    K : numpy.ndarray
        Training dissimilarity matrix.
    alp_y_node : numpy.ndarray
        Current branch-node coefficients, including fixed nodes.
    index_node : numpy.ndarray of int64
        Active observation indices for each branch node.
    len_node : numpy.ndarray of int64
        Number of active indices per node.
    list_tree : numpy.ndarray of int64
        Tree depth, training sample count, and active root variable count.
    leaf_node, BF_in_NL, BF_in_NL_L, BF_in_NL_R : numba.typed.Dict
        Leaf mapping and path information.
    estr : numpy.ndarray
        First and last leaf-node indices.
    value : float
        Steepness of the routing logistic function.
    l2_penalty : float
        L2 regularization weight, if positive.

    Returns
    -------
    numpy.ndarray
        Gradient entries for the active root coefficients and intercept.
    """

    depth = int(list_tree[0])
    number_samples = int(list_tree[1])
    n_alp_var = int(list_tree[2])
    b_0 = var[-1]
    
    alp_y = np.ascontiguousarray(var[:-1])
    KK = np.ascontiguousarray(K)
    
    # Each parallel iteration accumulates its own gradient
    grad_local = np.zeros((number_samples, n_alp_var + 1))
    
    for i in prange(number_samples):
        local_grad = np.zeros(n_alp_var + 1)
        
        for tl in leaf_node:
            mini_output = 1.0
            
            if tl != estr[1]:
                for t in BF_in_NL_L[tl]:
                    idx = index_node[t][:len_node[t]]
                    if t == 0:
                        idx_0 = idx
                        val = -value*(KK[i, idx].dot(alp_y)/number_samples - b_0)
                        if val > 700:
                            mini_output *= 0.
                        else:
                            exp_val = np.exp(val)
                            mini_output *= -exp_val / (1. + exp_val)**2
                    else:
                        
                        val = -value*(KK[i, idx].dot(alp_y_node[t][idx])/number_samples - b_node[t - 1])
                        if val > 700:
                            mini_output *= 0.
                        else:
                            mini_output *= 1. / (1. + np.exp(val))
            
            if tl != estr[0]:
                for tr in BF_in_NL_R[tl]:
                    idx = index_node[tr][:len_node[tr]]
                    if tr == 0:
                        idx_0 = idx
                        val = -value*(KK[i, idx].dot(alp_y)/number_samples - b_0)
                        if val > 700:
                            mini_output *= 0.
                        else:
                            exp_val = np.exp(val)
                            mini_output *= exp_val / (1. + exp_val)**2
                    else:
                        
                        val = -value*(KK[i, idx].dot(alp_y_node[tr][idx])/number_samples - b_node[tr - 1])
                        if val > 700:
                            mini_output *= 1.
                        else:
                            mini_output *= 1. - (1. / (1. + np.exp(val)))
            
            single_output = value*mini_output * sq_resp[leaf_node[tl]][i]/number_samples
            local_grad[:n_alp_var] += -single_output * KK[i, idx_0]
            local_grad[-1] += single_output*number_samples
        
        grad_local[i, :] = local_grad
    
    # Sum the per-observation gradients after the parallel loop
    grad = np.sum(grad_local, axis=0)
    
    if l2_penalty > 0.:
        grad[:n_alp_var]+= 2*l2_penalty*alp_y
    
    return grad




@jit(simple_fun_dec_val,nopython=True,fastmath=True)
def simple_dec_val(var,sq_resp,K,len_node,list_tree, leaf_node, estr,BF_in_NL, BF_in_NL_L, BF_in_NL_R,value,l_samples,l2_penalty):
       
    """Evaluate the regularized branch-node subtree objective (serial).

    The variables describe the routing functions of all branch nodes in
    the currently optimized subtree.

    Parameters
    ----------
    var : numpy.ndarray
        Active coefficients of successive subtree branch nodes, followed
        by one intercept per node.
    sq_resp : numpy.ndarray, shape (n_subtree_leaves, n_working_samples)
        Squared leaf prediction errors on the current working set.
    K : numpy.ndarray, shape (n_working_samples, len_node)
        Dissimilarities from working-set observations to active prototypes.
    len_node : int
        Number of active coefficients per subtree branch node.
    list_tree : numpy.ndarray of int64
        Subtree depth, working-set count times the number of branch nodes,
        and working-set count.
    leaf_node, BF_in_NL, BF_in_NL_L, BF_in_NL_R : numba.typed.Dict
        Leaf mapping and paths within the subtree.
    estr : numpy.ndarray
        First and last subtree leaf-node indices.
    value : float
        Steepness of the routing logistic function.
    l_samples : int
        Total training sample count used to normalize routing scores.
    l2_penalty : float
        Weight of the squared coefficient penalty, if positive.

    Returns
    -------
    float
        Weighted sum of squared leaf errors plus L2 regularization.
    """
    depth = int(list_tree[0])
    samples_per_nodes = int(list_tree[1]) # working-set size multiplied by the number of branch nodes
    len_samples = int(list_tree[2])   # working-set size; equals len_node without prototype selection, number of samples in the working set; 
                                      # if I do not apply prototype selection (kMPS or uniform), len_samples = len_node

    
    KK = np.ascontiguousarray(K)
    number_br_node = 2**depth - 1

    total_len_node = len_node*number_br_node   # equals samples_per_nodes without prototype selection

    alp_y = np.ascontiguousarray(var[:-number_br_node])
    KK = np.ascontiguousarray(K)
    b = var[-number_br_node:]
    
    output = 0
    
    
    
    for i in range(len_samples):
        single_output = 0
        for tl in leaf_node:
            
            mini_output = 1
            if tl != estr[1]:
                for t in BF_in_NL_L[tl]:
                    val = -value*(KK[i,:].dot(alp_y[(t*len_node):(t*len_node)+len_node])/l_samples-b[t])
                    if val > 700:
                        mini_output *=  0.
                    else:
                        mini_output *= 1 / (1 + np.exp(val))

            if tl != estr[0]:
                for tr in BF_in_NL_R[tl]:
                    val = -value*(KK[i,:].dot(alp_y[(tr*len_node):(tr*len_node)+len_node])/l_samples-b[tr])
                    if val > 700:
                        mini_output *=  1.
                    else:
                        mini_output *= 1. - (1 / (1 + np.exp(val)))
                        
            single_output += mini_output*sq_resp[leaf_node[tl]][i]
        output += single_output
        
    if l2_penalty > 0.:
        output += l2_penalty*sum(alp_y**2)
    return output


@jit(simple_fun_dec_val, nopython=True, fastmath=True, parallel=True)
def simple_dec_parallel_val(var, sq_resp, K, len_node, list_tree, leaf_node, estr,BF_in_NL, BF_in_NL_L, BF_in_NL_R,value,l_samples,l2_penalty):
    """Evaluate the regularized subtree objective in parallel.

    This is the parallel counterpart of ``simple_dec_val``; independent
    working-set contributions are summed at the end.

    Parameters
    ----------
    var : numpy.ndarray
        Subtree branch-node coefficients followed by their intercepts.
    sq_resp : numpy.ndarray
        Squared leaf prediction errors over the current working set.
    K : numpy.ndarray
        Working-set-to-prototype dissimilarity matrix.
    len_node : int
        Active coefficient count for each subtree branch node.
    list_tree : numpy.ndarray of int64
        Subtree depth, working-set count times branch-node count,
        and working-set count.
    leaf_node, BF_in_NL, BF_in_NL_L, BF_in_NL_R : numba.typed.Dict
        Leaf mapping and subtree path information.
    estr : numpy.ndarray
        First and last subtree leaf-node indices.
    value : float
        Steepness of the routing logistic function.
    l_samples : int
        Total training sample count used for score normalization.
    l2_penalty : float
        L2 regularization weight, if positive.

    Returns
    -------
    float
        Weighted sum of squared leaf errors plus L2 regularization.
    """

    depth = int(list_tree[0])
    samples_per_nodes = int(list_tree[1]) # working-set size multiplied by the number of branch nodes
    len_samples = int(list_tree[2])   # working-set size; equals len_node without prototype selection
    
    KK = np.ascontiguousarray(K)
    number_br_node = 2**depth - 1

    total_len_node = len_node*number_br_node  # equals samples_per_nodes without prototype selection


    b = var[-number_br_node:]
    
    alp_y = np.ascontiguousarray(var[:-number_br_node])
    KK = np.ascontiguousarray(K)
    
    partial_outputs = np.zeros(len_samples)  # Store each working-set contribution separately one for each observation

    for i in prange(len_samples):
        single_output = 0.0
        for tl in leaf_node:
            mini_output = 1.0
            if tl != estr[1]:
                for t in BF_in_NL_L[tl]:
                    start = t * len_node
                    val = -value*(KK[i, :].dot(alp_y[start:start + len_node])/l_samples - b[t])
                    if val > 700:
                        mini_output *= 0.
                    else:
                        mini_output *= 1. / (1. + np.exp(val))

            if tl != estr[0]:
                for tr in BF_in_NL_R[tl]:
                    start = tr * len_node
                    val = -value*(KK[i, :].dot(alp_y[start:start + len_node])/l_samples - b[tr])
                    if val > 700:
                        mini_output *= 1.
                    else:
                        mini_output *= 1. - (1. / (1. + np.exp(val)))

            single_output += mini_output * sq_resp[leaf_node[tl]][i]

        partial_outputs[i] = single_output
    
    res = np.sum(partial_outputs)
    
    if l2_penalty > 0.:
        res += l2_penalty*sum(alp_y**2)
    
    return res



@jit(grad_simple_fun_dec_val,nopython=True,fastmath=True)
def grad_dec_nuovo_val(var, sq_resp, K, len_node, list_tree, leaf_node, estr,BF_in_NL, BF_in_NL_L, BF_in_NL_R,value,l_samples,l2_penalty):
    """Evaluate the branch-node subtree gradient (serial version).

    Parameters
    ----------
    var : numpy.ndarray
        Active coefficients and intercepts of all subtree branch nodes.
    sq_resp : numpy.ndarray
        Squared leaf prediction errors for the current working set.
    K : numpy.ndarray
        Dissimilarities from working-set samples to active prototypes.
    len_node : int
        Number of active coefficients per subtree branch node.
    list_tree : numpy.ndarray of int64
        Subtree depth, working-set count times branch-node count,
        and working-set count.
    leaf_node, BF_in_NL, BF_in_NL_L, BF_in_NL_R : numba.typed.Dict
        Leaf mapping and subtree path information.
    estr : numpy.ndarray
        First and last subtree leaf-node indices.
    value : float
        Steepness of the routing logistic function.
    l_samples : int
        Total training sample count used for score normalization.
    l2_penalty : float
        L2 regularization weight, if positive.

    Returns
    -------
    numpy.ndarray
        Gradient with respect to all active subtree coefficients and
        intercepts, in the same order as ``var``.
    """
    
    depth = int(list_tree[0])
    samples_per_nodes = int(list_tree[1]) # working-set size (number of observations) multiplied by the number of branch nodes
    len_samples = int(list_tree[2])   # working-set size; equals len_node without prototype selection
    
    KK = np.ascontiguousarray(K)
    number_br_node = 2**depth - 1

    total_len_node = len_node*number_br_node  # equals samples_per_nodes without prototype selection

    alp_y = np.ascontiguousarray(var[:-number_br_node])
    KK = np.ascontiguousarray(K)
    b = var[-number_br_node:]

    
    gr = np.zeros(total_len_node+number_br_node)


       
    for k in range(samples_per_nodes):


        i = k % len_samples
        b_idx = int(k/len_samples)
            
        for tl in leaf_node:
            anchest = False
            for element in BF_in_NL[tl]:
                if element == b_idx:
                    anchest = True
            if anchest == True:
                    
                dentro = False
                if tl!= estr[1]:
                    for el in BF_in_NL_L[tl]:
                        if el == b_idx:
                            dentro = True
                start = b_idx * len_node
                val = -value*(KK[i, :].dot(alp_y[start:start + len_node])/l_samples - b[b_idx])
                if val > 700:
                    mini_output = 0.
                else:
                    exp_val = np.exp(val)
                    mini_output = exp_val / (1. + exp_val)**2
                    
                if dentro:
                    mini_output *= -1 
                else:
                    mini_output *= 1
                    
                if tl != estr[1]:
                    for t in BF_in_NL_L[tl]:
                        if t!= b_idx:
                            start = t * len_node
                            val = -value*(KK[i, :].dot(alp_y[start:start + len_node])/l_samples - b[t])
                            if val > 700:
                                mini_output *= 0.
                            else:
                                mini_output *= 1. / (1. + np.exp(val))
                            

                if tl != estr[0]:
                    for tr in BF_in_NL_R[tl]:
                        if tr!= b_idx:
                            start = tr * len_node
                            val = -value*(KK[i, :].dot(alp_y[start:start + len_node])/l_samples - b[tr])
                            if val > 700:
                                mini_output *= 1.
                            else:
                                mini_output *=   np.exp(val)/ (1. + np.exp(val))

                single_output = value*mini_output * sq_resp[leaf_node[tl]][i]/l_samples
                gr[b_idx*len_node:b_idx*len_node+len_node] += -single_output * KK[i, :]
                gr[-number_br_node+b_idx] += single_output*l_samples
      
    if l2_penalty >0.:
        gr[:-number_br_node] += 2*l2_penalty*alp_y
    return gr





@jit(sig_fun_val, nopython=True, fastmath=True, parallel=True)
def simple_root_parallel_val(var, b_node, sq_resp, K, alp_y_node, index_node, len_node, list_tree, leaf_node, estr,BF_in_NL, BF_in_NL_L, BF_in_NL_R,value,l2_penalty):
    """Evaluate the regularized root-node objective in parallel.

    Each observation contributes a weighted squared prediction error;
    the contributions are summed after the parallel loop.

    Parameters
    ----------
    var : numpy.ndarray
        Active root coefficients, followed by the root intercept.
    b_node : numpy.ndarray
        Intercepts of non-root branch nodes.
    sq_resp : numpy.ndarray, shape (n_leaves, n_samples)
        Squared leaf prediction errors for the training observations.
    K : numpy.ndarray
        Training dissimilarity matrix.
    alp_y_node : numpy.ndarray
        Coefficients of all branch nodes, including fixed non-root nodes.
    index_node : numpy.ndarray of int64
        Active observation indices for each branch node.
    len_node : numpy.ndarray of int64
        Number of active indices per node.
    list_tree : numpy.ndarray of int64
        Tree depth, training sample count, and active root variable count.
    leaf_node, BF_in_NL, BF_in_NL_L, BF_in_NL_R : numba.typed.Dict
        Leaf mapping and root-to-leaf path information.
    estr : numpy.ndarray
        First and last leaf-node indices.
    value : float
        Steepness of the routing logistic function.
    l2_penalty : float
        L2 regularization weight, if positive.

    Returns
    -------
    float
        Weighted sum of squared errors plus the L2 penalty.
    """
    depth = list_tree[0]
    number_samples = int(list_tree[1])
    n_alp_var = int(list_tree[2])

    alp_y = np.ascontiguousarray(var[:-1])
    b_0 = var[-1]


    KK = np.ascontiguousarray(K) 

    result = np.zeros(number_samples)

    for i in prange(number_samples):
        single_output = 0.0

        for tl in leaf_node:
            mini_output = 1.0

            if tl != estr[1]:
                for t in BF_in_NL_L[tl]:
                    idx = index_node[t][:len_node[t]]
                    if t == 0:
                        val = -value*(KK[i, idx].dot(alp_y)/number_samples - b_0)
                    else:
                        val = -value*(KK[i, idx].dot(alp_y_node[t, idx])/number_samples - b_node[t - 1])

                    val = min(val, 700.0)
                    mini_output *= 1.0 / (1.0 + np.exp(val))

            if tl != estr[0]:
                for tr in BF_in_NL_R[tl]:
                    idx = index_node[tr][:len_node[tr]]
                    if tr == 0:
                        val = -value*(KK[i, idx].dot(alp_y)/number_samples - b_0)
                    else:
                        val = -value*(KK[i, idx].dot(alp_y_node[tr, idx])/number_samples - b_node[tr - 1])

                    val = min(val, 700.0)
                    mini_output *= 1.0 - (1.0 / (1.0 + np.exp(val)))

            single_output += mini_output * sq_resp[leaf_node[tl], i]

        result[i] = single_output

    res = result.sum()
    
    if l2_penalty > 0.:
        res += l2_penalty*sum(alp_y**2)
    return res 


@jit(grad_simple_fun_dec_val,nopython=True,fastmath=True,parallel=True)
def grad_dec_nuovo_parallel_val(var, sq_resp, K, len_node, list_tree, leaf_node, estr, BF_in_NL, BF_in_NL_L, BF_in_NL_R,value,l_samples,l2_penalty):
    """Evaluate the branch-node subtree gradient in parallel.

    Contributions to the gradient are computed for separate observations
    and branch nodes, then summed after the parallel loop.

    Parameters
    ----------
    var : numpy.ndarray
        Active coefficients of all subtree branch nodes, then intercepts.
    sq_resp : numpy.ndarray
        Squared leaf prediction errors for the current working set.
    K : numpy.ndarray
        Working-set-to-prototype dissimilarity matrix.
    len_node : int
        Number of active coefficients per subtree branch node.
    list_tree : numpy.ndarray of int64
        Subtree depth, working-set count times branch-node count,
        and working-set count.
    leaf_node, BF_in_NL, BF_in_NL_L, BF_in_NL_R : numba.typed.Dict
        Leaf mapping and subtree path information.
    estr : numpy.ndarray
        First and last subtree leaf-node indices.
    value : float
        Steepness of the routing logistic function.
    l_samples : int
        Total training sample count used for score normalization.
    l2_penalty : float
        L2 regularization weight, if positive.

    Returns
    -------
    numpy.ndarray
        Gradient with respect to active subtree coefficients and
        intercepts, in the same order as ``var``.
    """

    depth = int(list_tree[0])
    samples_per_nodes = int(list_tree[1]) # working-set size multiplied by the number of branch nodes
    len_samples = int(list_tree[2])   # working-set size; equals len_node without prototype selection
    
    KK = np.ascontiguousarray(K)
    number_br_node = 2**depth - 1

    total_len_node = len_node*number_br_node  # equals samples_per_nodes without prototype selection

    alp_y = np.ascontiguousarray(var[:-number_br_node])
    b = var[-number_br_node:]

    # Store partial gradients separately to avoid concurrent writes
    gr_tmp = np.zeros((samples_per_nodes, total_len_node + number_br_node))
    

    for k in prange(samples_per_nodes):
        local_grad = np.zeros(total_len_node + number_br_node)
        i = k % len_samples
        b_idx = int(k / len_samples)
            
        for tl in leaf_node:
            anchest = False
            for element in BF_in_NL[tl]:
                if element == b_idx:
                    anchest = True
            if anchest:
                dentro = False
                if tl != estr[1]:
                    for el in BF_in_NL_L[tl]:
                        if el == b_idx:
                            dentro = True

                start = b_idx * len_node
                val = -value*(KK[i, :].dot(alp_y[start:start + len_node])/l_samples - b[b_idx])
                if val > 700:
                    mini_output = 0.
                else:
                    exp_val = np.exp(val)
                    mini_output = exp_val / (1. + exp_val)**2
                    
                if dentro:
                    mini_output *= -1
                    
                if tl != estr[1]:
                    for t in BF_in_NL_L[tl]:
                        if t != b_idx:
                            start = t * len_node
                            val = -value*(K[i, :].dot(alp_y[start:start + len_node])/l_samples - b[t])
                            if val <= 700:
                                mini_output *= 1. / (1. + np.exp(val))

                if tl != estr[0]:
                    for tr in BF_in_NL_R[tl]:
                        if tr != b_idx:
                            start = tr * len_node
                            val = -value*(K[i, :].dot(alp_y[start:start + len_node])/l_samples - b[tr])
                            if val <= 700:
                                mini_output *= np.exp(val) / (1. + np.exp(val))
                                
                single_output = value*mini_output * sq_resp[leaf_node[tl]][i]/l_samples
                local_grad[b_idx*len_node:b_idx*len_node+len_node] += -single_output * KK[i, :]
                local_grad[-number_br_node+b_idx] += single_output*l_samples

        gr_tmp[k, :] = local_grad

    # Sum the partial gradients after the parallel loop
    gr = np.sum(gr_tmp, axis=0)
    
    if l2_penalty > 0.:
        gr[:-number_br_node] += 2*l2_penalty*alp_y
    
    return gr