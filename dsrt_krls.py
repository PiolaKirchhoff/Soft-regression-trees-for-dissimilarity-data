# -*- coding: utf-8 -*-
"""DSRT extension with Kronecker regularized least-squares leaf models.

This module extends the distance-based soft regression tree (DSRT) of
``dsrt.py`` rather than providing an independent duplicate. It retains
logistic distance-based branch-node routing and the node-based
decomposition, while additionally supporting ``type_reg='kkls'`` for
pair-input prediction (Sections 3.3.2 and 5.4 of the paper).

In the KRLS setting, the response is a matrix of interactions between
two entity types. A single tree routes entities from the first type;
each leaf estimates its interaction matrix with the second type using
``rlscore.learner.KronRLS``. The experiments use setting B (drugs routed)
or setting C (targets routed), depth two, and no prototype selection.

The convergence safeguards discussed in the paper, including
Armijo-based reference updates, are not enforced in this
experimental implementation.
"""

from functools import reduce
import operator
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from scipy.optimize import lsq_linear,Bounds
from scipy import optimize
from util_regression_KLR import *
from sklearn.metrics.pairwise import polynomial_kernel,rbf_kernel,linear_kernel
import numpy as np
import SMO
import pandas as pd
from utils_numba_l2_nystrom import *
import time
from scipy.optimize import LinearConstraint
from scipy.optimize import BFGS
from rlscore.learner import KronRLS
from nystrom_utils_2 import *

class SSRT:
    """Distance-based soft regression tree with optional KRLS leaf regressions.
    
    This class shares branch-node routing and decomposition procedures with
    ``dsrt.SSRT``, but extends the leaf-node models to learn interactions
    between two types of entities through Kronecker regularized least
    squares. The regression mode is selected by ``set_dataset()``.
    
    Parameters
    ----------
    depth : int, default=2
        Number of branch decisions along each root-to-leaf path.
    
    Notes
    -----
    In ``'kkls'`` mode, ``self.K`` is the first-entity matrix used for
    routing, ``self.P`` is the second-entity matrix, and ``self.y_train``
    is a two-dimensional interaction-response matrix. Each leaf stores its
    own coefficient matrix and an index set for the first entity type.
    For settings B and C the script changes which entity type is routed,
    without requiring a different tree implementation.
    """

    def __init__(self,depth = 2):
        """Initialize the full binary tree and its branch-to-leaf paths.
        
        Parameters
        ----------
        depth : int, default=2
            Number of branch levels in the complete tree.
        
        Notes
        -----
        This structural initialization is shared with the ordinary DSRT;
        interaction matrices are supplied later through ``set_dataset()``.
        """

        self.depth = depth
        self.number_br_nodes = 2**depth -1
        self.number_leaf_nodes = 2**depth
        self.leaf_node  = {self.number_br_nodes+i:i for i in range(self.number_leaf_nodes)}
        self.BF_in_NL_R = {self.number_br_nodes+i:sorted(alt_anchesters(self.number_br_nodes+i,left=False)) for i in range(self.number_leaf_nodes)}
        self.BF_in_NL_L = {self.number_br_nodes+i:sorted(alt_anchesters(self.number_br_nodes+i,right=False)) for i in range(self.number_leaf_nodes)}
        self.zero_division = False
    
    def set_dataset(self,dissimilarity,dataset,type_reg = 'simple', matrix = np.array([]),reg_kkls=1e-3):
        """Store first-entity routing data and configure the leaf regression mode.
        
        Parameters
        ----------
        dissimilarity : numpy.ndarray of shape (n_first, n_first)
            Matrix used in the routing functions for the first entity type;
            may encode dissimilarities or distances according to the experiment.
        dataset : numpy.ndarray
            Responses: a one-dimensional vector for ``'simple'``; a feature
            matrix with the response in its last column for ``'linear_reg'``;
            or an interaction matrix of shape (n_first, n_second) for ``'kkls'``.
        type_reg : {'simple', 'linear_reg', 'kkls'}, default='simple'
            Constant, ordinary linear, or Kronecker least-squares leaf model.
        matrix : numpy.ndarray, optional
            Second-entity matrix of shape (n_second, n_second), required
            for ``type_reg='kkls'``. Stored in ``self.P``.
        reg_kkls : float, default=1e-3
            Regularization parameter passed to KronRLS in leaf updates.
        
        Returns
        -------
        bool
            True after recording the dataset and initializing node arrays.
        
        Notes
        -----
        KRLS additionally allocates per-leaf entity indices and lengths
        (``index_node_leaves``, ``len_node_leaves``) and stores the matrix
        response without flattening it. No leaf model is fitted here.
        """

        
        self.type_reg = type_reg
        self.K = dissimilarity
        if len(dataset.shape) > 1 and (self.type_reg=='linear_reg'):
            self.X_train = dataset[:,:-1]
            self.y_train = dataset[:,-1]
            self.number_features = self.X_train.shape[1]
            self.number_samples = np.int64(self.y_train.shape[0])
        elif self.type_reg=='kkls':
            self.y_train = dataset
            self.P = matrix
            self.number_samples = np.int64(self.K.shape[0])
            self.number_second_matrix = np.int64(self.P.shape[0])
            self.len_node_leaves = np.ones(self.number_leaf_nodes ,dtype=np.int64)*int(self.number_samples)
            self.index_node_leaves = np.multiply(np.ones((self.number_leaf_nodes,self.number_samples)),range(self.number_samples)).astype(np.int64)
        else:
            self.y_train = dataset
            self.number_samples = np.int64(self.y_train.shape[0])    
        
        self.reg_kkls = reg_kkls
        self.alphas_node = np.zeros((self.number_br_nodes,self.number_samples))
        self.alphas_y_node = np.zeros((self.number_br_nodes,self.number_samples))
        self.index_node = np.multiply(np.ones((self.number_br_nodes,self.number_samples)),range(self.number_samples)).astype(np.int64)
        self.len_node = np.ones(self.number_br_nodes,dtype=np.int64)*int(self.number_samples)
        
        
        return True
    
    def set_nystrom(self,typ=None,percentage=0.2,ridge_leverage_lambda = 1):
        
        """Configure optional prototype selection for the branch-node subproblems.
        
        Parameters
        ----------
        typ : {None, 'uniform', 'kmediod', 'DAC', 'recursive'}, default=None
            Selection strategy; None disables prototype selection.
        percentage : float, default=0.2
            Fraction of first-entity training observations requested as prototypes.
        ridge_leverage_lambda : float, default=1
            Parameter passed to the prototype selection utility.
        
        Returns
        -------
        bool
            True after saving the configuration.
        
        Notes
        -----
        The KRLS experiments use ``typ=None``. Other branches are retained from
        the base implementation, but prototype selection has not been verified
        for this KRLS extension. This signature does not expose ``k_med_type``.
        """
        self.type_nystrom = typ
        self.perc = percentage
        self.reduced_number = int(percentage*self.number_samples)
        if typ != None:
            self.nystrom = Nystrom(n_components = self.reduced_number,sampling = self.type_nystrom,ridge_leverage_lambda = ridge_leverage_lambda,random_state = self.seed)
        else:
            self.nystrom = None
            
        return True
    
    def set_param_KLR(self,C=1000.0,nu=1e-6,value = 70.,seed=23):
        
        """Set parameters for weighted kernel logistic routing subproblems.
        
        Parameters
        ----------
        C : float, default=1000.0
            Coefficient-bound parameter used by the logistic subproblems.
        nu : float, default=1e-6
            Numerical parameter passed to the Cython SMO routine.
        value : float, default=70.0
            Steepness parameter of the branch-node logistic functions.
        seed : int, default=23
            Random seed retained for prototype selection.
        
        Returns
        -------
        bool
            True after storing the parameter values.
        """

        self.seed = seed
        self.C = C
        self.nu = nu 
        self.value = value
        return True
    
    def starting_point(self,seed = 5):
        """Randomly initialize branch parameters and the selected leaf model.
        
        Parameters
        ----------
        seed : int, default=5
            Seed assigned to NumPy's global random generator.
        
        Returns
        -------
        bool
            True after initialization.
        
        Notes
        -----
        For KRLS, the leaf coefficients ``a_L`` are initialized as an array of
        shape (n_leaves, n_first, n_second), rather than feature-weight vectors.
        The branch-node coefficients and intercepts are initialized using the
        same routing structure as the ordinary DSRT; this version calls the
        historical ``b_calc`` utility to form those intercepts.
        """
        
        
        np.random.seed(seed)
        
        if self.type_reg=='linear_reg':
            self.a_L = np.random.uniform(low=-600.0, high=600.0, size=(self.number_leaf_nodes,self.number_features))
            
        if self.type_reg == 'kkls':
            self.a_L = np.random.uniform(low=-600.0, high=600.0, size=(self.number_leaf_nodes,self.number_samples,self.number_second_matrix))
            
        self.mu_L = np.random.uniform(low=-600.0, high=600.0, size=(self.number_leaf_nodes,1))
        self.b_node = np.empty(self.number_br_nodes)
        for i in range(self.number_br_nodes):
            self.alphas_node[i] = np.random.uniform(low=0.0, high=self.C, size=self.number_samples)
            label = np.random.choice([-1,1], size=self.number_samples)
            self.alphas_y_node[i] = np.multiply(self.alphas_node[i],label)
            self.b_node[i] = b_calc(self.alphas_node[i],label.astype(np.float64),self.alphas_y_node[i],self.K,self.C,self.number_samples)
        return True
    
    
    
    def Phi_it(self,x,a_leaf,mu_leaf):
        """Evaluate the prediction of one leaf for one observation.
        
        Parameters
        ----------
        x : numpy.ndarray
            Feature vector; ignored for constant-prediction leaves.
        a_leaf : numpy.ndarray
            Linear coefficients of the leaf; ignored for constant leaves.
        mu_leaf : float
            Leaf intercept or constant prediction.
        
        Returns
        -------
        float or str
            Leaf prediction. If a linear model receives mismatched feature
            and coefficient lengths, the method returns the string 'error'.
        """
        if self.type_reg== 'simple':
            
            return mu_leaf
        else:
            if len(x)==len(a_leaf):
                return a_leaf.dot(x)+mu_leaf
            else:
                return "error"
    
    def Phi_it_kkls(self,k,a_leaf,p):
        """Evaluate the KRLS prediction for one first-entity/second-entity pair.
        
        Parameters
        ----------
        k : numpy.ndarray of shape (n_selected_first,)
            Similarities or distances to the first-entity indices of this leaf.
        a_leaf : numpy.ndarray of shape (n_selected_first, n_second)
            Coefficient matrix associated with the leaf.
        p : numpy.ndarray
            Similarity vector or matrix for the second entity type.
        
        Returns
        -------
        numpy.ndarray or float
            Matrix product ``k @ a_leaf @ p`` with the shape determined by p.
        
        Notes
        -----
        Unlike ``Phi_it()``, this predictor uses interactions between two
        entity types and does not add the ordinary OLS leaf intercept.
        """
        
        return k.dot(a_leaf.dot(p))
        
    
    def solving_root(self,max_iter = 20,parallel= True,solver='TNC'):
        """Optimize the root branch-node routing parameters (BN Step).
        
        For KRLS leaves, the loss matrix is formed by summing squared
        interaction prediction errors across the second entity dimension,
        using each leaf's current KRLS coefficient matrix and selected indices.
        The resulting branch objective uses the same Numba solvers as the
        ordinary DSRT.
        
        Parameters
        ----------
        max_iter : int, default=20
            Requested optimizer iteration limit; not explicitly set for TNC.
        parallel : bool, default=True
            Select the parallel rather than serial Numba objective and gradient.
        solver : str, default='TNC'
            SciPy optimization method.
        
        Returns
        -------
        bool
            True after the root update, or False for an unknown Adam solver (not supported in this version).
        
        Notes
        -----
        Updates the root routing coefficients and intercept in place. Uses an
        L2 penalty scaled by the number of first-entity training observations.
        """

        
        l2_penalty=1e-3/self.number_samples
        
        
        if self.type_reg == 'linear_reg':
            y = self.y_train.copy().reshape(1,self.number_samples)
            sq_resp = np.square((self.a_L.dot(self.X_train.T) + self.mu_L) - y)
        elif self.type_reg== 'simple':
            y = self.y_train.copy().reshape(1,self.number_samples)
            sq_resp = np.square(self.mu_L - y)
        elif self.type_reg== 'kkls':
            y = self.y_train.copy()
            sq_resp = np.zeros((2**self.depth,self.number_samples))

            
            for tt in range(2**self.depth):
                
                idxs = self.index_node_leaves[tt][:self.len_node_leaves[tt]]
                
                KK = self.K[:,idxs]
                
                A = self.a_L[tt][idxs ,:]
                prediction = KK @ A @ self.P.T

                residual = (y - prediction)**2
                
                sq_resp[tt,:] = residual.sum(axis=1)
        if self.type_nystrom == None:
            start_point = np.zeros((self.number_samples)+1)
            start_point[:self.number_samples] = self.alphas_y_node[0,:]
            start_point[-1] = self.b_node[0]
            list_tree = np.array([self.depth,self.number_samples,self.number_samples])
        else:
            
            list_tree = np.array([self.depth,self.number_samples,self.len_node[0]])
            
            start_point = np.zeros(self.len_node[0]+1)
            start_point[:self.len_node[0]] = self.alphas_y_node[0,self.temp_list_root]
            start_point[-1] = self.b_node[0]
        
        

        leaf_node, estr, BF_in_NL, BF_in_NL_L, BF_in_NL_R = precompute_tree_structures_final(self.depth)
        argument = (self.b_node[1:],sq_resp,self.K,self.alphas_y_node,self.index_node,self.len_node,list_tree,leaf_node,estr,BF_in_NL,BF_in_NL_L,BF_in_NL_R,self.value,l2_penalty)
        beg = time.time()
        if parallel:
            
            if solver == 'L-BFGS-B':
                self.sol = optimize.minimize(simple_root_parallel_val, x0 = start_point,jac = grad_simple_root_parallel_val ,args = argument, method = 'L-BFGS-B',options={'maxiter': max_iter,'disp': False,'maxcor': 50})
            elif solver == 'TNC':
                self.sol = optimize.minimize(simple_root_parallel_val, x0 = start_point,jac = grad_simple_root_parallel_val ,args = argument, method = solver,options={'disp': False})#'maxCGit': 300,
            else:
                self.sol = optimize.minimize(simple_root_parallel_val, x0 = start_point,jac = grad_simple_root_parallel_val ,args = argument, method = solver,options={'maxiter': max_iter,'disp': False})
        else: #'TNC'
            if solver == 'L-BFGS-B':
                self.sol = optimize.minimize(simple_root_val, x0 = start_point,jac = grad_simple_root_val ,args = argument, method = 'L-BFGS-B' ,options={'maxiter': max_iter,'disp': False,'maxcor': 50})
            elif solver == 'TNC':
                self.sol = optimize.minimize(simple_root_val, x0 = start_point,jac = grad_simple_root_val ,args = argument,  method = solver ,options={'disp': False}) #'maxCGit': 300,
            else:    
                self.sol = optimize.minimize(simple_root_val, x0 = start_point,jac = grad_simple_root_val ,args = argument,  method = solver ,options={'maxiter': max_iter,'disp': False}) 
        end = time.time()
        self.solution = self.sol.x
        
        
        print('Time',end-beg)

        self.alphas_y_node[0,self.index_node[0][:self.len_node[0]]] = self.solution[:-1]
        self.alphas_node[0,self.index_node[0][:self.len_node[0]]] = np.abs(self.solution[:-1])
        
            
        self.b_node[0] = self.solution[-1]

        return True


    def solving_phase_branching(self,node,max_iter = 5,tol = 1e-6,parallel=True,solver='TNC'):
        """Optimize the branch rules in the subtree rooted at a branch node.
        
        For ``'kkls'``, each descendant leaf contributes the sum of its squared
        matrix-prediction residuals across the second entity dimension. The
        branch-node optimizer then operates on the resulting per-entity losses,
        without changing the shared BN objective and gradient implementations.
        
        Parameters
        ----------
        node : int
            Root branch-node index of the subtree being optimized.
        max_iter : int, default=5
            Iteration limit for the supported optimizer branches.
        tol : float, default=1e-6
            Accepted but not used by this implementation.
        parallel : bool, default=True
            Whether to select the parallel Numba objective and gradient.
        solver : str, default='TNC'
            Branch-node optimization method.
        
        Returns
        -------
        bool
            True following the update, or False for an unknown Adam solver.
        
        Notes
        -----
        Requires the temporary working-set matrices prepared by the training
        procedure, including ``temp_indices_per_kernel`` in KRLS mode.
        """
        
        l2_penalty=1e-3/self.number_samples
        
        steps = self.depth - math.floor(math.log(node+1,2))
        list_involved_leaf = []
        init = node
        for _ in range(steps):
        
            init = alt_descendant(init,right=False)[0]
        
        for t in range(int(2**steps)):
            list_involved_leaf.append(init+t)
        
        eq_leaf = [self.leaf_node[i] for i in list_involved_leaf]
        len_samples = len(self.temp_y)
        
        
        
        if self.type_reg == 'linear_reg':
            y = self.temp_y.copy().reshape(1,len_samples)
            sq_resp = np.square((self.a_L[eq_leaf].dot(self.temp_x.T) + self.mu_L[eq_leaf]) - y)
        elif self.type_reg== 'simple':
            y = self.temp_y.copy().reshape(1,len_samples)
            sq_resp = np.square(self.mu_L[eq_leaf] - y)
        elif self.type_reg== 'kkls':
            y = self.temp_y.copy()

            sq_resp = np.zeros((2**steps,len_samples))
            
            for tt in range(2**steps):
                
                idxs = self.index_node_leaves[eq_leaf[tt]][:self.len_node_leaves[eq_leaf[tt]]]
                
                KK = self.K[np.ix_(self.temp_indices_per_kernel,idxs)]
                
                A = self.a_L[eq_leaf[tt]][idxs ,:]
                prediction = KK @ A @ self.P.T


                residual = (y - prediction)**2
                
                sq_resp[tt,:] = residual.sum(axis=1)
            
        temp_branch = []
        subtree(node,steps-1,temp_branch)
        list_involved_branch = flat_list([[node],temp_branch])

        len_involved_branch = len(list_involved_branch)
        len_node = self.len_node[node]
        index_involved = self.index_node[node][:len_node]
        
        
        
       
        if self.type_nystrom == None:
            total_len = len_involved_branch*len_node
            list_tree = np.array([steps,total_len,len_node])
            start_point = np.zeros(total_len+len_involved_branch)
        else:
            len_samples = self.temp_K.shape[0]
            total_len = len_involved_branch*len_samples
            list_tree = np.array([steps,total_len,len_samples])
            start_point = np.zeros((len_involved_branch*len_node)+len_involved_branch)
        
        for j,i in enumerate(list_involved_branch):
            
            self.len_node[i] = len_node
            self.index_node[i][:len_node] = index_involved
            temp_al_y = self.alphas_y_node[i,index_involved].copy()
            start_point[j*len_node:(j+1)*len_node] = temp_al_y

        
        for jj,i in enumerate(list_involved_branch):
            start_point[-len_involved_branch+jj] = self.b_node[i]
      

        leaf_node, estr, BF_in_NL, BF_in_NL_L, BF_in_NL_R = precompute_tree_structures_final(steps)

        argument = (sq_resp,self.temp_K,len_node,list_tree, leaf_node, estr,BF_in_NL, BF_in_NL_L, BF_in_NL_R,self.value,self.number_samples,l2_penalty)
        beg = time.time()
        if parallel:
            if solver == 'L-BFGS-B':
                self.sol = optimize.minimize(simple_dec_parallel_val, x0 = start_point, jac =grad_dec_nuovo_parallel_val, args = argument,  method = 'L-BFGS-B',options={'maxiter': max_iter,'disp': False,'maxcor': 50})
            elif solver == 'TNC':
                self.sol = optimize.minimize(simple_dec_parallel_val, x0 = start_point, jac =grad_dec_nuovo_parallel_val, args = argument,  method = solver,options={'disp': False}) #
            else:
                self.sol = optimize.minimize(simple_dec_parallel_val, x0 = start_point, jac =grad_dec_nuovo_parallel_val, args = argument,  method = solver,options={'maxiter': max_iter,'disp': False})
            
        else:
            if solver == 'L-BFGS-B':
                self.sol = optimize.minimize(simple_dec_val, x0 = start_point, jac = grad_dec_nuovo_val, args = argument,  method = 'L-BFGS-B',options={'maxiter': max_iter,'disp': False,'maxcor': 50})                
            elif solver == 'TNC':
                self.sol = optimize.minimize(simple_dec_val, x0 = start_point, jac = grad_dec_nuovo_val, args = argument,  method = solver,options={'disp': False}) #'maxCGit': 300,        
            else:
                self.sol = optimize.minimize(simple_dec_val, x0 = start_point, jac = grad_dec_nuovo_val, args = argument,  method = solver,options={'maxiter': max_iter,'disp': False})             
        end = time.time()
        self.solution = self.sol.x

        print('TIME',end-beg)
        
        alp_plus = self.solution[:total_len]
        

        for j,i in enumerate(list_involved_branch):

            self.alphas_node[i,index_involved] = np.abs(alp_plus[j*len_node:(j+1)*len_node])
            self.alphas_y_node[i,index_involved] = alp_plus[j*len_node:(j+1)*len_node] 
            self.b_node[i] = self.solution[-len_involved_branch+j]

        return True
        
    def generic_instances_in_node(self,nodo,K_temp,threshold=0.5):
        """Find observations routed to a given node by hard decisions.
        
        Parameters
        ----------
        nodo : int
            Index of a branch or leaf node in the binary tree.
        K_temp : numpy.ndarray of shape (n_observations, n_train)
            Distances from observations to the training data.
        threshold : float, default=0.5
            Routing threshold: probabilities greater than this value go
            left; probabilities at or below it go right.
        
        Returns
        -------
        list of int
            Row indices of the observations reaching the node.
        """
        indices_x = list(range(K_temp.shape[0]))
        
        steps = math.floor(math.log(nodo+1,2))
        
        left = alt_anchesters(nodo,right = False)
        if left:
            left = left[::-1]
        right = alt_anchesters(nodo,left = False)
        if right:
            right = right[::-1]
        
        for j in range(steps):
            temp_indices = []
            temp_node, direction = erase_first(left,right)
            
            b_temp = self.b_node[temp_node]
            ay_temp = self.alphas_y_node[temp_node,self.index_node[temp_node][:self.len_node[temp_node]]]
            
            
            
            for i in indices_x:
                pr_i = Pr_it(K_temp[i,self.index_node[temp_node][:self.len_node[temp_node]]],ay_temp,b_temp,self.value,self.number_samples)
                if direction == -1:
                    if pr_i>threshold:
                        temp_indices.append(i)                    
                else:
                    if pr_i<=threshold:
                        temp_indices.append(i)
            indices_x = temp_indices.copy()
    
        return indices_x
    
    
    def predict(self,similarity,x): 
        """Predict responses using deterministic routing to a leaf.
        
        Each observation is assigned to one leaf using a routing threshold
        of 0.5, then evaluated with that leaf's prediction model.
        
        Parameters
        ----------
        dissimilarity : numpy.ndarray of shape (n_observations, n_train)
            Distances from each observation to the training observations.
        x : numpy.ndarray
            Feature rows for linear-regression leaves. May be an empty
            array when type_reg='simple'.
        
        Returns
        -------
        numpy.ndarray of shape (n_observations,)
            Predictions for the input observations.
        
        This method supports only constant and OLS leaves; use
        ``predict_kkls()`` for interaction-matrix predictions.
        """
        res = np.zeros(similarity.shape[0])

            
        for t in self.leaf_node:
            ind = self.generic_instances_in_node(t,similarity)
            
            if self.type_reg == 'simple':
                
                for i in ind:
                    res[i] = self.Phi_it(np.array([]),np.array([]),self.mu_L[self.leaf_node[t]][0])
                
            elif self.type_reg == 'linear_reg':
            
                for i in ind:
                    res[i] = self.Phi_it(x[i],self.a_L[self.leaf_node[t]],self.mu_L[self.leaf_node[t]][0])
                
                        
        return res
    
    def predict_kkls(self,similarity,p): 
        """Predict an interaction matrix by hard routing to KRLS leaf models.
        
        Parameters
        ----------
        similarity : numpy.ndarray of shape (n_test_first, n_train_first)
            First-entity test-to-training matrix used for hard routing and
            for the KRLS prediction within the reached leaf.
        p : numpy.ndarray of shape (n_second, n_output_second)
            Matrix of second-entity inputs used in the leaf KRLS predictor.
        
        Returns
        -------
        numpy.ndarray of shape (n_test_first, n_output_second)
            Predicted interactions for each supplied first entity.
        
        Notes
        -----
        For each test row, only the KRLS coefficient matrix and first-entity
        indices associated with its deterministically assigned leaf are used.
        """

        res = np.zeros((similarity.shape[0],p.shape[1]))

        for t in self.leaf_node:
            ind = self.generic_instances_in_node(t,similarity)
            
            idxs = self.index_node_leaves[self.leaf_node[t]][:self.len_node_leaves[self.leaf_node[t]]]
            
            for i in ind:

                res[i] = self.Phi_it_kkls(similarity[i,idxs],self.a_L[self.leaf_node[t]][idxs,:],p)
                
        
        return res
    
    
    def predict_contribution_single_data_kkls(self,similarity,p): 
        """Extract per-pair coefficient contributions for one routed first entity.
        
        Parameters
        ----------
        similarity : numpy.ndarray of shape (1, n_train_first)
            First-entity test-to-training row used for routing and weighting.
        p : numpy.ndarray of shape (1, n_second)
            Second-entity similarity row used to weight coefficient entries.
        
        Returns
        -------
        numpy.ndarray of shape (n_train_first, n_second)
            For the assigned leaf, entries equal the leaf coefficient times the
            corresponding values from similarity and p; other entries stay zero.
        
        Notes
        -----
        This is an interpretation helper for individual interactions, rather
        than the full matrix predictor used in the setting B/C experiments.
        """
        res = np.zeros((similarity.shape[1],p.shape[1]))

            
        for t in self.leaf_node:
            ind = self.generic_instances_in_node(t,similarity)
            
            
            if ind:
                
                idxs = self.index_node_leaves[self.leaf_node[t]][:self.len_node_leaves[self.leaf_node[t]]]
                
                for jj in idxs:
                    
                    for kk in range(p.shape[1]):
                        res[jj,kk] = self.a_L[self.leaf_node[t]][jj,kk]*similarity[0,jj]*p[0,kk]                
        
        return res
    
    
    
    def predict_label_level(self,K_temp,level = 0):# la root è a livello 0
        """Return the node assignment of each observation at a given depth.
        
        Parameters
        ----------
        K_temp : numpy.ndarray of shape (n_observations, n_train)
            Distances from each observation to the training data.
        level : int, default=0
            Tree level, with the root at 0 and leaves at self.depth.
        
        Returns
        -------
        numpy.ndarray of shape (n_observations,)
            Local node indices at the requested level, stored as floats.
            For an invalid level, prints a warning and returns zeros.
        """

        res = np.zeros(K_temp.shape[0])
        
        if level == self.depth:
            for t in self.leaf_node:
                ind = self.generic_instances_in_node(t,K_temp)
                res[ind] = self.leaf_node[t]
        elif level < self.depth and level >= 0:
            for j,t in enumerate(range(2**(level),2**(level+1))):
                ind = self.generic_instances_in_node(t-1,K_temp)
                res[ind] = j
        else:
            print('Wrong level!')
                          
        return res
    
    @staticmethod
    @jit(float64[:,:](int64,int64,float64[:,:],float64[:],float64[:,:],float64[:,:],float64[:],int64[:],int64[:],float64,int64), nopython=True, fastmath= True)
    def create_AA_numba(node,steps,alp_y,b_node,K,temp_x,temp_y,list_tree,index_involved,value,l_samples):
        """Build a probability-weighted least-squares system for leaf regressions.
        
        For each leaf of the selected subtree, the design rows and responses
        are multiplied by the square root of that leaf's routing probability.
        
        Parameters
        ----------
        node : int
            Root node of the selected subtree.
        steps : int
            Number of branch levels between node and its descendant leaves.
        alp_y : numpy.ndarray
            Signed routing coefficients for all branch nodes.
        b_node : numpy.ndarray
            Branch-node intercepts.
        K : numpy.ndarray
            Working-set distances restricted to the selected prototype columns.
        temp_x : numpy.ndarray
            Feature rows of the current working set.
        temp_y : numpy.ndarray
            Responses of the current working set.
        list_tree : numpy.ndarray
            Array containing tree depth and number of input features.
        index_involved : numpy.ndarray
            Training indices used for routing coefficients in this subproblem.
        value : float
            Logistic steepness parameter.
        l_samples : int
            Normalization factor in the routing probabilities.
        
        Returns
        -------
        numpy.ndarray
            Augmented weighted least-squares matrix, with the response in
            the last column.
        """

        depth = list_tree[0]
        number_features = list_tree[1]
        temp_dic = {}
        init = node
        for _ in range(steps):
    
            init = alt_descendant_numba(init,left=1,right=0)[0]
    
        for t in range(int(2**steps)):
            temp_dic[t] = init
            init+=1
        N = len(temp_x)
        indices_x = range(N)
        len_indx = len(indices_x)
        
        lll = len(index_involved)
        Ab = np.zeros(((2**steps)*N,1+(2**steps)*(number_features+1)))
        
        for i in temp_dic:
            
            for j in indices_x:
    
                left = List(alt_anchesters_numba(temp_dic[i],left=1,right = 0))
                if left:
                    temp_left = [nl for nl in left if nl>=node]
                    left = List(temp_left)
                
                right = List(alt_anchesters_numba(temp_dic[i],left = 0,right=1))
                if right:
                    temp_right = [nl for nl in right if nl>=node]
                    right = List(temp_right)
    
                new_x = np.append(temp_x[j],1)
                
                Pr = 1
                
                for step in range(steps):
                    
                    temp_node, direction = erase_first_numba(left,right)
                    if direction == -1:
                        Pr*= Pr_it(K[j],alp_y[temp_node][index_involved],b_node[temp_node],value,l_samples)
                    else:
                        Pr*= (1-Pr_it(K[j],alp_y[temp_node][index_involved],b_node[temp_node],value,l_samples))
                
                
                Ab[i*N + j,-1] = np.sqrt(Pr)*temp_y[j]
                Ab[i*N + j,i*(number_features+1):i*(number_features+1)+(number_features+1)] = np.sqrt(Pr)*new_x
                              
                                            
        return Ab
    
    @staticmethod
    @jit(float64[:,:](int64,int64,float64[:,:],float64[:],float64[:,:],int64[:],int64[:],float64,int64), nopython=True, fastmath= True)
    def create_P_numba(node,steps,alp_y,b_node,K,list_tree,index_involved,value,l_samples):
        """Build square-root routing-probability weights for KRLS leaf fitting.
        
        The returned weights are used to premultiply the first-entity matrix
        and the response matrix before solving a KronRLS problem for each leaf.
        
        Parameters
        ----------
        node : int
            Root of the selected subtree.
        steps : int
            Number of branch levels below the subtree root.
        alp_y : numpy.ndarray
            Signed branch routing coefficients.
        b_node : numpy.ndarray
            Branch-node intercepts.
        K : numpy.ndarray
            Working-set rows of the first-entity routing matrix.
        list_tree : numpy.ndarray
            Tree metadata; its first element is the full tree depth.
        index_involved : numpy.ndarray
            Indices used by routing rules in the current subproblem.
        value : float
            Logistic steepness parameter.
        l_samples : int
            Normalization factor in the routing functions.
        
        Returns
        -------
        numpy.ndarray of shape (2**steps, n_working_first)
            Square roots of leaf-routing probabilities for the working set.
        """
        depth = list_tree[0]

        temp_dic = {}
        init = node
        for _ in range(steps):
    
            init = alt_descendant_numba(init,left=1,right=0)[0]
    
        for t in range(int(2**steps)):
            temp_dic[t] = init
            init+=1
        N = K.shape[0]
        indices_x = range(N)

        
        lll = len(index_involved)
        PP = np.zeros((2**steps,N))
        
        
        for i in temp_dic:
            
            for j in indices_x:
    
                left = List(alt_anchesters_numba(temp_dic[i],left=1,right = 0))
                if left:
                    temp_left = [nl for nl in left if nl>=node]
                    left = List(temp_left)
                
                right = List(alt_anchesters_numba(temp_dic[i],left = 0,right=1))
                if right:
                    temp_right = [nl for nl in right if nl>=node]
                    right = List(temp_right)
    
                
                
                Pr = 1
                
                for step in range(steps):
                    
                    temp_node, direction = erase_first_numba(left,right)
                    if direction == -1:
                        Pr*= Pr_it(K[j],alp_y[temp_node][index_involved],b_node[temp_node],value,l_samples)
                    else:
                        Pr*= (1-Pr_it(K[j],alp_y[temp_node][index_involved],b_node[temp_node],value,l_samples))
                
                PP[i,j] = np.sqrt(Pr)
                
                              
                                            
        return PP
    
    
    @staticmethod
    @jit(float64[:](int64,int64,float64[:,:],float64[:],float64[:,:],float64[:],int64[:],int64[:],float64,int64), nopython=True, fastmath= True)
    def A_numba_solved(node,steps,alp_y,b_node,K,temp_y,list_tree,index_involved,value,l_samples):
        """Compute constant leaf predictions as weighted response means.
        
        The weights are routing probabilities within the selected subtree.
        If both accumulated weight and weighted response are zero, the
        corresponding leaf prediction is set to zero.
        
        Parameters
        ----------
        node : int
            Root node of the selected subtree.
        steps : int
            Number of branch levels in the selected subtree.
        alp_y : numpy.ndarray
            Signed routing coefficients of all branch nodes.
        b_node : numpy.ndarray
            Branch-node intercepts.
        K : numpy.ndarray
            Working-set distances restricted to selected prototype columns.
        temp_y : numpy.ndarray
            Responses of the current working set.
        list_tree : numpy.ndarray
            Tree metadata array; its first element is the full tree depth.
        index_involved : numpy.ndarray
            Training indices involved in the routing rules.
        value : float
            Logistic steepness parameter.
        l_samples : int
            Normalization factor used in the routing probabilities.
        
        Returns
        -------
        numpy.ndarray of shape (2**steps,)
            Constant prediction for each descendant leaf.
        """
        depth = list_tree[0]

        temp_dic = {}
        init = node
        for _ in range(steps):
    
            init = alt_descendant_numba(init,left=1,right=0)[0]
    
        for t in range(int(2**steps)):
            temp_dic[t] = init
            init+=1
        N = len(temp_y)
        indices_x = range(N)
        len_indx = len(indices_x)
        
        lll= len(index_involved)
        
        Ab = np.zeros(((2**steps)*N,1+(2**steps)))
        res = np.zeros(2**steps)
        for i in temp_dic:
            
            num = 0
            den = 0
            for j in indices_x:
    
                left = List(alt_anchesters_numba(temp_dic[i],left=1,right = 0))
                if left:
                    temp_left = [nl for nl in left if nl>=node]
                    left = List(temp_left)
                
                right = List(alt_anchesters_numba(temp_dic[i],left = 0,right=1))
                if right:
                    temp_right = [nl for nl in right if nl>=node]
                    right = List(temp_right)
    
                new_x = 1 
                
                Pr = 1
                
                for step in range(steps):
                    
                    temp_node, direction = erase_first_numba(left,right)
                    if direction == -1:
                        Pr*= Pr_it(K[j],alp_y[temp_node][index_involved],b_node[temp_node],value,l_samples)
                    else:
                        Pr*= (1-Pr_it(K[j],alp_y[temp_node][index_involved],b_node[temp_node],value,l_samples))
                num+= Pr*temp_y[j]
                den+= Pr
            if num==0 and den == 0:
                res[i] = 0
            else:
                res[i] = num/den
                                            
        return res
    
    @staticmethod
    @jit(float64[:,:](int64,int64,float64[:,:],float64[:],int64[:]), nopython=True,fastmath=True)
    def create_AA_numba_clust(node,steps,temp_x,temp_y,list_tree):
        """Build an unweighted leaf least-squares system for initialization.
        
        Parameters
        ----------
        node : int
            Leaf node being initialized.
        steps : int
            Number of levels below node; zero in the clustering initialization.
        temp_x : numpy.ndarray
            Features of the observations assigned to the leaf.
        temp_y : numpy.ndarray
            Corresponding responses.
        list_tree : numpy.ndarray
            Tree depth and number of features.
        
        Returns
        -------
        numpy.ndarray
            Augmented least-squares matrix with the response as the last
            column, without routing-probability weights.
        """
        depth = list_tree[0]
        number_features = list_tree[1]

        temp_dic = {}
        init = node
        for _ in range(steps):
    
            init = alt_descendant_numba(init,left=1,right=0)[0]
    
        for t in range(int(2**steps)):
            temp_dic[t] = init
            init+=1
        N = len(temp_x)
        indices_x = range(N)
        
        
        Ab = np.zeros(((2**steps)*N,1+(2**steps)*(number_features+1)))
        
        for i in temp_dic:
            
            for j in indices_x:
    
                new_x = np.append(temp_x[j],1)
                
                Ab[i*N + j,-1] = temp_y[j]
                Ab[i*N + j,i*(number_features+1):i*(number_features+1)+(number_features+1)] = new_x
                             
        return Ab
    
    
    def solving_phase_leaf(self,node = 0,list_idx=[]):
        """Update descendant leaf regressions in the selected subtree (LN Step).
        
        Parameters
        ----------
        node : int, default=0
            Root of the selected subtree (which may be an individual leaf).
        list_idx : sequence of int, optional
            Indices of the first-entity observations assigned to the working set.
        
        Returns
        -------
        bool
            True after updating the descendant leaf parameters.
        
        Notes
        -----
        Constant leaves use weighted means and OLS leaves use a sparse
        least-squares system, as in the ordinary DSRT. KRLS leaves instead
        compute square-root routing weights with ``create_P_numba()`` and fit
        ``KronRLS(X1=..., X2=self.P, Y=...)`` separately for each leaf, with
        regularization ``self.reg_kkls``. The resulting coefficient matrices
        are assigned to each leaf's selected first-entity indices. In KRLS
        mode the method also updates ``index_node_leaves`` and
        ``len_node_leaves`` to track the interactions contributing to a leaf.
        An existing IndexError-specific fallback adds a small diagonal jitter
        to X1; other exceptions are propagated.
        """
        steps = self.depth - math.floor(math.log(node+1,2))
        list_involved_leaf = []
        init = node
        for _ in range(steps):
            init = alt_descendant(init,right=False)[0]
            
        len_idx_leaf = len(list_idx)
        for t in range(int(2**steps)):
            leaf_idx = init+t
            list_involved_leaf.append(leaf_idx)
            
            self.len_node_leaves[self.leaf_node[leaf_idx]] = len_idx_leaf
            self.index_node_leaves[self.leaf_node[leaf_idx]][:len_idx_leaf] = list_idx
            
        if steps > 0:
            len_node = self.len_node[node]
            index_involved = self.index_node[node][:len_node] 
        else:
            len_node = len(list_idx)
            index_involved = np.array(list_idx,dtype=np.int64)
        steps = int(self.depth - math.floor(math.log(node+1,2)))
        
        if self.type_reg == 'simple':
            
            x = self.A_numba_solved(node,steps,self.alphas_y_node,self.b_node,self.K[np.ix_(list_idx,index_involved)],self.temp_y,np.array([self.depth]),index_involved,self.value,self.number_samples)
            for t in range(2**steps):
                self.mu_L[self.leaf_node[list_involved_leaf[t]]][0] = x[t]
        elif self.type_reg == 'linear_reg':
        
            Ab = self.create_AA_numba(node,steps,self.alphas_y_node,self.b_node,self.K[np.ix_(list_idx,index_involved)],self.temp_x,self.temp_y,np.array([self.depth,self.number_features]),index_involved,self.value,self.number_samples)
    
            A_matrix = csc_matrix(Ab[:,:-1], dtype=float)
            res, istop, itn, normr = lsqr(A_matrix,Ab[:,-1])[:4]
            x = res
            
            for t in range(2**steps):
                for j in range(self.number_features):
                    self.a_L[self.leaf_node[list_involved_leaf[t]],j] = x[((self.number_features+1)*t)+j]
                self.mu_L[self.leaf_node[list_involved_leaf[t]]][0] = x[((self.number_features+1)*t)+self.number_features]
        
        elif self.type_reg == 'kkls':
            # Refit leaf-wise KRLS coefficients from routing-weighted data
            jitter = 1e-8
            # Use soft routing weights to fit separate KronRLS models at each leaf
            PP = self.create_P_numba(node,steps,self.alphas_y_node,self.b_node,self.K[np.ix_(list_idx,index_involved)],np.array([self.depth],dtype=np.int64),index_involved,self.value,self.number_samples)
            
            for t in range(2**steps):
                

                try:
                    temp_P = np.diag(PP[t,:])
                    learner = KronRLS(X1 = temp_P.dot(self.K[np.ix_(list_idx,list_idx)]), X2 = self.P, Y = temp_P.dot(self.temp_y)) 
                    learner.solve(self.reg_kkls) 
                except IndexError as e:
                    if "boolean index did not match" in str(e):
                        temp_P = np.diag(PP[t,:])
                        learner = KronRLS(X1 = temp_P.dot(self.K[np.ix_(list_idx,list_idx)])+jitter*np.eye(len(list_idx)), X2 = self.P, Y = temp_P.dot(self.temp_y)) # devo garantire che temp_y sia una matrice di dimensione list_idx x dimensione matrice P
                        learner.solve(self.reg_kkls) 
                    else:
                        raise
                
                # Retain coefficients only for the first-entity observations in this leaf
                self.a_L[self.leaf_node[list_involved_leaf[t]]][list_idx,:] = learner.W 
                
        return True
    
    def solving_phase_leaf_clust(self,node):
        """Initialize a constant or OLS leaf from its clustering-assigned data.
        
        Parameters
        ----------
        node : int
            Leaf index being initialized.
        
        Returns
        -------
        bool
            True after attempting the leaf initialization.
        
        Notes
        -----
        Only ``'simple'`` and ``'linear_reg'`` are implemented here. There is
        no KRLS-specific clustering leaf update, so the clustering option has
        not been validated for KRLS experiments.
        """
        if self.type_reg == 'linear_reg':
            Ab = self.create_AA_numba_clust(node,0,self.temp_x,self.temp_y,np.array([self.depth,self.number_features]))
            
            A_matrix = csc_matrix(Ab[:,:-1], dtype=float)
            res, istop, itn, normr = lsqr(A_matrix,Ab[:,-1])[:4]
            x = res


            for j in range(self.number_features):
                self.a_L[self.leaf_node[node],j] = x[j]
            self.mu_L[self.leaf_node[node]][0] = x[self.number_features]
        elif self.type_reg == 'simple':
            self.mu_L[self.leaf_node[node]][0] = np.mean(self.temp_y)

        
        return True
    
    def Pr_it_n_train(self,i,tl):
        """Compute the soft probability of reaching a training leaf.
        
        Parameters
        ----------
        i : int
            Index of a training observation.
        tl : int
            Encoded index of a leaf node in the full binary tree.
        
        Returns
        -------
        float
            Product of the left- or right-routing probabilities along
            the path from the root to the specified leaf.
        """

        return  reduce(operator.mul,(Pr_it(self.K[i,self.index_node[t][:self.len_node[t]]],self.alphas_y_node[t,self.index_node[t][:self.len_node[t]]] ,self.b_node[t],self.value,self.number_samples)  for t in self.BF_in_NL_L[tl]),1)* \
        reduce(operator.mul,(1-Pr_it(self.K[i,self.index_node[tr][:self.len_node[tr]]],self.alphas_y_node[tr,self.index_node[tr][:self.len_node[tr]]] ,self.b_node[tr],self.value,self.number_samples) for tr in self.BF_in_NL_R[tl]),1)

    
    def objective_train(self):
        """Evaluate the mean soft-routing training loss over first-entity rows.
        
        Returns
        -------
        float
            Mean of ``y_response_train(i)`` over the first-entity training set.
        
        Notes
        -----
        For KRLS leaves, each row contributes a sum of squared errors over the
        second entity dimension, weighted by soft probabilities of the leaves.
        This objective is used to retain and restore the best model state.
        """
        out = 0
        for i in range(self.number_samples):
            out += self.y_response_train(i)
        
        return out/self.number_samples
    
    def y_response_train(self,i):
        """Compute one first entity's probability-weighted prediction loss.
        
        Parameters
        ----------
        i : int
            Index of the first-entity training observation.
        
        Returns
        -------
        float or None
            Sum over leaves of their soft routing probabilities times the
            prediction loss. Returns None for an unsupported regression mode.
        
        Notes
        -----
        For ``'kkls'``, each leaf's loss is the sum of squared differences
        between its predicted interaction row and the observed response row.
        Constant and OLS modes use scalar squared residuals as in ``dsrt.py``.
        """
        
        response = 0
        
        if self.type_reg == 'simple':
            
            for t in self.leaf_node:
                response += self.Pr_it_n_train(i,t)*(self.Phi_it(np.array([]),np.array([]),self.mu_L[self.leaf_node[t]][0])-self.y_train[i])**2

        elif self.type_reg == 'linear_reg':
            
            for t in self.leaf_node:
                response += self.Pr_it_n_train(i,t)*(self.Phi_it(self.X_train[i],self.a_L[self.leaf_node[t]],self.mu_L[self.leaf_node[t]][0])-self.y_train[i])**2
        
        elif self.type_reg == 'kkls':
            
            for t in self.leaf_node:
                idxx = self.index_node_leaves[self.leaf_node[t]][:self.len_node_leaves[self.leaf_node[t]]]
                a_leaf = self.a_L[self.leaf_node[t]][idxx,:]
                k = self.K[i,idxx]
                res = (self.Phi_it_kkls(k,a_leaf,self.P) -self.y_train[i])**2
                
                response += self.Pr_it_n_train(i,t)*sum(res)
        else:
            
            return None
        
        return response
    
    def instances_in_node_simple(self,nodo,threshold=0.5):
        """Find training observations routed to a node by hard decisions.
        
        Parameters
        ----------
        nodo : int
            Index of the target branch or leaf node.
        threshold : float, default=0.5
            Probabilities above the threshold go left; probabilities at or
            below it go right.
        
        Returns
        -------
        list of int or range
            Training observation indices reaching the selected node.
            For the root, the initial range object is returned unchanged.
        """
        indices_x = range(self.number_samples)
        
        steps = math.floor(math.log(nodo+1,2))
        
        left = alt_anchesters(nodo,right = False)
        if left:
            left = left[::-1]
        right = alt_anchesters(nodo,left = False)
        if right:
            right = right[::-1]
        
        for j in range(steps):
            temp_indices = []
            temp_node, direction = erase_first(left,right)
            
            for i in indices_x:
                pr_i = Pr_it(self.K[i,self.index_node[temp_node][:self.len_node[temp_node]]],self.alphas_y_node[temp_node,self.index_node[temp_node][:self.len_node[temp_node]]] ,self.b_node[temp_node],self.value,self.number_samples)
                if direction == -1:
                    if pr_i>threshold:
                        temp_indices.append(i)                    
                else:
                    if pr_i<=threshold: 
                        temp_indices.append(i)
            indices_x = temp_indices.copy()
    
        return indices_x
    
    def solving_logistic(self,node_left,parent,node_right):
        """Recompute a branch routing rule from its current hard partition.
        
        Observations reaching parent are labeled according to whether they
        are routed to node_right, then a weighted KLR subproblem is solved
        using the Cython SMO implementation.
        
        Parameters
        ----------
        node_left : int
            Left child index; retained in the interface but not read.
        parent : int
            Index of the branch node whose rule is updated.
        node_right : int
            Right child used to construct the binary labels.
        
        Returns
        -------
        bool
            True after updating the parent's routing coefficients and
            intercept, or False if all parent's observations go right.
        
        Notes
        -----
        When prototype selection is enabled, the SMO variant with excluded
        indices is used. Updates self.alphas_y_node and self.b_node.
        The updated intercept in this extension is computed by ``b_calc``.
        """
        
        list_indeces = self.instances_in_node_simple(parent)
        lun = len(list_indeces)
        right_indeces = self.instances_in_node_simple(node_right)

        temp_y = np.zeros(lun)
        temp_K = self.K[np.ix_(list_indeces,list_indeces)]
        
        if len(list_indeces) == len(right_indeces):
            return False
        
        it = 0
        for i in list_indeces:
            if i in right_indeces:
                temp_y[it]=1
            else:
                temp_y[it]=-1
            it+=1
        
        if self.type_nystrom == None:
        
            alphas, _ =  SMO.SMO_lin_init_weighted_new(self.C/self.number_samples, 1e-5, 100,lun,temp_y,np.array([]),temp_K,self.nu,1e-5,1e-3,1e-8)
        else:
            lista_esclusi = []
            if parent == 0:
                for kk,iii in enumerate(list_indeces):
                    if iii not in self.temp_list_root:
                        lista_esclusi.append(kk)
            else: 
                for kk,iii in enumerate(list_indeces):
                    if iii not in self.temp_list_indices:
                        lista_esclusi.append(kk)
            alphas, _ =  SMO.SMO_lin_init_weighted_new_snello(self.C/self.number_samples, 1e-5, 100,lun,temp_y,np.array([]),temp_K,self.nu,1e-5,1e-3,1e-8,np.array(lista_esclusi,dtype=int))
        self.alphas_y_node[parent,list_indeces] = self.number_samples*np.multiply(alphas,temp_y)/self.value
        self.alphas_node[parent,list_indeces] = self.number_samples*alphas/self.value
        self.b_node[parent] = b_calc(alphas,temp_y,self.alphas_y_node[parent,list_indeces]/self.number_samples,temp_K,self.C,lun)/self.value


        return True
    
    def solving_patologic_logistic(self,node,left_indeces,parent,brother):
        """Recompute a branch rule after selecting observations to be reassigned.
        
        The binary labels are formed from left_indeces and the relative
        ordering of node and brother, then a weighted KLR problem is solved.
        
        Parameters
        ----------
        node : int
            Child node intended to receive reassigned observations.
        left_indeces : sequence of int
            Observation indices used to form the updated binary partition.
        parent : int
            Parent branch node whose parameters are updated.
        brother : int
            Sibling node providing observations for reassignment.
        
        Returns
        -------
        bool
            True after updating the parent's routing coefficients and
            intercept.
        
        Notes
        -----
        Uses the SMO variant with excluded indices when prototype selection
        is enabled. Updates self.alphas_y_node and self.b_node.
        This extension computes the updated intercept using ``b_calc``.
        """
        
        list_indeces = self.instances_in_node_simple(parent)
        
        lun = len(list_indeces)
        temp_y = np.zeros(lun)
        temp_K = self.K[np.ix_(list_indeces,list_indeces)]
        
        if node<brother:
            it = 0
            for i in list_indeces:
                if i in left_indeces:
                    temp_y[it]=1
                else:
                    temp_y[it]=-1
                it+=1
        else:    
            it = 0
            for i in list_indeces:
                if i in left_indeces:
                    temp_y[it]=-1
                else:
                    temp_y[it]=1
                it+=1    
        

        if self.type_nystrom == None:
        
            alphas, _ =  SMO.SMO_lin_init_weighted_new(self.C/self.number_samples, 1e-5, 100,lun,temp_y,np.array([]),temp_K,self.nu,1e-5,1e-3,1e-8)
        else:
            lista_esclusi = []
            if parent == 0:
                for kk,iii in enumerate(list_indeces):
                    if iii not in self.temp_list_root:
                        lista_esclusi.append(kk)
            else: 
                for kk,iii in enumerate(list_indeces):
                    if iii not in self.temp_list_indices:
                        lista_esclusi.append(kk)
            alphas, _ =  SMO.SMO_lin_init_weighted_new_snello(self.C/self.number_samples, 1e-5, 100,lun,temp_y,np.array([]),temp_K,self.nu,1e-5,1e-3,1e-8,np.array(lista_esclusi,dtype=int))
        
        self.alphas_y_node[parent,list_indeces] = self.number_samples*np.multiply(alphas,temp_y)/self.value
        self.alphas_node[parent,list_indeces] = self.number_samples*alphas/self.value
        self.b_node[parent] = b_calc(alphas,temp_y,self.alphas_y_node[parent,list_indeces]/self.number_samples,temp_K,self.C,lun)/self.value
        
        
        return True
    
    
    @staticmethod
    @jit(float64(float64[:],float64[:],float64[:],float64[:],int64[:],float64[:],int64[:,:],int64[:],float64,int64),nopython=True,parallel=True)
    def simple_dec_eval(var,b_node,sq_resp,K,len_node,list_tree,index_node,len_total,value,l_samples):
        """Evaluate one observation's loss contribution within a subtree.
        
        Parameters
        ----------
        var : numpy.ndarray
            Positive and negative parts of the subtree routing coefficients.
        b_node : numpy.ndarray
            Intercepts of the subtree branch nodes.
        sq_resp : numpy.ndarray
            Per-leaf loss terms supplied by the caller; in the KRLS internal-subtree evaluation these are sums of unsquared residuals.
        K : numpy.ndarray
            Distances from the observation to the training observations.
        len_node : numpy.ndarray
            Number of active coefficients at each branch node.
        list_tree : numpy.ndarray
            Subtree metadata, including depth and coefficient-array length.
        index_node : numpy.ndarray
            Active training indices associated with the branch nodes.
        len_total : numpy.ndarray
            Prefix offsets defining the coefficient slices of each node.
        value : float
            Logistic steepness parameter.
        l_samples : int
            Normalization factor in the routing probabilities.
        
        Returns
        -------
        float
            Probability-weighted sum of the supplied leaf loss terms for one observation.
        """
        depth = list_tree[0]
        number_samples = list_tree[1]
        C = list_tree[2]
        
        number_br_node = 2**depth - 1

        
        leaf_node = {}
        for i,j in enumerate(range(2**depth-1,2**(depth+1)-1)):
            leaf_node[j] = i  
        estr = [min(leaf_node),max(leaf_node)]
        BF_in_NL_L = {}
        BF_in_NL_R = {}
        for t in leaf_node:
            if len(alt_anchesters_numba(t,1,0))!=0:
                BF_in_NL_L[t] = alt_anchesters_numba(t,1,0)
            if len(alt_anchesters_numba(t,0,1))!=0:
                BF_in_NL_R[t] = alt_anchesters_numba(t,0,1)
        
        alp_plus = var[:number_samples]
        alp_neg = var[number_samples:]
        
        alp = alp_plus + alp_neg
        alp_y = alp_plus - alp_neg
        yy = np.sign(alp_y)
        
        output = 0
        
                
        
        single_output = 0
        for tl in leaf_node:
            
            mini_output = 1
            if tl != estr[1]:
                for t in BF_in_NL_L[tl]:
                    mini_output *= Pr_it(K[index_node[t][:len_node[t]]],alp_y[len_total[t]:len_total[t+1]],b_node[t],value,l_samples)
            if tl != estr[0]:
                for tr in BF_in_NL_R[tl]:
                    mini_output *= (1-Pr_it(K[index_node[tr][:len_node[tr]]],alp_y[len_total[tr]:len_total[tr+1]],b_node[tr],value,l_samples))
            single_output += mini_output*sq_resp[leaf_node[tl]]
        
            
        return single_output
    
    def evaluation_objective_node(self,list_indeces,node):
        """Evaluate working-set contributions for an imbalance-driven shuffle.
        
        Parameters
        ----------
        list_indeces : sequence of int
            First-entity training indices whose contributions are evaluated.
        node : int
            Root of the subtree, or an individual leaf.
        
        Returns
        -------
        numpy.ndarray
            One score for each requested first-entity observation.
        
        Notes
        -----
        For ``'kkls'`` at a leaf, the score sums squared errors over the
        second entity dimension. For a KRLS *internal subtree*, the original
        code instead sums unsquared residuals before probability weighting.
        This distinction is preserved here; these scores should not be
        assumed to equal the standard squared-error training objective.
        """
        
        steps = self.depth - math.floor(math.log(node+1,2))
        
        init = node
        for _ in range(steps):
        
            init = alt_descendant(init,right=False)[0]
        
        
        
        t = init
        
        if steps > 0:
            
            list_involved_leaf = []
            
            
            init = node
            for _ in range(steps):
            
                init = alt_descendant(init,right=False)[0]
            
            leaf_nodes_encoded = [] 
            for t in range(int(2**steps)):
                list_involved_leaf.append(init+t)
                leaf_nodes_encoded.append(self.leaf_node[init+t])
            
            
            
            if self.type_reg == 'linear_reg':
                y = self.y_train[list_indeces].copy().reshape(1,len(list_indeces))
                sq_resp = np.square((self.a_L[leaf_nodes_encoded].dot(self.X_train[list_indeces,:].T) + self.mu_L[leaf_nodes_encoded]) - y)
            elif self.type_reg == 'simple':
                y = self.y_train[list_indeces].copy().reshape(1,len(list_indeces))
                sq_resp = np.square(self.mu_L[leaf_nodes_encoded] - y)
            elif self.type_reg== 'kkls':
                y = self.y_train[list_indeces,:].copy()
                len_samples = len(list_indeces)
                
                sq_resp = np.zeros((2**steps,len_samples))
                for tt in range(2**steps):
                    
                    idxs = self.index_node_leaves[leaf_nodes_encoded[tt]][:self.len_node_leaves[leaf_nodes_encoded[tt]]]
                    
                    KK = self.K[np.ix_(list_indeces,idxs)]
                    
                    A = self.a_L[leaf_nodes_encoded[tt]][idxs ,:]
                    prediction = KK @ A @ self.P.T

                    residual = y - prediction
                    
                    sq_resp[tt,:] = residual.sum(axis=1)
            
            
            
            temp_branch = []
            subtree(node,steps-1,temp_branch)
            list_involved_branch = flat_list([[node],temp_branch])

            len_i = 0
            list_len = [0]
            for i in list_involved_branch:
                len_i += self.len_node[i]
                list_len.append(len_i)
            list_tree = np.array([steps,len_i,self.C])
            start_point = np.zeros(2*len_i)
            
            for j,i in enumerate(list_involved_branch):
                temp_len = self.len_node[i]
                temp_al_y = self.alphas_y_node[i,self.index_node[i][:temp_len]].copy()
                
                start_point[list_len[j]:list_len[j+1]] = (temp_al_y+np.abs(temp_al_y))/2
                start_point[len_i+(list_len[j]):len_i+(list_len[j+1])] = (np.abs(temp_al_y) - temp_al_y)/2
                
            
            evaluation = np.array([self.simple_dec_eval(start_point,self.b_node[list_involved_branch],sq_resp[:,j],self.K[i,:],self.len_node[list_involved_branch],list_tree,self.index_node[list_involved_branch],np.array(list_len,dtype=np.int64),self.value,self.number_samples) for j,i in enumerate(list_indeces)])
        else:
            
            if self.type_reg == 'simple':
                
                evaluation = np.array([(self.Phi_it(np.array([]),np.array([]),self.mu_L[self.leaf_node[init]][0])-self.y_train[i])**2 for i in list_indeces])
            elif self.type_reg == 'linear_reg':
            
                evaluation = np.array([(self.Phi_it(self.X_train[i],self.a_L[self.leaf_node[init]],self.mu_L[self.leaf_node[init]][0])-self.y_train[i])**2 for i in list_indeces])
            elif self.type_reg == 'kkls':
                idxx = self.index_node_leaves[self.leaf_node[init]][:self.len_node_leaves[self.leaf_node[init]]]
                a_leaf = self.a_L[self.leaf_node[init]][idxx,:]
                temp_res = []
                for i in list_indeces:
                    k = self.K[i,idxx]
                    
                    
                    res = (self.Phi_it_kkls(k,a_leaf,self.P) -self.y_train[i])**2
                    temp_res.append(sum(res))
                evaluation = np.array(temp_res)
                
                
        return evaluation
    
    
    def shuffle(self,node,brother,parent,eps3):
        """Try to rebalance sibling nodes by updating their routing rule.
        
        Select observations with the largest subtree loss contributions
        from brother, then update the parent routing rule to reassign a
        fraction toward node.
        
        Parameters
        ----------
        node : int
            Child intended to receive observations.
        brother : int
            Sibling from which candidate observations are selected.
        parent : int
            Parent branch node whose logistic routing rule is updated.
        eps3 : float
            Fraction controlling the number of candidates selected.
        
        Returns
        -------
        float
            Training objective evaluated after the attempted reassignment.
        
        Notes
        -----
        The method only calls the routing update if the candidate count
        computed from eps3 is greater than one. It prints diagnostic output.
        In KRLS mode the candidate ranking uses ``evaluation_objective_node()``
        with the original scoring rules described in that method.
        """


        list_indeces = np.array(self.instances_in_node_simple(brother))
        
        evaluations = self.evaluation_objective_node(list_indeces,brother)
        n = len(list_indeces)
        # number of instances to shuffle
        N = n*eps3
        #indeces to shake

        if N > 1:
            indeces = list_indeces[np.argpartition(evaluations, -int(N))[-int(N):]]
            
            #indeces to leave
            
            left_indeces_brother = list(set(list_indeces) - set(indeces))
            
            self.solving_patologic_logistic(node,left_indeces_brother,parent,brother)
            
            new_error = self.objective_train()
            print("loss post shuffle:")
            print(new_error)
        else:
            new_error = self.objective_train()
            print('No more shuffle')
        
        return new_error
        
    def back_up(self):
        """Save the current branch rules and leaf parameters for restoration.
        
        Returns
        -------
        bool
            True after copying the current parameter arrays.
        
        Notes
        -----
        Unlike the standard DSRT, the KRLS mode additionally saves the
        three-dimensional leaf coefficient array ``a_L`` and the per-leaf
        index and length arrays. This keeps the pairwise predictors and
        assigned first-entity subsets consistent when the state is restored.
        """
        self.temp_alphas_y_node = self.alphas_y_node.copy()
        self.temp_index_node = self.index_node.copy()
        self.temp_len_node = self.len_node.copy()
        self.temp_alphas_node = self.alphas_node.copy()
        if self.type_reg == 'linear_reg':
            self.temp_a_L = self.a_L.copy()
        elif self.type_reg == 'kkls':
            self.temp_a_L = self.a_L.copy()
            self.temp_index_node_leaves = self.index_node_leaves.copy()
            self.temp_len_node_leaves = self.len_node_leaves.copy()
            
        self.temp_mu_L = self.mu_L.copy()
        self.temp_b_node = self.b_node.copy() 
        return True
    
    def set_back_up(self):
        """Restore the most recently saved branch and leaf parameters.
        
        Returns
        -------
        bool
            True after restoring the saved arrays.
        
        Notes
        -----
        In ``'kkls'`` mode this restores ``a_L``, ``index_node_leaves`` and
        ``len_node_leaves`` in addition to the ordinary DSRT state. Call
        ``back_up()`` before using this method.
        """
        self.alphas_y_node = self.temp_alphas_y_node.copy()
        self.alphas_node = self.temp_alphas_node.copy()
        self.index_node = self.temp_index_node.copy()
        self.len_node = self.temp_len_node.copy()
        if self.type_reg == 'linear_reg':
            self.a_L = self.temp_a_L.copy()
        elif self.type_reg == 'kkls':
            self.a_L = self.temp_a_L.copy()
            self.index_node_leaves = self.temp_index_node_leaves.copy()
            self.len_node_leaves = self.temp_len_node_leaves.copy()
        self.mu_L = self.temp_mu_L.copy()
        self.b_node = self.temp_b_node.copy()
        return True
        
    def my_logistic_dec(self,n_iter = 50, mini_iter = 40, seed=5,eps1= 0.3,eps2=0.1,eps3= 0.4,decrement_rate = 0.8,init_flag = False,num_clust = 0,assign_labels= 'discretize',gower=True,parallel=True,solver='TNC'):
        """Train the DSRT with the original branch-node decomposition scheme.
        
        This is the entry point used by the setting B/C KRLS experiment script.
        The method traverses branch nodes in fixed index order, solves their
        routing subproblems, and updates descendant KRLS leaf regressions
        within the associated subtrees; it does not visit every leaf as a
        separate iteration step. It saves improving model states and restores
        the last saved state at the end.
        
        The convergence safeguards discussed in the paper, including
        Armijo-based reference updates, are not enforced in this
        experimental implementation.
        
        Parameters
        ----------
        n_iter : int, default=50
            Maximum number of passes over branch nodes.
        mini_iter : int, default=40
            Requested iteration limit for BN optimizers.
        seed : int, default=5
            Seed assigned to NumPy's global random generator at training start.
        eps1 : float, default=0.3
            Threshold for identifying imbalanced child-node assignments.
        eps2 : float, default=0.1
            Stronger threshold for selecting an imbalance-handling update.
        eps3 : float, default=0.4
            Fraction of candidate observations considered for reassignment.
        decrement_rate : float, default=0.8
            Multiplicative reduction of all three thresholds after each pass.
        init_flag : bool, default=False
            Whether to attempt clustering-based initialization. This option
            has not been verified with KRLS leaves.
        num_clust : int, default=0
            Number of clustering attempts when init_flag is enabled.
        assign_labels : str, default='discretize'
            Labeling option for the clustering routine.
        gower : bool, default=True
            Selects 1-K versus K in clustering initialization only; it does
            not change the precomputed matrices supplied to set_dataset().
        parallel : bool, default=True
            Select the Numba-parallel branch-node functions.
        solver : str, default='TNC'
            Optimizer for the branch-node subproblems.
        
        Returns
        -------
        list of float
            Recorded training-objective values at initialization and updates.
        
        Notes
        -----
        For the KRLS experiment, use ``type_reg='kkls'`` and
        ``set_nystrom(typ=None)``. This routine does not perform an additional
        individual-leaf sweep, unlike ``my_logistic_dec_new()``.
        """
        list_va_fu = []
        
        np.random.seed(seed)
        if init_flag:
            
            self.clust_init(num_clust,assign_labels,gower)
            
            objective_fun = self.objective_train()
            print("loss function:")
            print(objective_fun)
        else:
            if self.type_nystrom == None:
                

                print('Nonystrom')
            else:

                list_indeces= self.nystrom.fit(self.K,np.array(range(self.number_samples)))

                self.temp_list_root = list_indeces
                self.len_node[0] = len(self.temp_list_root)
                self.index_node[0][:self.len_node[0]] = self.temp_list_root
                

                
            self.solving_root(mini_iter,parallel=parallel,solver=solver)
            
            print("loss function:")
            objective_fun = self.objective_train()
            print(objective_fun)
            list_va_fu.append(objective_fun)
        self.back_up()
        
        i = 0
        better = True
        while( i < n_iter and better):
            print("ITERATION")
            print(i)
            random_nodes = list(range(2**self.depth-1))
            
            for t in random_nodes:
                print("NODE:")
                print(t)
                if t == 0:
                    
                    if self.type_nystrom == None:
                        self.temp_K = self.K.copy()

                    else:
                        list_indeces= self.nystrom.fit(self.K,np.array(range(self.number_samples)))

                        self.temp_list_root = list_indeces
                        self.len_node[0] = len(self.temp_list_root)
                        self.index_node[0][:self.len_node[0]] = self.temp_list_root
                    left_node = alt_descendant(t,right=False)[0]
                    right_node = alt_descendant(t,left=False)[0] 
                    number_instances_left = len(self.instances_in_node_simple(left_node))


                    if (number_instances_left/self.number_samples <= eps1 or number_instances_left/self.number_samples>=(1-eps1)):
                        
                    
                        
                        if (number_instances_left/self.number_samples <= eps2):
                            print("SOLVING SHUFFLE")
                            self.shuffle(left_node,right_node,t,eps3)
                        
                        elif (number_instances_left/self.number_samples >= (1-eps2)):
                            print("SOLVING SHUFFLE")
                            self.shuffle(right_node,left_node,t,eps3)
                        
                        else:
                            print("SOLVING LOGISTIC")
                            self.solving_logistic(left_node,t,right_node)
                    else:
                        print("SOLVING ROOT")
                            
                        self.solving_root(mini_iter,parallel=parallel,solver=solver)
                        

                    list_va_fu.append(self.objective_train())
                    if self.depth == 1:
                        self.set_temp_dataset(range(self.number_samples))
                        self.solving_phase_leaf(t,range(self.number_samples)) 
                else:
                    if self.type_nystrom == None:
                        list_indeces = self.instances_in_node_simple(t)
                        self.temp_K = self.K[np.ix_(list_indeces,list_indeces)]
                        number_instances = len(list_indeces)
                        self.set_temp_dataset(list_indeces)
                        self.len_node[t] = len(list_indeces)
                        self.index_node[t][:self.len_node[t]] = list_indeces
                        if self.type_reg== 'kkls':
                            self.temp_indices_per_kernel = list_indeces
                    else:
                        temp_list_indices = self.instances_in_node_simple(t)
                        self.set_temp_dataset(temp_list_indices)
                        if self.reduced_number < len(temp_list_indices):
                            list_indeces= self.nystrom.fit(self.K[np.ix_(temp_list_indices,temp_list_indices)],np.array(temp_list_indices))
                            self.temp_K = self.K[np.ix_(temp_list_indices,list_indeces)]
                            number_instances = len(temp_list_indices)
                            
                        else:
                            list_indeces = temp_list_indices
                            self.temp_K = self.K[np.ix_(temp_list_indices,temp_list_indices)]
                            number_instances = len(temp_list_indices)
                        self.len_node[t] = len(list_indeces)
                        self.index_node[t][:self.len_node[t]] = list_indeces
                        self.temp_list_indices = list_indeces
                      
                        if self.type_reg== 'kkls':
                            self.temp_indices_per_kernel = temp_list_indices
                    
                    if list_indeces:
                        left_node = alt_descendant(t,right=False)[0]
                        right_node = alt_descendant(t,left=False)[0] 
                        
                        number_instances_left = len(self.instances_in_node_simple(left_node))
                        
                        if (number_instances_left/number_instances <= eps1 or number_instances_left/number_instances>=(1-eps1)):

                            
                            if (number_instances_left/number_instances <= eps2):
                                if ( int((number_instances-number_instances_left)*eps3) < 1):
                                    continue
                                print("SOLVING SHUFFLE")
                                self.shuffle(left_node,right_node,t,eps3)
                        
                            elif (number_instances_left/number_instances >= (1-eps2)):
                                
                                if (int(number_instances_left*eps3)<1):
                                    continue
                                print("SOLVING SHUFFLE")
                                self.shuffle(right_node,left_node,t,eps3)
                        
                            else:
                                print("SOLO LOGISTICA")
                                self.solving_logistic(left_node,t,right_node)
                        else:
                            print("SOLVING NORMAL")
                            
                            self.solving_phase_branching(t,mini_iter,parallel=parallel,solver=solver)
                        if self.type_nystrom == None:
                            self.solving_phase_leaf(t,list_indeces)
                        else:
                            self.solving_phase_leaf(t,temp_list_indices)

                    else:
                        print("EMPTY NODE:")
                        print(t)
                new_objective = self.objective_train()
                if new_objective < objective_fun:
                    if (objective_fun - new_objective)/objective_fun <= 0.00000001:
                        better = True
                    print("SAVE ITERATION:")
                    print(i)
                    print('objective function stored')
                    print(self.objective_train())
                    self.back_up()
                    
                    objective_fun = new_objective
                    list_va_fu.append(new_objective)
            i+=1
            eps1 *= decrement_rate
            eps2 *= decrement_rate
            eps3 *= decrement_rate
        
        self.set_back_up()
        return list_va_fu
    
    
    
    
    def set_temp_dataset(self,list_indeces):
        """Store the response and optional feature rows of a working set.
        
        Parameters
        ----------
        list_indeces : sequence of int
            Training observation indices included in the current working set.
        
        Returns
        -------
        bool
            True after storing the temporary data.
        
        Notes
        -----
        Writes self.temp_y and, for linear-regression leaves, self.temp_x.
        In KRLS mode ``temp_y`` retains its two-dimensional interaction rows.
        """
        if self.type_reg == 'linear_reg':
            self.temp_x = self.X_train[list_indeces]
        self.temp_y = self.y_train[list_indeces]
        
        return True
    
    def logistic_for_clustering(self,list_indeces,parent,right_indeces):
        """Fit a branch-node routing rule from clustering-based binary labels.
        
        Parameters
        ----------
        list_indeces : sequence of int
            First-entity indices reaching the parent node.
        parent : int
            Branch node whose logistic coefficients are updated.
        right_indeces : sequence of int
            Indices assigned to the right child by clustering.
        
        Returns
        -------
        bool
            True after updating the routing coefficients and intercept;
            False when all parent observations go to the right child.
        
        Notes
        -----
        Calls the weighted Cython SMO solver and the historical ``b_calc``
        intercept routine. This initializes routing only; the KRLS-specific
        clustering leaf initialization is not implemented.
        """
        lun = len(list_indeces)
        temp_y = -np.ones(lun)
        temp_K = self.K[np.ix_(list_indeces,list_indeces)]
        if lun == len(right_indeces):
            return False
        
        it = 0
        for i in list_indeces:
            if i in right_indeces:
                temp_y[it]=1

            it+=1

        alphas, _ =  SMO.SMO_lin_init_weighted_new(self.C/self.number_samples, 1e-5, 1000,lun,temp_y,np.array([]),temp_K,self.nu,1e-5,1e-3,1e-8)
        
        self.alphas_y_node[parent,list_indeces] = self.number_samples*np.multiply(alphas,temp_y)/self.value
        self.alphas_node[parent,list_indeces] = self.number_samples*alphas/self.value
        self.b_node[parent] = b_calc(alphas,temp_y,self.alphas_y_node[parent,list_indeces]/self.number_samples,temp_K,self.C,lun)/self.value
        
        return True
    
    def clust_init(self,num_clust,assign_labels= 'cluster_qr',gower = True):
        
        """Initialize DSRT branch rules and leaf models using clustering.
        
        Several random clustering seeds are evaluated, and the valid
        partition with the largest returned silhouette score is selected.
        Leaf models are initialized first, followed by branch-node routing
        rules from the lowest internal level to the root.
        
        Parameters
        ----------
        num_clust : int
            Number of random clustering attempts.
        assign_labels : str, default='cluster_qr'
            Label-assignment method passed to the clustering utility.
        gower : bool, default=True
            Whether to supply 1-self.K (rather than self.K) to the
            clustering utility.
        
        Returns
        -------
        bool
            True after initialization or after falling back to a random
            starting point if no valid clustering partition is found.
        
        Notes
        -----
        The selected silhouette score is stored in self.sil. The method
        may reset NumPy's global RNG state several times.
        
        The leaf-initialization helper does not have a ``kkls`` branch;
        clustering initialization should not be assumed to work for KRLS.
        """
        
        ini = list(np.random.randint(low=0,high=10000, size=num_clust))
        temp_dic = {}
        
        for s in ini:
            np.random.seed(s)
            if gower:
                dic,sil = clustering_init_new(1-self.K,self.depth,s,assign_labels)
            else:
                dic,sil = clustering_init_new(self.K,self.depth,s,assign_labels)
            if dic !=-1 and sil!=-1:
                temp_dic[sil] = dic
        
        try:
            key = max(temp_dic)

        except ValueError:
            print("INITIALIZATION NOT FOUND!")
            self.starting_point(ini[0])
            self.sil = -1
            return True
        self.sil = key
        for t in temp_dic[key][self.depth]:
            self.set_temp_dataset(temp_dic[key][self.depth][t])
            self.solving_phase_leaf_clust(t)
            
            
        for d in reversed(range(self.depth)):
            
            for t in temp_dic[key][d]:
                left,right = alt_descendant(t)
                self.len_node[t] = len(temp_dic[key][d][t])
                self.index_node[t][:self.len_node[t]] = temp_dic[key][d][t] 
                self.logistic_for_clustering(list_indeces= temp_dic[key][d][t],parent=t,right_indeces=temp_dic[key][d+1][right])
                

        return True
    
    def sparsity_level(self,tau=1e-4):
        
        """Calculate the fraction of effectively zero coefficients per branch node.
        
        Parameters
        ----------
        tau : float, default=1e-4
            Absolute-value threshold above which a coefficient is nonzero.
        
        Returns
        -------
        list of float
            One value per branch node, calculated as one minus the number
            of active coefficients with absolute value above tau divided
            by the total number of training observations.
        
        Notes
        -----
        Training observations omitted by prototype selection contribute
        to the zero-coefficient fraction through the denominator.
        
        This is a thresholded routing-coefficient diagnostic, not the sparsity
        penalty of a KRLS objective or a prototype-selection rate.
        """

        list_sparse = []
        
        for i in range(self.alphas_node.shape[0]):
            
            list_sparse.append(float(1 - (sum(self.alphas_node[i,self.index_node[i][:self.len_node[i]]]> tau)/self.number_samples)))
        
        return list_sparse
            