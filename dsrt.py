# -*- coding: utf-8 -*-
"""
Created on May  7 10:56:51 2025

@author: Utente
"""
from functools import reduce
import operator
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from scipy import optimize
from util_regression_KLR import *
import numpy as np
import SMO
from utils_numba_l2_nystrom import * 
import time
from nystrom_utils_2 import *



class SSRT:
    """Distance-based Soft Regression Tree (DSRT) regression model.

    Branch nodes use distance-based logistic routing functions. Leaf nodes
    provide either constant predictions or linear regression models.
    Training alternates between branch-node and leaf-node subproblems.

    Parameters
    ----------
    depth : int, default=2
        Depth of the full binary tree.

    Notes
    -----
    Branch nodes are indexed from 0 to 2**depth - 2 and leaf nodes from
    2**depth - 1 to 2**(depth + 1) - 2. Call set_dataset() and
    set_param_KLR() before configuring prototype selection or training.
    """
    def __init__(self,depth = 2):
        """Initialize the tree structure and branch-to-leaf paths.

        Parameters
        ----------
        depth : int, default=2
            Number of levels of branch decisions in the full binary tree.

        Notes
        -----
        Initializes node counts, the leaf-index mapping, and the lists of
        left- and right-branch ancestors. It does not load training data.
        """
        self.depth = depth
        self.number_br_nodes = 2**depth -1
        self.number_leaf_nodes = 2**depth
        self.leaf_node  = {self.number_br_nodes+i:i for i in range(self.number_leaf_nodes)}
        self.BF_in_NL_R = {self.number_br_nodes+i:sorted(alt_anchesters(self.number_br_nodes+i,left=False)) for i in range(self.number_leaf_nodes)}
        self.BF_in_NL_L = {self.number_br_nodes+i:sorted(alt_anchesters(self.number_br_nodes+i,right=False)) for i in range(self.number_leaf_nodes)}
        self.zero_division = False
    
    def set_dataset(self,dissimilarity,dataset,type_reg = 'simple'):
        """Store the training data and initialize branch-node arrays.

        Parameters
        ----------
        dissimilarity : numpy.ndarray of shape (n_samples, n_samples)
            Pairwise training dissimilarities used by the routing functions.
        dataset : numpy.ndarray
            For type_reg='simple', the response array. For
            type_reg='linear_reg', a two-dimensional array with features in
            all columns except the last, which contains the responses.
        type_reg : {'simple', 'linear_reg'}, default='simple'
            Leaf prediction model: a constant or a linear regression.

        Returns
        -------
        bool
            True after initializing the model state.

        Notes
        -----
        Initializes the branch coefficients to zero and the index arrays to
        all training observations. The distance matrix is stored as self.K.
        """
        
        self.type_reg = type_reg
        if len(dataset.shape) > 1 and (self.type_reg=='linear_reg'):
            self.X_train = dataset[:,:-1]
            self.y_train = dataset[:,-1]
            self.number_features = self.X_train.shape[1]
        else:
            self.y_train = dataset
            
        self.number_samples = np.int64(self.y_train.shape[0])
        
        self.alphas_y_node = np.zeros((self.number_br_nodes,self.number_samples))
        self.index_node = np.multiply(np.ones((self.number_br_nodes,self.number_samples)),range(self.number_samples)).astype(np.int64)
        self.len_node = np.ones(self.number_br_nodes,dtype=np.int64)*int(self.number_samples)
        
        
        self.K = dissimilarity
        
        
        return True
    
    def set_nystrom(self,typ=None,percentage=0.2,ridge_leverage_lambda = 1,k_med_type = 0):
        """Configure optional prototype selection for branch-node optimization.

        Parameters
        ----------
        typ : {None, 'uniform', 'kmediod', 'DAC', 'recursive'}, default=None
            Selection strategy. 'uniform' implements RPS and 'kmediod' with
            k_med_type=0 implements the kMPS strategy in the paper. None
            disables prototype selection.
        percentage : float, default=0.2
            Fraction of training observations used to compute the requested
            prototype count, truncated to an integer.
        ridge_leverage_lambda : float, default=1
            Parameter passed to the alternative leverage-score strategies.
        k_med_type : int, default=0
            k-medoids selection rule. Zero selects medoids; values 1, 2,
            and 3 select alternative representatives (see nystrom_utils_2).

        Returns
        -------
        bool
            True after storing the configuration.

        Notes
        -----
        Requires self.number_samples, and a previously configured self.seed
        when typ is not None. The method does not select prototypes yet.
        """
        self.type_nystrom = typ
        self.perc = percentage
        self.reduced_number = int(percentage*self.number_samples)
        if typ != None:
            self.nystrom = Nystrom(n_components = self.reduced_number,sampling = self.type_nystrom,ridge_leverage_lambda = ridge_leverage_lambda,random_state = self.seed,k_med_type = k_med_type)
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
        self.nu = nu # tolerance for the bounds of the alpha variables
        self.value = value
        return True
    
    def starting_point(self,seed = 5):
        """Randomly initialize leaf predictions and branch-node parameters.

        Parameters
        ----------
        seed : int, default=5
            Seed used to initialize NumPy's global random generator.

        Returns
        -------
        bool
            True after initialization.

        Notes
        -----
        Initializes leaf intercepts and, for linear leaves, feature weights;
        then generates signed branch coefficients and computes intercepts
        using b_calc_snello(). This method changes NumPy's global RNG state.
        """
        
        
        np.random.seed(seed)
        
        if self.type_reg=='linear_reg':
            self.a_L = np.random.uniform(low=-600.0, high=600.0, size=(self.number_leaf_nodes,self.number_features))
            
        self.mu_L = np.random.uniform(low=-600.0, high=600.0, size=(self.number_leaf_nodes,1))
        self.b_node = np.empty(self.number_br_nodes)
        for i in range(self.number_br_nodes):
            temp_alp = np.random.uniform(low=0.0, high=self.C, size=self.number_samples)#low=1e-3 TODO: modificare con low=0.0
            label = np.random.choice([-1,1], size=self.number_samples)
            self.alphas_y_node[i] = np.multiply(temp_alp,label)
            self.b_node[i] = b_calc_snello(temp_alp,label.astype(np.float64),self.K.dot(self.alphas_y_node[i]),self.C,self.number_samples)
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
    
    
    def solving_root(self,max_iter = 20,parallel= True,solver='TNC'):
        """Optimize the routing coefficients and intercept of the root node.

        The remaining branch-node parameters and the leaf predictions are held
        fixed while the root subproblem is solved using the selected optimizer.

        Parameters
        ----------
        max_iter : int, default=20
            Iteration limit passed to supported solvers; the TNC branch does
            not explicitly pass this limit.
        parallel : bool, default=True
            Whether to use the parallel Numba objective and gradient.
        solver : str, default='TNC'
            Optimization method passed to scipy.optimize.minimize, or one
            of the separately handled Adam method names.

        Returns
        -------
        bool
            True after updating the root, or False for an unknown Adam solver.

        Notes
        -----
        Writes self.solution, self.alphas_y_node[0] and self.b_node[0].
        The subproblem includes an L2 penalty on routing coefficients.
        """

        
        l2_penalty=1e-3/self.number_samples
        
        # Initial stage of the decomposition procedure
        
        # a_L: n_leaf x n_features ; mu_L: n_leaf x 1; y: 1 x n_samples
        y = self.y_train.copy().reshape(1,self.number_samples)
        
        if self.type_reg == 'linear_reg':
            sq_resp = np.square((self.a_L.dot(self.X_train.T) + self.mu_L) - y)
        elif self.type_reg== 'simple':
            sq_resp = np.square(self.mu_L - y)
        
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
        

        self.alphas_y_node[0,self.index_node[0][:self.len_node[0]]] = self.solution[:-1]

        
            
        self.b_node[0] = self.solution[-1]

        return True


    def solving_phase_branching(self,node,max_iter = 5,tol = 1e-6,parallel=True,solver='TNC'):
        """Optimize the branch-node parameters of a subtree.

        The method builds the subtree rooted at node, forms the leaf residual
        terms for its working set, and jointly updates the routing coefficients
        and intercepts of the branch nodes in that subtree.

        Parameters
        ----------
        node : int
            Index of the subtree's root branch node.
        max_iter : int, default=5
            Iteration limit passed to supported solvers; not explicitly
            passed for TNC.
        tol : float, default=1e-6
            Accepted for interface compatibility; not used in this method.
        parallel : bool, default=True
            Whether to use the parallel Numba objective and gradient.
        solver : str, default='TNC'
            Optimization method passed to scipy.optimize.minimize, or an
            alternatively handled Adam method name.

        Returns
        -------
        bool
            True after updating the subtree, or False for an unknown Adam solver.

        Notes
        -----
        Uses the temporary working-set data and distance matrix prepared by
        the training procedure. Updates self.alphas_y_node and self.b_node
        for the branch nodes belonging to the subtree.
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
        y = self.temp_y.copy().reshape(1,len(self.temp_y))
        
        if self.type_reg == 'linear_reg':
            sq_resp = np.square((self.a_L[eq_leaf].dot(self.temp_x.T) + self.mu_L[eq_leaf]) - y)
        elif self.type_reg== 'simple':
            sq_resp = np.square(self.mu_L[eq_leaf] - y)
        
        
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
        
        
        
        alp_plus = self.solution[:total_len]
        

        for j,i in enumerate(list_involved_branch):


            self.alphas_y_node[i,index_involved] = alp_plus[j*len_node:(j+1)*len_node] 
            self.b_node[i] = self.solution[-len_involved_branch+j]

        return True
        
    # Utility for evaluating assignments on new observations
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
    
    
    def predict(self,dissimilarity,x): 
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
        """
        res = np.zeros(dissimilarity.shape[0])

            
        for t in self.leaf_node:
            ind = self.generic_instances_in_node(t,dissimilarity)
            
            if self.type_reg == 'simple':
                
                for i in ind:
                    res[i] = self.Phi_it(np.array([]),np.array([]),self.mu_L[self.leaf_node[t]][0])
                
            elif self.type_reg == 'linear_reg':
            
                for i in ind:
                    res[i] = self.Phi_it(x[i],self.a_L[self.leaf_node[t]],self.mu_L[self.leaf_node[t]][0])
                        
        return res
    
    def predict_label_level(self,K_temp,level = 0):# The root is at level 0
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
    @jit(float64[:](int64,int64,float64[:,:],float64[:],float64[:,:],float64[:],int64[:],int64[:],float64,int64), nopython=True, fastmath= True)
    def A_numba_solved(node,steps,alp_y,b_node,K,temp_y,list_tree,index_involved,value,l_samples):
        # Constant leaf predictions: intercept only, without input features
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
        """Update the leaf models within a selected subtree (LN Step).

        For constant leaves, use routing-probability-weighted response means.
        For linear leaves, solve the corresponding weighted least-squares
        system using scipy.sparse.linalg.lsqr.

        Parameters
        ----------
        node : int, default=0
            Root branch node of the selected subtree.
        list_idx : sequence of int, default=[]
            Training observation indices used for the current leaf update.

        Returns
        -------
        bool
            True after updating the selected leaf parameters.

        Notes
        -----
        Writes self.mu_L and, for linear leaves, self.a_L. The existing
        mutable default argument is preserved unchanged.
        """
        steps = self.depth - math.floor(math.log(node+1,2))
        list_involved_leaf = []
        init = node
        for _ in range(steps):
            init = alt_descendant(init,right=False)[0]
            
        for t in range(int(2**steps)):
            list_involved_leaf.append(init+t)
            
        len_node = self.len_node[node]
        index_involved = self.index_node[node][:len_node]
            
        steps = int(self.depth - math.floor(math.log(node+1,2)))
        
        if self.type_reg == 'simple':
            
            x = self.A_numba_solved(node,steps,self.alphas_y_node,self.b_node,self.K[np.ix_(list_idx,index_involved)],self.temp_y,np.array([self.depth],dtype=np.int64),index_involved,self.value,self.number_samples)
            for t in range(2**steps):
                self.mu_L[self.leaf_node[list_involved_leaf[t]]][0] = x[t]
        elif self.type_reg == 'linear_reg':
        
            Ab = self.create_AA_numba(node,steps,self.alphas_y_node,self.b_node,self.K[np.ix_(list_idx,index_involved)],self.temp_x,self.temp_y,np.array([self.depth,self.number_features],dtype=np.int64),index_involved,self.value,self.number_samples)
    
            A_matrix = csc_matrix(Ab[:,:-1], dtype=float)
            res, istop, itn, normr = lsqr(A_matrix,Ab[:,-1])[:4]
            x = res
            
            for t in range(2**steps):
                for j in range(self.number_features):
                    self.a_L[self.leaf_node[list_involved_leaf[t]],j] = x[((self.number_features+1)*t)+j]
                self.mu_L[self.leaf_node[list_involved_leaf[t]]][0] = x[((self.number_features+1)*t)+self.number_features]

        return True
    
    def solving_phase_leaf_clust(self,node):
        """Fit one leaf model from its clustering-based initialization set.

        Parameters
        ----------
        node : int
            Leaf node whose parameters are initialized.

        Returns
        -------
        bool
            True after setting the leaf model.

        Notes
        -----
        For a linear leaf, uses unweighted least squares; for a constant
        leaf, uses the arithmetic mean of the temporary responses.
        """
        if self.type_reg == 'linear_reg':
            Ab = self.create_AA_numba_clust(node,0,self.temp_x,self.temp_y,np.array([self.depth,self.number_features],dtype=np.int64))
            
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
        """Evaluate the weighted mean soft-routing squared error on the training set.

        Returns
        -------
        float
            Mean over observations of the leaf-specific squared errors,
            weighted by their soft routing probabilities.

        Notes
        -----
        This is the prediction-error component evaluated for monitoring;
        it does not add the L2 penalties of the branch-node subproblems.
        """
        out = 0
        for i in range(self.number_samples):
            out += self.y_response_train(i)
        
        return out/self.number_samples
    
    def y_response_train(self,i):
        """Compute one observation's contribution to the training objective.

        Parameters
        ----------
        i : int
            Index of a training observation.

        Returns
        -------
        float
            Sum over leaves of squared prediction errors weighted by the
            observation's soft probability of reaching each leaf.
        """

        
        response = 0
        
        if self.type_reg == 'simple':
            
            for t in self.leaf_node:
                response += self.Pr_it_n_train(i,t)*(self.Phi_it(np.array([]),np.array([]),self.mu_L[self.leaf_node[t]][0])-self.y_train[i])**2

        else:
            
            for t in self.leaf_node:
                response += self.Pr_it_n_train(i,t)*(self.Phi_it(self.X_train[i],self.a_L[self.leaf_node[t]],self.mu_L[self.leaf_node[t]][0])-self.y_train[i])**2
        
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
        
        self.b_node[parent] = b_calc_snello(alphas,temp_y,temp_K.dot(self.alphas_y_node[parent,list_indeces]/self.number_samples),self.C,lun)/self.value

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

        
        self.b_node[parent] = b_calc_snello(alphas,temp_y,temp_K.dot(self.alphas_y_node[parent,list_indeces]/self.number_samples),self.C,lun)/self.value
        
        
        
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
            Squared prediction errors for the subtree leaves.
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
            Probability-weighted sum of leaf squared errors for one observation.
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
        """Evaluate per-observation losses for the subtree rooted at node.

        Parameters
        ----------
        list_indeces : sequence of int
            Training indices of the observations to evaluate.
        node : int
            Root of the selected subtree, which may also be a leaf.

        Returns
        -------
        numpy.ndarray
            Loss contribution for each requested observation. For a leaf,
            this is its squared prediction error; for a branch subtree,
            the squared errors are weighted by routing probabilities
            computed within that subtree.
        """

        # array with values of the impact of instances (list_indeces) in the objective function of the subtree from the root = node 
        
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
            
            y = self.y_train[list_indeces].copy().reshape(1,len(list_indeces))
            
            if self.type_reg == 'linear_reg':
                sq_resp = np.square((self.a_L[leaf_nodes_encoded].dot(self.X_train[list_indeces,:].T) + self.mu_L[leaf_nodes_encoded]) - y)
            elif self.type_reg == 'simple':
                sq_resp = np.square(self.mu_L[leaf_nodes_encoded] - y)
            
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
        """
        # node receives observations; brother is the sibling supplying them



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
        """Save a copy of the current branch and leaf model parameters.

        Returns
        -------
        bool
            True after saving the state.

        Notes
        -----
        Copies the routing coefficients, active-index arrays, branch
        intercepts, and leaf parameters into temporary attributes.
        """
        self.temp_alphas_y_node = self.alphas_y_node.copy()
        self.temp_index_node = self.index_node.copy()
        self.temp_len_node = self.len_node.copy()

        if self.type_reg == 'linear_reg':
            self.temp_a_L = self.a_L.copy()
        self.temp_mu_L = self.mu_L.copy()
        self.temp_b_node = self.b_node.copy() 
        return True
    
    def set_back_up(self):
        """Restore the model parameters from the last saved state.

        Returns
        -------
        bool
            True after restoring the saved state.

        Notes
        -----
        Restores the routing coefficients, active indices, branch
        intercepts, and leaf parameters stored by back_up().
        """
        self.alphas_y_node = self.temp_alphas_y_node.copy()
        self.index_node = self.temp_index_node.copy()
        self.len_node = self.temp_len_node.copy()
        if self.type_reg == 'linear_reg':
            self.a_L = self.temp_a_L.copy()
        self.mu_L = self.temp_mu_L.copy()
        self.b_node = self.temp_b_node.copy()
        return True
        
    def my_logistic_dec(self,n_iter = 50, mini_iter = 40, seed=5,eps1= 0.3,eps2=0.1,eps3= 0.4,decrement_rate = 0.8,init_flag = False,num_clust = 0,assign_labels= 'discretize',gower=True,parallel=True,solver='TNC'):
        """Train the DSRT using the branch-node/leaf-node decomposition.

        The procedure initializes the model (optionally from clustering),
        updates branch-node routing rules and leaf models, and handles
        imbalanced child partitions. It retains the best state encountered
        according to the monitored training objective.

        The convergence safeguards discussed in the paper, including
        Armijo-based reference updates, are not enforced in this
        experimental implementation.

        Parameters
        ----------
        n_iter : int, default=50
            Maximum number of full passes over the branch nodes.
        mini_iter : int, default=40
            Iteration limit requested for individual branch subproblems;
            it is not explicitly passed to SciPy's TNC branch.
        seed : int, default=5
            Seed used for NumPy's global random generator at training start.
        eps1 : float, default=0.3
            Outer threshold for identifying an imbalanced child split.
        eps2 : float, default=0.1
            Stronger imbalance threshold for triggering reassignment.
        eps3 : float, default=0.4
            Fraction of candidate observations considered for reassignment.
        decrement_rate : float, default=0.8
            Multiplicative reduction applied to all three thresholds after
            each full pass over branch nodes.
        init_flag : bool, default=False
            Whether to use clustering-based initialization.
        num_clust : int, default=0
            Number of random clustering initializations attempted.
        assign_labels : str, default='discretize'
            Label-assignment strategy passed to the clustering routine.
        gower : bool, default=True
            If clustering initialization is enabled, whether to pass 1-K
            rather than K to the spectral-clustering utility. This flag
            does not control how the original distance matrix is computed.
        parallel : bool, default=True
            Whether to use the parallel Numba branch objectives and gradients.
        solver : str, default='TNC'
            Optimization method for the branch-node subproblems.

        Returns
        -------
        list of float
            Recorded training-objective values, including monitored values
            from initial and subsequent updates. This is not a fixed-length
            record of one value per full iteration.

        Notes
        -----
        After training, restores the best saved model state. The method
        updates the model in place and changes NumPy's global RNG state.
        """
        list_va_fu = []
        
        np.random.seed(seed)
        if init_flag:
            
            self.clust_init(num_clust,assign_labels,gower)
            
            objective_fun = self.objective_train()
            print("funzione obiettivo:")
            print(objective_fun)
        else:
            if self.type_nystrom == None:
                
                print('No prototype selection')
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
                                print("ONLY LOGISTIC")
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
                        better = False
                    print("SAVE ITERATION:")
                    print(i)
                    print('loss stored')
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
        """
        if self.type_reg == 'linear_reg':
            self.temp_x = self.X_train[list_indeces]
        self.temp_y = self.y_train[list_indeces]
        
        return True
    
    def logistic_for_clustering(self,list_indeces,parent,right_indeces):
        """Fit a branch routing rule from clustering-generated child labels.

        Parameters
        ----------
        list_indeces : sequence of int
            Training indices assigned to the parent node.
        parent : int
            Parent branch node whose routing rule is initialized.
        right_indeces : sequence of int
            Indices assigned to the right child; these receive label +1.

        Returns
        -------
        bool
            True after initialization, or False if all parent observations
            belong to the right child.

        Notes
        -----
        Solves weighted KLR with the Cython SMO routine, then stores signed
        routing coefficients and the branch intercept in the model.
        """
        # Initialize a routing rule from the clustering partition
        
        # list_indeces contains the observations assigned to the parent node



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
        self.b_node[parent] = b_calc_snello(alphas,temp_y,temp_K.dot(self.alphas_y_node[parent,list_indeces]/self.number_samples),self.C,lun)/self.value
        
        
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
            print("Good initialization not found!")
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
        """
        # Include unselected prototype coefficients as zeros in the sparsity measure

        list_sparse = []
        
        for i in range(self.alphas_y_node.shape[0]):
            
            list_sparse.append(float(1 - (sum(np.abs(self.alphas_y_node[i,self.index_node[i][:self.len_node[i]]])> tau)/self.number_samples)))
        
        return list_sparse
            