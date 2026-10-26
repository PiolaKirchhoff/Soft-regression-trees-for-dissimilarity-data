# -*- coding: utf-8 -*-
"""Weighted Kernel Logistic Regression (KLR) optimization for DSRTs.

This module implements an SMO-type decomposition algorithm for
weighted KLR subproblems arising during the training of
Distance-based Soft Regression Trees (DSRTs).

The implementation is based on the decomposition approach proposed
by Consolo, Manno, and Amaldi (2026), adapted to weighted KLR
without the sparsity-inducing term considered in their paper.

Reference
---------
Consolo, A., Manno, A., and Amaldi, E. (2026).
Binary kernel logistic regression: A sparsity-inducing formulation
and a convergent decomposition training algorithm.
Computers & Operations Research, 188, 107363.
https://doi.org/10.1016/j.cor.2025.107363
"""
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
import numpy as np
cimport numpy as np
import util
import function
import time
from cython cimport cdivision, boundscheck,wraparound
from scipy.optimize import minimize
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t DTYPE_tt
np.import_array()
cdef extern from "math.h":
    double log(double x) nogil  
    


@cdivision(True)
@boundscheck(False)
@wraparound(False)
def SMO_lin_init_weighted_new(double C, double tol, unsigned long max_iter,unsigned long length,np.ndarray[DTYPE_t, ndim=1] y,np.ndarray[DTYPE_t, ndim=1] init,np.ndarray[DTYPE_t, ndim=2] K,double ni,double xi,double tau,double toll):
    
    """Solve a weighted KLR subproblem using an SMO-type decomposition.

    The class weights are based on the number of observations in each
    class when the initial coefficient array is empty. Each iteration
    selects a pair of coefficients and solves a one-dimensional
    subproblem using Newton-CG.

    Parameters
    ----------
    C : float
        Upper bound for the coefficients.
    tol : float
        Stopping tolerance for the decomposition algorithm.
    max_iter : int
        Maximum number of SMO iterations.
    length : int
        Number of observations in the subproblem.
    y : numpy.ndarray, shape (length,)
        Binary labels encoded as -1 and 1.
    init : numpy.ndarray
        Initial coefficients. An empty array requests the built-in
        initialization; otherwise, the supplied array is used directly.
    K : numpy.ndarray, shape (length, length)
        Pairwise matrix used by the KLR subproblem (a distance matrix
        in the DSRT experiments).
    ni : float
        Parameter passed to the gradient initialization and the
        one-dimensional subproblem routines.
    xi : float
        Tolerance used to identify eligible coefficient indices.
    tau : float
        Positive fallback curvature used if the computed curvature
        is not positive.
    toll : float
        Tolerance used to keep updated coefficients inside their bounds.

    Returns
    -------
    alphas : numpy.ndarray
        Optimized coefficient array. If `init` was nonempty, this is
        the same array, modified in place.
    it : int
        Number of iterations performed.
    """
    cdef int m = length  
    cdef int i
    cdef int ii
    cdef int k
    cdef int new_i_up
    cdef int i_up
    cdef int i_low
    cdef int it = 0
    cdef int m1 = y[y == 1].shape[0]
    cdef int m2 = y[y == -1].shape[0]
    cdef double obj_old = np.inf
    cdef double obj_new
    cdef double temp_max
    cdef double temp_min
    cdef double tt
    cdef double mm
    cdef double M
    cdef double v
    cdef double old_alpha_up
    cdef double old_alpha_low
    cdef double b_up = float('-inf')
    cdef double b_low = float('inf')
    cdef bint condition = True
    cdef double[:,::1] K_view = K
    cdef double res
    cdef double ntol = tol
    cdef double alp_up
    cdef double alp_down
    cdef np.ndarray[DTYPE_t, ndim=1] alphas
    cdef double init_val = min(1.0,C)

    if init.shape[0] == 0:
        
        alphas = np.zeros(m)
    else:
        alphas = init
   
    cdef double* alp_view = &alphas[0]
    cdef double* y_view = &y[0]
    
    # Set class weights according to the class frequencies
    cdef np.ndarray[DTYPE_t, ndim=1] class_weight = np.zeros(m)
    if init.shape[0] == 0:
        for ii in range(m):
            if y_view[ii] == 1:
                alp_view[ii] = init_val/m1
                class_weight[ii] = m/(2*m1)
            else:
                alp_view[ii] = init_val/m2
                class_weight[ii] = m/(2*m2)
    
    cdef double[::1] HH = util.init_H_weight(alphas,y,K,class_weight,m,C,ni)
    cdef DTYPE_tt[::1] I_up = util.first_Iup(alphas,y,C-xi,xi,m)
    cdef DTYPE_tt[::1] I_low = util.first_Ilow(alphas,y,C-xi,xi,m)
    
    cdef DTYPE_tt* I_up_view = &I_up[0]
    cdef DTYPE_tt* I_low_view = &I_low[0]
    cdef double* HH_view = &HH[0]
    
    
    cdef int iii
    cdef int j
    cdef double temp_val = 1.0
    
    cdef double grad = 0.0
    cdef double a = 0.0
    cdef double vall 

    cdef int index = -1

    cdef double H_up
    cdef double H_down

    new_i_up = -1
    temp_max = float('-inf')
    for k in range(m):
        if I_up_view[k] == 1:
            if temp_max < -HH_view[k]:
                new_i_up = k
                temp_max = -HH_view[k]
 
    
    while condition and it < max_iter:


        i_up = new_i_up
        
        index = -1

        temp_val = 0.0
        H_up = HH_view[i_up]
        for iii in range(m):
            
            if I_low_view[iii] == 1:
                
                if -HH_view[iii] < temp_max:
                    # Compute the curvature used to select the second index
                    condition2 = True
                    a = K_view[i_up, i_up] + K_view[iii, iii] - 2*K_view[i_up, iii] + class_weight[i_up]*(C)/((alp_view[i_up])*(C-(alp_view[i_up]))) + class_weight[iii]*(C)/((alp_view[iii])*(C-(alp_view[iii])))
                    
                    
                    grad = H_up - HH_view[iii]
                    if a > 0:
                        vall = - (grad*grad)/a
                    else:
                        vall = - (grad*grad)/tau
                    
                    if vall < temp_val:
                        index = iii
                        temp_val = vall

        
        
        i_low = index
        if index == -1:
            break
        # Optimize the selected pair with a one-dimensional Newton-CG step
        result = minimize(function.alt_objective_sc_new_weight, x0 = 0,args= (i_up,i_low,alphas,y,K,C,ni,m,class_weight[i_up],class_weight[i_low]) ,jac = function.line_search_grad_sc_weight,hess = function.hess_weight,method='Newton-CG')
        
        res = result.x

        
        
        M = max(alp_view[i_up] + res/y_view[i_up],alp_view[i_low] - res/y_view[i_low]) - C
        mm = min(alp_view[i_up] + res/y_view[i_up],alp_view[i_low] - res/y_view[i_low])

        if M > 0 and mm < 0:
            v = max(M,abs(mm))
            if res >0:
                tt = max(res - v,0)
            else:
                tt = max(res + v, 0)
        elif M > 0:
            if res >0:
                tt = max(res - M,0)
            else:
                tt = max(res + M, 0)

        elif mm < 0:
            if res >0:
                tt = max(res + mm,0)
            else:
                tt = max(res - mm, 0)

        else:
            tt = res


        old_alpha_up = alp_view[i_up]
        old_alpha_low = alp_view[i_low]
        
        
        alp_up = old_alpha_up + tt/y_view[i_up]
        alp_down = old_alpha_low - tt/y_view[i_low]
        
        if alp_up < toll:
            alp_view[i_up] = toll
        elif alp_up > C - toll:
            alp_view[i_up] = C - toll
        else:
            alp_view[i_up] = alp_up
            
        if alp_down < toll:
            alp_view[i_low] = toll
        elif alp_down > C - toll:
            alp_view[i_low] = C - toll
        else:
            alp_view[i_low] = alp_down
            
        
        # update I_up
        if y_view[i_up] == 1:
            if alp_view[i_up] >= C-xi:
                I_up_view[i_up] = 0
        else:
            if alp_view[i_up] <= xi:
                I_up_view[i_up] = 0

        if y_view[i_low] == 1:
            if alp_view[i_low] < C-xi:
                I_up_view[i_low] = 1
        else:
            if alp_view[i_low] > xi:
                I_up_view[i_low] = 1
        
        # update I_low
        if y_view[i_low] == -1:
            if alp_view[i_low] >= C-xi:
                I_low_view[i_low] = 0
        else:
            if alp_view[i_low] <= xi:
                I_low_view[i_low] = 0
                
        if y_view[i_up] == -1:
            if alp_view[i_up] < C-xi:
                I_low_view[i_up] = 1
        else:
            if alp_view[i_up] > xi:
                I_low_view[i_up] = 1
        # update no pair 
        
        b_up = HH_view[i_up]
        b_low = HH_view[i_low]
        
        temp_max = float('-inf')
        temp_min = float('inf')
        for i in range(m):
            
            if i!= i_up and i!= i_low: 
                HH_view[i]+= tt*(K_view[i,i_up]-K_view[i,i_low])  
            elif i == i_up:
                HH_view[i_up] += tt*K_view[i_up,i_up] -tt*K_view[i_up,i_low] + class_weight[i_up]*y_view[i_up]*(log(alp_view[i_up]/(C-alp_view[i_up])) - log(old_alpha_up/(C-old_alpha_up)))
            else:
                HH_view[i_low] += tt*K_view[i_low,i_up] -tt*K_view[i_low,i_low]+ class_weight[i_low]*y_view[i_low]*(log(alp_view[i_low]/(C-alp_view[i_low])) - log(old_alpha_low/(C-old_alpha_low)))
            
            if I_up_view[i] == 1:
                if temp_max < -HH_view[i]:
                    new_i_up = i
                    temp_max = -HH_view[i]
            
            if I_low_view[i] == 1:
                if temp_min > -HH_view[i]:
                    temp_min = -HH_view[i]
            
        if temp_max - ntol <= temp_min:
            condition = False

        it+= 1


    return alphas,it



@cdivision(True)
@boundscheck(False)
@wraparound(False)
def SMO_lin_init_weighted_new_snello(double C, double tol, unsigned long max_iter,unsigned long length,np.ndarray[DTYPE_t, ndim=1] y,np.ndarray[DTYPE_t, ndim=1] init,np.ndarray[DTYPE_t, ndim=2] K,double ni,double xi,double tau,double toll,np.ndarray[DTYPE_tt, ndim=1] lista_esclusi):
    """Solve the weighted KLR subproblem with restricted SMO updates.

    This variant is used with prototype selection in the DSRT
    experiments. Indices in `lista_esclusi` are skipped when choosing
    coefficient pairs to update. With an empty `init` array, their
    coefficients are initialized to `xi`.

    Parameters
    ----------
    C : float
        Upper bound for the coefficients.
    tol : float
        Stopping tolerance for the decomposition algorithm.
    max_iter : int
        Maximum number of SMO iterations.
    length : int
        Number of observations in the subproblem.
    y : numpy.ndarray, shape (length,)
        Binary labels encoded as -1 and 1.
    init : numpy.ndarray
        Initial coefficients. An empty array requests the built-in
        initialization; otherwise, the supplied array is used directly.
    K : numpy.ndarray, shape (length, length)
        Pairwise matrix used by the KLR subproblem (a distance matrix
        in the DSRT experiments).
    ni : float
        Parameter passed to the gradient initialization and the
        one-dimensional subproblem routines.
    xi : float
        Tolerance for eligible indices, also used to initialize
        excluded coefficients when `init` is empty.
    tau : float
        Positive fallback curvature used if the computed curvature
        is not positive.
    toll : float
        Tolerance used to keep updated coefficients inside their bounds.
    lista_esclusi : numpy.ndarray of int64
        Indices excluded from selection for the SMO coefficient updates.

    Returns
    -------
    alphas : numpy.ndarray
        Optimized coefficient array. If `init` was nonempty, this is
        the same array, modified in place.
    it : int
        Number of iterations performed.
    """

    cdef int m = length  
    cdef int i
    cdef int ii
    cdef int k
    cdef int new_i_up
    cdef int i_up
    cdef int i_low
    cdef int it = 0
    cdef int m1 = y[y == 1].shape[0]
    cdef int m2 = y[y == -1].shape[0]
    cdef double obj_old = np.inf
    cdef double obj_new
    cdef double temp_max
    cdef double temp_min
    cdef double tt
    cdef double mm
    cdef double M
    cdef double v
    cdef double old_alpha_up
    cdef double old_alpha_low
    cdef double b_up = float('-inf')
    cdef double b_low = float('inf')
    cdef bint condition = True
    cdef double[:,::1] K_view = K
    cdef double res
    cdef double ntol = tol
    cdef double alp_up
    cdef double alp_down
    cdef np.ndarray[DTYPE_t, ndim=1] alphas
    cdef double init_val = min(1.0,C)

    if init.shape[0] == 0:
        
        alphas = np.zeros(m)
    else:
        alphas = init
   
    cdef double* alp_view = &alphas[0]
    cdef double* y_view = &y[0]
    
    cdef np.ndarray[DTYPE_t, ndim=1] class_weight = np.zeros(m)
    if init.shape[0] == 0:
        for ii in range(m):
            if y_view[ii] == 1:
                if ii in lista_esclusi:
                    alp_view[ii] = xi
                else:
                    alp_view[ii] = init_val/m1
                class_weight[ii] = m/(2*m1)
                
            else:
                if ii in lista_esclusi:
                    alp_view[ii] = xi
                else:
                    alp_view[ii] = init_val/m2
                
                
                class_weight[ii] = m/(2*m2)
    
    cdef double[::1] HH = util.init_H_weight(alphas,y,K,class_weight,m,C,ni)
    cdef DTYPE_tt[::1] I_up = util.first_Iup(alphas,y,C-xi,xi,m)
    cdef DTYPE_tt[::1] I_low = util.first_Ilow(alphas,y,C-xi,xi,m)
    
    cdef DTYPE_tt* I_up_view = &I_up[0]
    cdef DTYPE_tt* I_low_view = &I_low[0]
    cdef double* HH_view = &HH[0]
    
    
    cdef int iii
    cdef int j
    cdef double temp_val = 1.0
    
    cdef double grad = 0.0
    cdef double a = 0.0
    cdef double vall 

    cdef int index = -1

    cdef double H_up
    cdef double H_down

    new_i_up = -1
    temp_max = float('-inf')
    for k in range(m):
        if k not in lista_esclusi:
            if I_up_view[k] == 1:
                if temp_max < -HH_view[k]:
                    new_i_up = k
                    temp_max = -HH_view[k]
 
    
    while condition and it < max_iter:


        i_up = new_i_up
        
        index = -1

        temp_val = 0.0
        H_up = HH_view[i_up]
        for iii in range(m):
            
            if iii not in lista_esclusi:
                if I_low_view[iii] == 1:
                    
                    if -HH_view[iii] < temp_max:
                        # Hessian calculation
                        condition2 = True
                        a = K_view[i_up, i_up] + K_view[iii, iii] - 2*K_view[i_up, iii] + class_weight[i_up]*(C)/((alp_view[i_up])*(C-(alp_view[i_up]))) + class_weight[iii]*(C)/((alp_view[iii])*(C-(alp_view[iii])))
                        
                        
                        grad = H_up - HH_view[iii]
                        if a > 0:
                            vall = - (grad*grad)/a
                        else:
                            vall = - (grad*grad)/tau
                        
                        if vall < temp_val:
                            index = iii
                            temp_val = vall

        
        
        i_low = index
        if index == -1:
            break
        
        result = minimize(function.alt_objective_sc_new_weight, x0 = 0,args= (i_up,i_low,alphas,y,K,C,ni,m,class_weight[i_up],class_weight[i_low]) ,jac = function.line_search_grad_sc_weight,hess = function.hess_weight,method='Newton-CG')
        
        res = result.x

        
        
        M = max(alp_view[i_up] + res/y_view[i_up],alp_view[i_low] - res/y_view[i_low]) - C
        mm = min(alp_view[i_up] + res/y_view[i_up],alp_view[i_low] - res/y_view[i_low])

        if M > 0 and mm < 0:
            v = max(M,abs(mm))
            if res >0:
                tt = max(res - v,0)
            else:
                tt = max(res + v, 0)
        elif M > 0:
            if res >0:
                tt = max(res - M,0)
            else:
                tt = max(res + M, 0)

        elif mm < 0:
            if res >0:
                tt = max(res + mm,0)
            else:
                tt = max(res - mm, 0)

        else:
            tt = res


        old_alpha_up = alp_view[i_up]
        old_alpha_low = alp_view[i_low]
        
        
        alp_up = old_alpha_up + tt/y_view[i_up]
        alp_down = old_alpha_low - tt/y_view[i_low]
        
        if alp_up < toll:
            alp_view[i_up] = toll
        elif alp_up > C - toll:
            alp_view[i_up] = C - toll
        else:
            alp_view[i_up] = alp_up
            
        if alp_down < toll:
            alp_view[i_low] = toll
        elif alp_down > C - toll:
            alp_view[i_low] = C - toll
        else:
            alp_view[i_low] = alp_down
            
        
        # update I_up
        if y_view[i_up] == 1:
            if alp_view[i_up] >= C-xi:
                I_up_view[i_up] = 0
        else:
            if alp_view[i_up] <= xi:
                I_up_view[i_up] = 0

        if y_view[i_low] == 1:
            if alp_view[i_low] < C-xi:
                I_up_view[i_low] = 1
        else:
            if alp_view[i_low] > xi:
                I_up_view[i_low] = 1
        
        # update I_low
        if y_view[i_low] == -1:
            if alp_view[i_low] >= C-xi:
                I_low_view[i_low] = 0
        else:
            if alp_view[i_low] <= xi:
                I_low_view[i_low] = 0
                
        if y_view[i_up] == -1:
            if alp_view[i_up] < C-xi:
                I_low_view[i_up] = 1
        else:
            if alp_view[i_up] > xi:
                I_low_view[i_up] = 1
        # update no pair 
        
        b_up = HH_view[i_up]
        b_low = HH_view[i_low]
        
        temp_max = float('-inf')
        temp_min = float('inf')
        for i in range(m):
            
            if i!= i_up and i!= i_low: 
                HH_view[i]+= tt*(K_view[i,i_up]-K_view[i,i_low])  
            elif i == i_up:
                HH_view[i_up] += tt*K_view[i_up,i_up] -tt*K_view[i_up,i_low] + class_weight[i_up]*y_view[i_up]*(log(alp_view[i_up]/(C-alp_view[i_up])) - log(old_alpha_up/(C-old_alpha_up)))
            else:
                HH_view[i_low] += tt*K_view[i_low,i_up] -tt*K_view[i_low,i_low]+ class_weight[i_low]*y_view[i_low]*(log(alp_view[i_low]/(C-alp_view[i_low])) - log(old_alpha_low/(C-old_alpha_low)))
            
            if I_up_view[i] == 1:
                if i not in lista_esclusi:
                    if temp_max < -HH_view[i]:
                        new_i_up = i
                        temp_max = -HH_view[i]
            
            if I_low_view[i] == 1:
                if temp_min > -HH_view[i]:
                    temp_min = -HH_view[i]
            
        if temp_max - ntol <= temp_min:
            condition = False

        it+= 1


    return alphas,it

