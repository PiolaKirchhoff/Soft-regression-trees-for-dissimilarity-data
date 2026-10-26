# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:20:11 2022

@author: 
"""
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
cimport scipy.special.cython_special
import numpy as np
cimport numpy as np
cdef extern from "math.h":
    double log(double x) nogil  
np.import_array()
from cython cimport cdivision, boundscheck
#cython: language_level=3
ctypedef np.int64_t DTYPE_tt
ctypedef np.float64_t DTYPE_t

@cdivision
@boundscheck(False)
def init_H(np.ndarray[DTYPE_t, ndim=1] alpha,np.ndarray[DTYPE_t, ndim=1] y,np.ndarray[DTYPE_t, ndim=2] K,unsigned long int l,double C,double ni):
    cdef int i
    cdef double[::1] H_view = np.zeros(l)
    cdef double[::1] alp_view = alpha
    cdef double[::1] y_view = y
    
    for i in range(l):
        H_view[i] = K[i,:].dot(np.multiply(alpha,y)) + y_view[i]*log(alp_view[i]/(C-alp_view[i])) -y_view[i]*ni
    return H_view

@cdivision
@boundscheck(False)
def init_H_weight(np.ndarray[DTYPE_t, ndim=1] alpha,np.ndarray[DTYPE_t, ndim=1] y,np.ndarray[DTYPE_t, ndim=2] K,double[::1] weight_class,unsigned long int l,double C,double ni):
    cdef int i
    cdef double[::1] H_view = np.zeros(l)
    cdef double[::1] alp_view = alpha
    cdef double[::1] y_view = y
    
    for i in range(l):
        H_view[i] = K[i,:].dot(np.multiply(alpha,y)) + weight_class[i]*y_view[i]*log(alp_view[i]/(C-alp_view[i])) -y_view[i]*ni
    return H_view

@cdivision
@boundscheck(False)
def first_Iup(np.ndarray[DTYPE_t, ndim=1] alphas,np.ndarray[DTYPE_t, ndim=1] y,double th_up,double th_low,unsigned long int le):
    
    cdef int i
    cdef DTYPE_tt[::1] l_view = np.zeros(le,dtype=np.int64) #memoryview
    
    cdef double[::1] alphas_view = alphas
    cdef double[::1] y_view = y
    
    for i in range(le):
        if y_view[i] == 1:
            if alphas_view[i] < th_up:
                l_view[i] = 1
        else:
            if alphas_view[i] > th_low:
                l_view[i] = 1
    return l_view
@cdivision
@boundscheck(False)
def first_Ilow(np.ndarray[DTYPE_t, ndim=1] alphas,np.ndarray[DTYPE_t, ndim=1] y,double th_up,double th_low,unsigned long int le):
    
    cdef int i
    cdef DTYPE_tt[::1] l_view = np.zeros(le,dtype=np.int64) #memoryview
    
    cdef double[::1] alphas_view = alphas
    cdef double[::1] y_view = y
    
    for i in range(le):
        if y_view[i] == -1:
            if alphas_view[i] < th_up:
                l_view[i] = 1
        else:
            if alphas_view[i] > th_low:
                l_view[i] = 1
    return l_view

@cdivision
@boundscheck(False)
def grad(unsigned long int i_up,unsigned long int i_down,np.ndarray alphas,np.ndarray y,np.ndarray K, double C,double ni,unsigned long int le):
    
    cdef int i
    cdef double[:] alphas_view = alphas
    cdef double[:] y_view = y
    cdef double[:,:] K_view = K
    cdef double H_up
    cdef double H_down
    
    H_up =  y_view[i_up]*log((alphas_view[i_up])/(C-(alphas_view[i_up])))  - ni*y_view[i_up]
    H_down = y_view[i_down]*log((alphas_view[i_down])/(C-(alphas_view[i_down]))) - ni*y_view[i_down]
    for i in range(le):
        H_up += K_view[i_up,i]*y_view[i]*(alphas_view[i])
        H_down += K_view[i_down,i]*y[i]*(alphas_view[i])

    return H_up - H_down

@cdivision(True)
@boundscheck(False)
def get_j(double[::1] alphas_view,double[::1] y_view,int[::1] I_low_view,double[::1] HH_view,double grad_i,unsigned long int i_up,double C,double[:,::1] K_view,double tau,double ni,unsigned long int le):
    
    cdef int i
    cdef int j
    cdef double temp_val = 1.0
    
    cdef double grad = 0.0
    cdef double a = 0.0
    cdef double vall 

    cdef int index = -1

    cdef double H_up
    cdef double H_down
    
    
    for i in range(le):
        
        if I_low_view[i] == 1:
            
            if -HH_view[i] < -grad_i:
                # Hessian calculation
                a = K_view[i_up,i_up] + K_view[i,i] - 2*K_view[i_up,i] + (C)/((alphas_view[i_up])*(C-(alphas_view[i_up]))) + (C)/((alphas_view[i])*(C-(alphas_view[i])))
                
                # Gradient computation
                
                H_up =  y_view[i_up]*log((alphas_view[i_up])/(C-(alphas_view[i_up])))  - ni*y_view[i_up]
                H_down = y_view[i]*log((alphas_view[i])/(C-(alphas_view[i]))) - ni*y_view[i]
                for j in range(le):
                    H_up += K_view[i_up,j]*y_view[j]*(alphas_view[j])
                    H_down += K_view[i,j]*y_view[j]*(alphas_view[j])
                grad = H_up - H_down
                if a > 0:
                    vall = - (grad*grad)/a
                else:
                    vall = - (grad*grad)/tau
                
                if vall < temp_val:
                    index = i
                    temp_val = vall
    return index

@cdivision(True)
@boundscheck(False)
cdef int get_j_new(double[::1] alphas_view,double[::1] y_view,int[::1] I_low_view,double[::1] HH_view,double grad_i,unsigned long int i_up,double C,double[:,::1] K_view,double tau,double ni,unsigned long int le):
    
    cdef int i
    cdef int j
    cdef double temp_val = 1.0
    
    cdef double grad = 0.0
    cdef double a = 0.0
    cdef double vall 

    cdef int index = -1

    cdef double H_up
    cdef double H_down
    
    
    for i in range(le):
        
        if I_low_view[i] == 1:
            
            if -HH_view[i] < -grad_i:
                # Hessian calculation
                a = K_view[i_up,i_up] + K_view[i,i] - 2*K_view[i_up,i] + (C)/((alphas_view[i_up])*(C-(alphas_view[i_up]))) + (C)/((alphas_view[i])*(C-(alphas_view[i])))
                
                # Gradient computation
                
                H_up =  y_view[i_up]*log((alphas_view[i_up])/(C-(alphas_view[i_up])))  - ni*y_view[i_up]
                H_down = y_view[i]*log((alphas_view[i])/(C-(alphas_view[i]))) - ni*y_view[i]
                for j in range(le):
                    H_up += K_view[i_up,j]*y_view[j]*(alphas_view[j])
                    H_down += K_view[i,j]*y_view[j]*(alphas_view[j])
                grad = H_up - H_down
                if a > 0:
                    vall = - (grad*grad)/a
                else:
                    vall = - (grad*grad)/tau
                
                if vall < temp_val:
                    index = i
                    temp_val = vall
    return index
