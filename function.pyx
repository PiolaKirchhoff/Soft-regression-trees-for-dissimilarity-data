# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 16:37:07 2022

@author: 
"""
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
cimport scipy.special.cython_special
import numpy as np
cimport numpy as np
from cython cimport cdivision, boundscheck,wraparound
np.import_array()
ctypedef np.int64_t DTYPE_tt
ctypedef np.float64_t DTYPE_t
cdef extern from "math.h":
    double log(double x) nogil  


@cdivision(True)
@boundscheck(False)
@wraparound(False)
def alt_objective_sc(double t,np.intp_t i_up,np.intp_t i_down,np.ndarray[DTYPE_t, ndim=1, mode='c'] alpha,np.ndarray[DTYPE_t, ndim=1, mode='c'] y,np.ndarray[DTYPE_t, ndim=2, mode='c'] K,double C,double ni,unsigned long int le):
    
    cdef np.intp_t i
    cdef double term = 0.0
    temp_alp = alpha.copy()
    cdef double[:] y_view = y
    
    for i in range(le):
        
        if i == i_up:
            term += (alpha[i]+(t/y[i]))*(log((alpha[i]+(t/y[i]))/C)) + (C-(alpha[i]+(t/y[i])))*log(1-((alpha[i]+(t/y[i]))/C)) - ni*(alpha[i]+(t/y[i]))
        elif i == i_down:
            term += (alpha[i]-(t/y[i]))*(log((alpha[i]-(t/y[i]))/C)) + (C-(alpha[i]-(t/y[i])))*log(1-((alpha[i]-(t/y[i]))/C)) -ni*(alpha[i]-(t/y[i]))
        else:
            term += alpha[i]*(log(alpha[i]/C)) + (C-alpha[i])*log(1-((alpha[i])/C)) -ni*alpha[i]
            
    temp_alp[i_up] += (t/y_view[i_up])
    temp_alp[i_down] += -(t/y_view[i_down])
    
    tay = np.multiply(temp_alp,y)
    
    return term + tay.dot(K.dot(tay))/2

@cdivision(True)
@boundscheck(False)
@wraparound(False)
def alt_objective_sc_new(double t,np.intp_t i_up,np.intp_t i_down,double[::1] alpha,double[::1] y,double[:,::1] K_view,double C,double ni,unsigned long int le):
    
    cdef np.intp_t i
    cdef double term = 0.0
    cdef double term_up = (alpha[i_up]+(t/y[i_up]))*(alpha[i_up]+(t/y[i_up]))*K_view[i_up,i_up]+K_view[i_up,i_down]*y[i_up]*y[i_down]*(alpha[i_up]+(t/y[i_up]))*(alpha[i_down]-(t/y[i_down]))
    cdef double term_down = (alpha[i_down]-(t/y[i_down]))*(alpha[i_down]-(t/y[i_down]))*K_view[i_down,i_down]+K_view[i_up,i_down]*y[i_up]*y[i_down]*(alpha[i_up]+(t/y[i_up]))*(alpha[i_down]-(t/y[i_down]))
    
    for i in range(le):
        
        if i == i_up:
            term += (alpha[i]+(t/y[i]))*(log((alpha[i]+(t/y[i]))/C)) + (C-(alpha[i]+(t/y[i])))*log(1-((alpha[i]+(t/y[i]))/C)) - ni*(alpha[i]+(t/y[i]))
        elif i == i_down:
            term += (alpha[i]-(t/y[i]))*(log((alpha[i]-(t/y[i]))/C)) + (C-(alpha[i]-(t/y[i])))*log(1-((alpha[i]-(t/y[i]))/C)) -ni*(alpha[i]-(t/y[i]))
        else:
            term += alpha[i]*(log(alpha[i]/C)) + (C-alpha[i])*log(1-((alpha[i])/C)) -ni*alpha[i]# TODO: remove it
            term_up += 2*(y[i_up]*y[i]*(alpha[i_up]+(t/y[i_up]))*alpha[i]*K_view[i_up,i])
            term_down += 2*(y[i_down]*y[i]*(alpha[i_down]-(t/y[i_down]))*alpha[i]*K_view[i_down,i])
            
    return term + (term_up+term_down)/2

@cdivision(True)
@boundscheck(False)
@wraparound(False)
def alt_objective_sc_new_weight(double t,np.intp_t i_up,np.intp_t i_down,double[::1] alpha,double[::1] y,double[:,::1] K_view,double C,double ni,unsigned long int le,double wei_iup,double wei_ilow):
    
    cdef np.intp_t i
    cdef double term = 0.0
    cdef double term_up = (alpha[i_up]+(t/y[i_up]))*(alpha[i_up]+(t/y[i_up]))*K_view[i_up,i_up]+K_view[i_up,i_down]*y[i_up]*y[i_down]*(alpha[i_up]+(t/y[i_up]))*(alpha[i_down]-(t/y[i_down]))
    cdef double term_down = (alpha[i_down]-(t/y[i_down]))*(alpha[i_down]-(t/y[i_down]))*K_view[i_down,i_down]+K_view[i_up,i_down]*y[i_up]*y[i_down]*(alpha[i_up]+(t/y[i_up]))*(alpha[i_down]-(t/y[i_down]))
    
    for i in range(le):
        
        if i == i_up:
            term += wei_iup*(alpha[i]+(t/y[i]))*(log((alpha[i]+(t/y[i]))/C)) + wei_iup*(C-(alpha[i]+(t/y[i])))*log(1-((alpha[i]+(t/y[i]))/C)) - ni*(alpha[i]+(t/y[i]))
        elif i == i_down:
            term += wei_ilow*(alpha[i]-(t/y[i]))*(log((alpha[i]-(t/y[i]))/C)) + wei_ilow*(C-(alpha[i]-(t/y[i])))*log(1-((alpha[i]-(t/y[i]))/C)) -ni*(alpha[i]-(t/y[i]))
        else:
            #term += alpha[i]*(log(alpha[i]/C)) + (C-alpha[i])*log(1-((alpha[i])/C)) -ni*alpha[i]# TODO: remove it
            term_up += 2*(y[i_up]*y[i]*(alpha[i_up]+(t/y[i_up]))*alpha[i]*K_view[i_up,i])
            term_down += 2*(y[i_down]*y[i]*(alpha[i_down]-(t/y[i_down]))*alpha[i]*K_view[i_down,i])
            
    return term + (term_up+term_down)/2

def alt_objective_sc_new_debug(double t,np.intp_t i_up,np.intp_t i_down,double[::1] alpha,double[::1] y,double[:,::1] K_view,double C,double ni,unsigned long int le):
    
    cdef np.intp_t i
    cdef double term = 0.0
    cdef double term_up = (alpha[i_up]+(t/y[i_up]))*(alpha[i_up]+(t/y[i_up]))*K_view[i_up,i_up]+K_view[i_up,i_down]*y[i_up]*y[i_down]*(alpha[i_up]+(t/y[i_up]))*(alpha[i_down]-(t/y[i_down]))
    cdef double term_down = (alpha[i_down]-(t/y[i_down]))*(alpha[i_down]-(t/y[i_down]))*K_view[i_down,i_down]+K_view[i_up,i_down]*y[i_up]*y[i_down]*(alpha[i_up]+(t/y[i_up]))*(alpha[i_down]-(t/y[i_down]))
    
    for i in range(le):
        
        if i == i_up:
            term += (alpha[i]+(t/y[i]))*(log((alpha[i]+(t/y[i]))/C)) + (C-(alpha[i]+(t/y[i])))*log(1-((alpha[i]+(t/y[i]))/C)) - ni*(alpha[i]+(t/y[i]))
        elif i == i_down:
            term += (alpha[i]-(t/y[i]))*(log((alpha[i]-(t/y[i]))/C)) + (C-(alpha[i]-(t/y[i])))*log(1-((alpha[i]-(t/y[i]))/C)) -ni*(alpha[i]-(t/y[i]))
        else:
            term += alpha[i]*(log(alpha[i]/C)) + (C-alpha[i])*log(1-((alpha[i])/C)) -ni*alpha[i]
            term_up += 2*(y[i_up]*y[i]*(alpha[i_up]+(t/y[i_up]))*alpha[i]*K_view[i_up,i])
            term_down += 2*(y[i_down]*y[i]*(alpha[i_down]-(t/y[i_down]))*alpha[i]*K_view[i_down,i])
            
    return term + (term_up+term_down)/2
@cdivision(True)
@boundscheck(False)
@wraparound(False)
def line_search_grad_sc(double t,np.intp_t i_up,np.intp_t i_down,double[::1] alp_view,double[::1] y_view,double[:,::1] K_view,double C,double ni,unsigned long int le):
    
    

    cdef double H_up
    cdef double H_down
    cdef np.intp_t i
    
    H_up =  y_view[i_up]*log( ((alp_view[i_up] + (t/y_view[i_up]))/(C-(alp_view[i_up] + (t/y_view[i_up]))))) - ni*y_view[i_up]
    H_down = y_view[i_down]*log(((alp_view[i_down] - (t/y_view[i_down]))/(C-(alp_view[i_down] - (t/y_view[i_down]))))) - ni*y_view[i_down]
    for i in range(le):
        
        if i == i_up:
            H_up += K_view[i_up,i]*y_view[i]*(alp_view[i] + (t/y_view[i]))
            H_down += K_view[i_down,i]*y_view[i]*(alp_view[i] + (t/y_view[i]))
        elif i == i_down:
            H_up += K_view[i_up,i]*y_view[i]*(alp_view[i] - (t/y_view[i]))
            H_down += K_view[i_down,i]*y_view[i]*(alp_view[i] - (t/y_view[i]))
        else:
            H_up += K_view[i_up,i]*y_view[i]*alp_view[i]
            H_down += K_view[i_down,i]*y_view[i]*alp_view[i]
        
    return H_up - H_down


@cdivision(True)
@boundscheck(False)
@wraparound(False)
def line_search_grad_sc_weight(double t,np.intp_t i_up,np.intp_t i_down,double[::1] alp_view,double[::1] y_view,double[:,::1] K_view,double C,double ni,unsigned long int le,double wei_iup,double wei_ilow):
    
    

    cdef double H_up
    cdef double H_down
    cdef np.intp_t i
    
    H_up =  wei_iup*y_view[i_up]*log( ((alp_view[i_up] + (t/y_view[i_up]))/(C-(alp_view[i_up] + (t/y_view[i_up]))))) - ni*y_view[i_up]
    H_down = wei_ilow*y_view[i_down]*log(((alp_view[i_down] - (t/y_view[i_down]))/(C-(alp_view[i_down] - (t/y_view[i_down]))))) - ni*y_view[i_down]
    for i in range(le):
        
        if i == i_up:
            H_up += K_view[i_up,i]*y_view[i]*(alp_view[i] + (t/y_view[i]))
            H_down += K_view[i_down,i]*y_view[i]*(alp_view[i] + (t/y_view[i]))
        elif i == i_down:
            H_up += K_view[i_up,i]*y_view[i]*(alp_view[i] - (t/y_view[i]))
            H_down += K_view[i_down,i]*y_view[i]*(alp_view[i] - (t/y_view[i]))
        else:
            H_up += K_view[i_up,i]*y_view[i]*alp_view[i]
            H_down += K_view[i_down,i]*y_view[i]*alp_view[i]
        
    return H_up - H_down

def line_search_grad_sc_debug(double t,np.intp_t i_up,np.intp_t i_down,double[::1] alp_view,double[::1] y_view,double[:,::1] K_view,double C,double ni,unsigned long int le):
    
    

    cdef double H_up
    cdef double H_down
    cdef np.intp_t i
    
    H_up =  y_view[i_up]*log( ((alp_view[i_up] + (t/y_view[i_up]))/(C-(alp_view[i_up] + (t/y_view[i_up]))))) - ni*y_view[i_up]
    H_down = y_view[i_down]*log(((alp_view[i_down] - (t/y_view[i_down]))/(C-(alp_view[i_down] - (t/y_view[i_down]))))) - ni*y_view[i_down]
    for i in range(le):
        
        if i == i_up:
            H_up += K_view[i_up,i]*y_view[i]*(alp_view[i] + (t/y_view[i]))
            H_down += K_view[i_down,i]*y_view[i]*(alp_view[i] + (t/y_view[i]))
        elif i == i_down:
            H_up += K_view[i_up,i]*y_view[i]*(alp_view[i] - (t/y_view[i]))
            H_down += K_view[i_down,i]*y_view[i]*(alp_view[i] - (t/y_view[i]))
        else:
            H_up += K_view[i_up,i]*y_view[i]*alp_view[i]
            H_down += K_view[i_down,i]*y_view[i]*alp_view[i]
        
    return H_up - H_down

@cdivision(True)
@boundscheck(False)
@wraparound(False)
def hess(double t,np.intp_t i_up,np.intp_t i_down,double[::1] alp_view,double[::1] y_view,double[:,::1] K_view,double C,double ni,unsigned long int le):
    
    return K_view[i_up,i_up] + K_view[i_down,i_down] - 2*K_view[i_up,i_down] + (C)/((alp_view[i_up]+t/y_view[i_up])*(C-(alp_view[i_up]+t/y_view[i_up]))) + (C)/((alp_view[i_down]-t/y_view[i_down])*(C-(alp_view[i_down]-t/y_view[i_down])))

@cdivision(True)
@boundscheck(False)
@wraparound(False)
def hess_weight(double t,np.intp_t i_up,np.intp_t i_down,double[::1] alp_view,double[::1] y_view,double[:,::1] K_view,double C,double ni,unsigned long int le,double wei_iup,double wei_ilow):
    
    return K_view[i_up,i_up] + K_view[i_down,i_down] - 2*K_view[i_up,i_down] + (wei_iup*C)/((alp_view[i_up]+t/y_view[i_up])*(C-(alp_view[i_up]+t/y_view[i_up]))) + (wei_ilow*C)/((alp_view[i_down]-t/y_view[i_down])*(C-(alp_view[i_down]-t/y_view[i_down])))


def hess_debug(double t,np.intp_t i_up,np.intp_t i_down,double[::1] alp_view,double[::1] y_view,double[:,::1] K_view,double C,double ni,unsigned long int le):
    
    return K_view[i_up,i_up] + K_view[i_down,i_down] - 2*K_view[i_up,i_down] + (C)/((alp_view[i_up]+t/y_view[i_up])*(C-(alp_view[i_up]+t/y_view[i_up]))) + (C)/((alp_view[i_down]-t/y_view[i_down])*(C-(alp_view[i_down]-t/y_view[i_down])))

@cdivision(True)
@boundscheck(False)
@wraparound(False)
def total_objective(np.ndarray[DTYPE_t, ndim=1] alpha,np.ndarray[DTYPE_t, ndim=1] yy,np.ndarray[DTYPE_t, ndim=2] K,double C,double ni, unsigned long int le):
    cdef int i 
    cdef double term = 0.0
    cdef double[:] alp_view = alpha
    ay = np.multiply(alpha,yy)
    
    for i in range(le):
        term += alp_view[i]*log(alp_view[i]/C)+(C-alp_view[i])*log(1-((alp_view[i])/C))
    return term + ay.dot(K.dot(ay))/2 -ni*alpha.sum()