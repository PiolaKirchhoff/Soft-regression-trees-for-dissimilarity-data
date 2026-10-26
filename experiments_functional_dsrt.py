# -*- coding: utf-8 -*-
"""Run DSRT regression experiments on functional datasets (Section 5.3).

This script trains Distance-based Soft Regression Trees (DSRTs) with
precomputed dissimilarity matrices. It uses five-fold cross-validation and
10 random initializations per fold, and evaluates both constant-leaf and
linear-regression-leaf models. Results include single-start metrics and
fold-wise multi-start selection based on the lowest training RMSE.

The experiments in Section 5.3 report constant-leaf results; the linear-leaf
results are considered in Appendix B. The current configuration uses the
energy dataset, depth 2, and no prototype selection (rho=100%).

Datasets are read from data/functional next to this script by default.
Set the DSRT_FUNCTIONAL_DATA_DIR environment variable to use another folder.
"""

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import time
from dsrt import *

# Select serial (False) or Numba-parallel (True) BN-step objectives
# and gradients; see utils_numba_l2_nystrom.py.
parallel= True
# Experimental configuration: depth 2 and ten starting points per fold.
number_seeds= 10
depth = 2
seed = 42
C = 1000.0
val = 70.0
# Command-line arguments: solver, clustering-based initialization
# (0=disabled, as in the paper; 1=enabled, using k-medoids), and k-medoids rule.
clust_init = bool(int(sys.argv[2]))
type_med = int(sys.argv[3])
type_nystrom = None  # No prototype selection (rho=100%).


# Map command-line solver names to the options used by DSRT.
if sys.argv[1] == 'lbfgs':
    solver = 'L-BFGS-B'
elif sys.argv[1] == 'tnc':
    solver = 'TNC'
elif sys.argv[1] == 'bfgs':
    solver = 'BFGS'
elif sys.argv[1] == 'n_cg':
    solver = 'Newton-CG'
elif sys.argv[1] == 'sqp':
    solver = 'SLSQP'
elif sys.argv[1] == 'ada_seq':
    solver = 'adam_seq'
elif sys.argv[1] == 'ada_vec':
    solver = 'adam_vec'
    
df_path = Path(os.environ.get("DSRT_FUNCTIONAL_DATA_DIR", Path(__file__).resolve().parent / "data" / "functional"))
# Active dataset
list_dataset =['energy','flood1','flood2','flood3','covid','iron','swissroll']


kkk = 5  # Number of folds


for data_name in list_dataset:

    # Load the precomputed dissimilarities, features, and target values.
    if data_name == 'flood2':
        sim_matrix = pd.read_csv(df_path / 'FloodModeling2_DTW_train_n466_flatten_fixed_length_D.csv',sep=',').values
        X = pd.read_csv(df_path / 'FloodModeling2_DTW_train_n466_flatten_fixed_length_X.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'FloodModeling2_DTW_train_n466_flatten_fixed_length_y.csv',sep=',').iloc[:,1:].values
    elif data_name == 'flood1':
        sim_matrix = pd.read_csv(df_path / 'FloodModeling1_DTW_train_n471_flatten_fixed_length_D.csv',sep=',').values
        X = pd.read_csv(df_path / 'FloodModeling1_DTW_train_n471_flatten_fixed_length_X.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'FloodModeling1_DTW_train_n471_flatten_fixed_length_y.csv',sep=',').iloc[:,1:].values
    elif data_name == 'flood3':
        sim_matrix = pd.read_csv(df_path / 'FloodModeling3_DTW_train_n429_flatten_fixed_length_D.csv',sep=',').values
        X = pd.read_csv(df_path / 'FloodModeling3_DTW_train_n429_flatten_fixed_length_X.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'FloodModeling3_DTW_train_n429_flatten_fixed_length_y.csv',sep=',').iloc[:,1:].values
    elif data_name == 'iron':
        sim_matrix = pd.read_csv(df_path / 'IronConcentration_DTW_train_n427_flatten_fixed_length_D.csv',sep=',').values
        X = pd.read_csv(df_path / 'IronConcentration_DTW_train_n427_flatten_fixed_length_X.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'IronConcentration_DTW_train_n427_flatten_fixed_length_y.csv',sep=',').iloc[:,1:].values
    elif data_name == 'covid':
        sim_matrix = pd.read_csv(df_path / 'Covid3Month_DTW_train_n140_flatten_fixed_length_D.csv',sep=',').values
        X = pd.read_csv(df_path / 'Covid3Month_DTW_train_n140_flatten_fixed_length_X.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'Covid3Month_DTW_train_n140_flatten_fixed_length_y.csv',sep=',').iloc[:,1:].values
    elif data_name == 'energy':
        sim_matrix = pd.read_csv(df_path / 'AppliancesEnergy_DTW_train_n95_flatten_fixed_length_D.csv',sep=',').values
        X = pd.read_csv(df_path / 'AppliancesEnergy_DTW_train_n95_flatten_fixed_length_X.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'AppliancesEnergy_DTW_train_n95_flatten_fixed_length_y.csv',sep=',').iloc[:,1:].values
    elif data_name == 'swissroll':
        
        sim_matrix = pd.read_csv(df_path / 'swissroll_isomap_D_new.csv',sep=',').values
        X = pd.read_csv(df_path / 'swissroll_isomap_X_new.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'swissroll_isomap_y_new.csv',sep=',').iloc[:,1].values  # Swiss roll target is read as one column
    else:
        print('dataset not found')
        
    
    
    
    
    # Five-fold cross-validation with fixed folds (Section 5.1).
    kf = KFold(n_splits=kkk, shuffle=True, random_state=42)
    
    
    
    # Reuse the same ten initial seeds across folds.
    np.random.seed(seed)
    random_seeds = np.random.randint(low=0, high=100, size=number_seeds)
    
    
    
    
    # Metrics for constant-leaf trees.
    accuracy_simple = []
    accuracy_train_simple = []
    mse_simple = []
    mse_train_simple = []
    
    time_simple = []
    sparsity_level_simple = []
    
    # Metrics for linear-regression-leaf trees.
    accuracy_linreg = []
    accuracy_train_linreg = []
    mse_linreg = []
    mse_train_linreg = []
    
    time_linreg = []
    sparsity_level_linreg = []
    
    kk = 1
    # Train and evaluate separately on each fold.
    
    
    for train_idx, test_idx in kf.split(X):
        # Split the dataset into training and test folds.
        
        X_train = X[train_idx]
        X_test = X[test_idx]
        
        # Select rows and columns from the precomputed distance matrix.
        sim_train = sim_matrix[np.ix_(train_idx,train_idx)].astype(np.float64)
        sim_test = sim_matrix[np.ix_(test_idx,train_idx)].astype(np.float64)
        
        

        
        # Scale features for linear-regression leaves using training data only.
        scaler_min = MinMaxScaler()
        X_train = scaler_min.fit_transform(X_train)
        
        
        X_test = scaler_min.transform(X_test)
        
        y_train = Y[train_idx]
        y_test = Y[test_idx]
        
        # Standardize the target using training-fold statistics.
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
        y_test = scaler.transform(y_test.reshape(-1,1)).flatten()
        
        # Ten single-start runs per fold; fitting times cover my_logistic_dec() only.
        it = 1
        for sgl_seed in random_seeds:
        # Train both leaf formulations from the same run seed.
        
            # Constant-leaf DSRT: train and evaluate.
            model = SSRT(depth)
            model.set_dataset(sim_train,y_train,type_reg = 'simple')
            model.set_param_KLR(C=C,nu=1e-6,value=val,seed = sgl_seed) 
            model.set_nystrom(typ=type_nystrom,percentage=perc/100,ridge_leverage_lambda = 1,k_med_type = type_med)
            model.starting_point(sgl_seed)
            print('Parallel: ',parallel)
            print('Training simple:')
            beg = time.time()
            model.my_logistic_dec(n_iter = 10, mini_iter = 1000,seed=sgl_seed,eps1= 0.3, eps2=0.1,eps3= 0.4,decrement_rate = 0.9,init_flag = clust_init,num_clust= 100,gower=False,parallel=parallel,solver=solver)
            end = time.time()
            time_simple.append(end-beg)
        
            sparsity_level_simple.append(model.sparsity_level())
        
            # Training-set predictions and scores.
            y_pred_train = model.predict(sim_train,np.array([]))
            temp_acc_train = r2_score(y_train,y_pred_train)
            accuracy_train_simple.append(temp_acc_train)
            mse_train_simple.append(np.sqrt(mean_squared_error(y_train,y_pred_train)))
            
            # Test-set predictions and scores.
            y_pred = model.predict(sim_test,np.array([]))
            temp_acc = r2_score(y_test,y_pred)
            mse_simple.append(np.sqrt(mean_squared_error(y_test,y_pred)))
            
            print('Run:',(kk-1)*number_seeds+it)
            print('Accuracy simple:',temp_acc)
            accuracy_simple.append(temp_acc)
            
            
            # Release the constant-leaf model before the next fit.
            del model
            # Linear-regression-leaf DSRT: same fold and seed.
            
            yyy = y_train[:,np.newaxis]
            model_linreg = SSRT(depth)
            model_linreg.set_dataset(sim_train,np.append(X_train,yyy,axis=1),type_reg = 'linear_reg')
            model_linreg.set_param_KLR(C=C,nu=1e-6,value=val,seed = sgl_seed) 
            model_linreg.set_nystrom(typ=type_nystrom,percentage=perc/100,ridge_leverage_lambda = 1,k_med_type = type_med)
            model_linreg.starting_point(sgl_seed)
            
            print('Training linear regression:')
            beg = time.time()
            model_linreg.my_logistic_dec(n_iter = 10, mini_iter = 1000,seed=sgl_seed,eps1= 0.3, eps2=0.1,eps3= 0.4,decrement_rate = 0.9,init_flag = clust_init,num_clust=100,gower=False,parallel=parallel,solver=solver)
            end = time.time()
            time_linreg.append(end-beg)
        
            sparsity_level_linreg.append(model_linreg.sparsity_level())
        
            # Training-set predictions and scores.
            y_pred_train = model_linreg.predict(sim_train,X_train)
            temp_acc_train = r2_score(y_train,y_pred_train)
            accuracy_train_linreg.append(temp_acc_train)
            mse_train_linreg.append(np.sqrt(mean_squared_error(y_train,y_pred_train)))
            
            # Test-set predictions and scores.
            y_pred = model_linreg.predict(sim_test,X_test)
            temp_acc = r2_score(y_test,y_pred)
            print('Run:',(kk-1)*number_seeds+it)
            print('Accuracy linear regression:',temp_acc)
            accuracy_linreg.append(temp_acc)
            mse_linreg.append(np.sqrt(mean_squared_error(y_test,y_pred)))
            
            del model_linreg
            
            it+=1
        kk+=1
    
    
    
    # Assemble and summarize the 50 single-start results.
    df_simple = pd.DataFrame({
        "R square train simple": accuracy_train_simple,
        "R square test simple": accuracy_simple,
        "RMSE train simple": mse_train_simple,
        "RMSE test simple": mse_simple,
        "time simple": time_simple,
    })
    
    df_linreg = pd.DataFrame({
        "R square train linear reg": accuracy_train_linreg,
        "R square test linear reg": accuracy_linreg,
        "RMSE train linear reg": mse_train_linreg,
        "RMSE test linear reg": mse_linreg,
        "time linear reg": time_linreg
        
    })
    
    
    
    df_sparse_simple = pd.DataFrame(sparsity_level_simple, columns=['s_'+str(i) for i in range(2**depth - 1)])
    df_sparse_linreg = pd.DataFrame(sparsity_level_linreg, columns=['lr_'+str(i) for i in range(2**depth - 1)])
    
    final_df_simple = pd.concat([df_simple, df_sparse_simple], axis=1)
    final_df_linreg = pd.concat([df_linreg, df_sparse_linreg], axis=1)
    
    mean_simple = final_df_simple.mean(numeric_only=True)
    std_simple = final_df_simple.std(numeric_only=True)
    mean_linreg = final_df_linreg.mean(numeric_only=True)
    std_linreg= final_df_linreg.std(numeric_only=True)
    
    final_df_simple.loc[final_df_simple.shape[0]] = mean_simple
    final_df_simple.loc[final_df_simple.shape[0]] = std_simple
    final_df_linreg.loc[final_df_linreg.shape[0]] = mean_linreg
    final_df_linreg.loc[final_df_linreg.shape[0]] = std_linreg
    
    
    # Multi-start (Section 5.1): select the lowest training RMSE per fold
    # separately for constant and linear-regression leaves.
    simple_best = []
    
    for k in range(kkk):
        temp_df = final_df_simple.iloc[k*number_seeds:(k+1)*number_seeds,:]
        riga_minima = temp_df.loc[temp_df['RMSE train simple'].idxmin()].values
        simple_best.append(riga_minima)
    
    best_simple_df = pd.DataFrame(simple_best)
    best_simple_df.columns = final_df_simple.columns
    
    lr_best = []
    
    # Select the best linear-regression-leaf run in each fold.
    for k in range(kkk):
        temp_df = final_df_linreg.iloc[k*number_seeds:(k+1)*number_seeds,:]
        riga_minima = temp_df.loc[temp_df['RMSE train linear reg'].idxmin()].values
        lr_best.append(riga_minima)
    
    best_lr_df = pd.DataFrame(lr_best)
    best_lr_df.columns = final_df_linreg.columns
    
    
    mean_simple_best = best_simple_df.mean(numeric_only=True)
    std_simple_best = best_simple_df.std(numeric_only=True)
    mean_linreg_best = best_lr_df.mean(numeric_only=True)
    std_linreg_best = best_lr_df.std(numeric_only=True)
    
    best_simple_df.loc[best_simple_df.shape[0]] = mean_simple_best
    best_simple_df.loc[best_simple_df.shape[0]] = std_simple_best
    best_lr_df.loc[best_lr_df.shape[0]] = mean_linreg_best
    best_lr_df.loc[best_lr_df.shape[0]] = std_linreg_best
    
    
    
    # Save single-start and fold-wise multi-start results as separate CSV files.
    final_merged = pd.concat([final_df_simple, final_df_linreg], axis=1)
    best_merged = pd.concat([best_simple_df, best_lr_df], axis=1)
    
    if type_nystrom == None:
        final_merged.to_csv(data_name+'depth_'+str(depth)+'_solver_'+solver+'init_'+str(clust_init)+'_nystrom_None.csv',sep=';', index=False)
        best_merged.to_csv(data_name+'depth_'+str(depth)+'_solver_'+solver+'init_'+str(clust_init)+'_nystrom_None_best_over_'+str(number_seeds)+'.csv',sep=';', index=False)
    else:
        final_merged.to_csv(data_name+'depth_'+str(depth)+'_solver_'+solver+'init_'+str(clust_init)+'_nystrom_'+type_nystrom+'_type_mediods_'+str(type_med)+'perc_'+str(perc)+'.csv',sep=';', index=False)
        best_merged.to_csv(data_name+'depth_'+str(depth)+'_solver_'+solver+'init_'+str(clust_init)+'_nystrom_'+type_nystrom+'_type_mediods_'+str(type_med)+'perc_'+str(perc)+'_best_over'+str(number_seeds)+'.csv',sep=';', index=False)
