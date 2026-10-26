"""
Run DSRT prototype-selection experiments on tabular regression data.

This script evaluates Random Prototype Selection (RPS) and k-Medoids
Prototype Selection (kMPS), as described in Section 4.2 of the paper,
following the cross-validation and multi-start protocol of Section 5.1.
It records prediction errors, training times, and branch-node sparsity
for the experiments in Section 5.2 (Figures 4 and 5).

To choose the prototype selection strategy, set `type_nystrom` below to:
- "uniform" for Random Prototype Selection (RPS);
- "kmediod" with `type_med=0` for k-Medoids Prototype Selection (kMPS);
- None to disable prototype selection (rho=100% baseline).

The current settings use the elevators dataset, a 15% prototype rate,
and k-medoids selection. The third command-line argument (`type_med`)
is required by the script but is only used for k-medoids selection.
"""
import os
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gower
import numpy as np
import pandas as pd
import sys
import time
from sklearn.metrics.pairwise import rbf_kernel
from dsrt import *
from util_gower import *

# Experiment settings: tree depth, optimization, and prototype selection

parallel= True # Select serial (False) or Numba-parallel (True) BN-step objectives and gradients; see utils_numba_l2_nystrom.py.
number_seeds= 10
depth = 3
seed = 5
C = 1000.0
val = 70.0
# Command-line arguments: solver, clustering-based initialization
# (0=disabled, as in the paper; 1=enabled, using k-medoids), and k-medoids prototype selection rule.
clust_init = bool(int(sys.argv[2]))#True
type_med = int(sys.argv[3])
type_nystrom = "kmediod" #"uniform", None


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

df_path = Path(os.environ.get("DSRT_FUNCTIONAL_DATA_DIR", Path(__file__).resolve().parent / "data" / "tabular"))
list_dataset = ['yacht','auto-mpg','airfoil','housing','ailerons','kin8nm','abalone','elevators']

kkk = 5 # number of folds


for data_name in list_dataset:
    
    for perc in [15]:# in the paper we also consider 5,10 percentage    
    
        if data_name == 'housing':

            data = pd.read_csv(df_path / 'housing.csv',sep=';')
        elif data_name == 'ailerons':
            data = pd.read_csv(df_path / 'ailerons.data',sep=',',header=None)
        elif data_name == 'kin8nm':
            kin = fetch_openml(data_id = 189, as_frame = True)
            X = kin.data
            y = kin.target
            data = pd.concat([X,y],axis=1)
        elif data_name == 'elevators':
            el = fetch_openml(name='elevators', version=1, as_frame=True)
            X = el.data
            y = el.target
            data = pd.concat([X,y],axis=1)
        elif data_name == 'abalone':
            
            data = pd.read_csv(df_path / 'abalone.csv',sep=';')

        elif data_name == 'airfoil':
            data = pd.read_csv('airfoil_self_noise.dat',sep='\t',header=None)
        elif data_name == 'auto-mpg':
            column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight","acceleration", "model_year", "origin", "car_name"]
            data = pd.read_csv(df_path / 'auto-mpg.data',delim_whitespace=True, names=column_names, na_values='?')
        
            # Remove the non-numeric 'car_name' column
            data.drop(columns=["car_name"], inplace=True)
            # Impute missing horsepower values with the column mean
            data["horsepower"].fillna(data["horsepower"].mean(), inplace=True)
            # Move the regression target to the last column
            mpg = data.pop("mpg")
            data["mpg"] = mpg
        elif data_name == 'yacht':
            data = pd.read_csv(df_path / 'yacht_hydrodynamics.data', delim_whitespace=True, header=None)
        else:
            print('dataset not found')
            
        X = np.asarray(data.iloc[:,:-1].values)
        Y = np.asarray(data.iloc[:,-1].values)

        # Reproducible five-fold cross-validation (Section 5.1)
        kf = KFold(n_splits=kkk, shuffle=True, random_state=42)
        
        
        # Reuse the same ten initial seeds across folds
        np.random.seed(seed)
        random_seeds = np.random.randint(low=0, high=100, size=number_seeds)
        
        
        
        
        # Metrics for constant-leaf trees
        accuracy_simple = []
        accuracy_train_simple = []
        mse_simple = []
        mse_train_simple = []
        
        time_simple = []
        sparsity_level_simple = []
        
        # Metrics for linear-regression-leaf trees
        accuracy_linreg = []
        accuracy_train_linreg = []
        mse_linreg = []
        mse_train_linreg = []
        
        time_linreg = []
        sparsity_level_linreg = []
        
        kk = 1
        # Train and evaluate separately on each fold
        for train_idx, test_idx in kf.split(X):
            # Partition observations into training and test sets
            
            X_train = X[train_idx]
            X_test = X[test_idx]
            
            # Compute Gower distances using only training-fold feature ranges
            if data_name == 'abalone':
                cat_mask = [False,False,False,False,False,False,False,True,True]
            else:
                cat_mask = [False for i in range(X_train.shape[1])]
            cat = np.asarray(cat_mask, dtype=bool)
            num = ~cat
            min_ = np.zeros(X_train.shape[1], dtype=float)
            range_ = np.ones(X_train.shape[1], dtype=float)
            if num.any():
                train_num = X_train[:, num].astype(float)
                min_[num] = np.nanmin(train_num, axis=0)
                range_[num] = np.nanmax(train_num, axis=0) - np.nanmin(train_num, axis=0)
            
            # Assemble training and test distance matrices before feature scaling
            # The transpose gives the test-by-training layout expected by predict()
            sim_train = gower_distance_fixed_blockwise(X_train, X_train, cat_mask, min_, range_, block_size=256, dtype=np.float32).astype(np.float64)
            sim_test = gower_distance_fixed_blockwise(X_train, X_test, cat_mask, min_, range_, block_size=256, dtype=np.float32).astype(np.float64).T
            

            # Scale features for linear-regression leaves using training data only.
            scaler_min = MinMaxScaler()
            X_train = scaler_min.fit_transform(X_train)
            

            
            X_test = scaler_min.transform(X_test)
            
            y_train = Y[train_idx]
            y_test = Y[test_idx]
            
            # Standardize target values using training-fold statistics
            # Reported RMSE values therefore refer to standardized targets
            scaler = StandardScaler()
            y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
            y_test = scaler.transform(y_test.reshape(-1,1)).flatten()
            
            # Ten single-start runs per fold; both leaf models use each seed
            # Measured training times cover my_logistic_dec() only
            it = 1
            for sgl_seed in random_seeds:
                
                # Constant-leaf DSRT: train and evaluate
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
            
                # Training-set predictions and scores
                y_pred_train = model.predict(sim_train,np.array([]))
                temp_acc_train = r2_score(y_train,y_pred_train)
                accuracy_train_simple.append(temp_acc_train)
                mse_train_simple.append(np.sqrt(mean_squared_error(y_train,y_pred_train)))
                
                # Test-set predictions and scores
                y_pred = model.predict(sim_test,np.array([]))
                temp_acc = r2_score(y_test,y_pred)
                mse_simple.append(np.sqrt(mean_squared_error(y_test,y_pred)))
                
                print('Dataset:',data_name)
                print("Perc:",perc)
                print('Run:',(kk-1)*number_seeds+it)
                print('Accuracy simple:',temp_acc)
                accuracy_simple.append(temp_acc)
                
                
                # Release the constant-leaf model before the next fit
                del model
                # Linear-regression-leaf DSRT: same fold, seed, and prototype setting
                
                yyy = y_train[:,np.newaxis]
                model_linreg = SSRT(depth)
                model_linreg.set_dataset(sim_train,np.append(X_train,yyy,axis=1),type_reg = 'linear_reg')
                model_linreg.set_param_KLR(C=C,nu=1e-6,value=val,seed = sgl_seed) #C=1000.0,nu=1e-6
                model_linreg.set_nystrom(typ=type_nystrom,percentage=perc/100,ridge_leverage_lambda = 1,k_med_type = type_med)
                model_linreg.starting_point(sgl_seed)
                
                print('Training linear regression:')
                beg = time.time()
                model_linreg.my_logistic_dec(n_iter = 10, mini_iter = 1000,seed=sgl_seed,eps1= 0.3, eps2=0.1,eps3= 0.4,decrement_rate = 0.9,init_flag = clust_init,num_clust=100,gower=False,parallel=parallel,solver=solver)
                end = time.time()
                time_linreg.append(end-beg)
            
                sparsity_level_linreg.append(model_linreg.sparsity_level())
            
                # Training-set predictions and scores
                y_pred_train = model_linreg.predict(sim_train,X_train)
                temp_acc_train = r2_score(y_train,y_pred_train)
                accuracy_train_linreg.append(temp_acc_train)
                mse_train_linreg.append(np.sqrt(mean_squared_error(y_train,y_pred_train)))
                
                # Test-set predictions and scores
                y_pred = model_linreg.predict(sim_test,X_test)
                temp_acc = r2_score(y_test,y_pred)
                
                print('Dataset:',data_name)
                print("Perc:",perc)
                print('Run:',(kk-1)*number_seeds+it)
                print('Accuracy linear regression:',temp_acc)
                accuracy_linreg.append(temp_acc)
                mse_linreg.append(np.sqrt(mean_squared_error(y_test,y_pred)))
                
                del model_linreg
                
                it+=1
            kk+=1
        
        
        
        # 1. Crea il DataFrame
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
        
        # Append means and standard deviations over the 50 runs
        mean_simple = final_df_simple.mean(numeric_only=True)
        std_simple = final_df_simple.std(numeric_only=True)
        mean_linreg = final_df_linreg.mean(numeric_only=True)
        std_linreg= final_df_linreg.std(numeric_only=True)
        
        final_df_simple.loc[final_df_simple.shape[0]] = mean_simple
        final_df_simple.loc[final_df_simple.shape[0]] = std_simple
        final_df_linreg.loc[final_df_linreg.shape[0]] = mean_linreg
        final_df_linreg.loc[final_df_linreg.shape[0]] = std_linreg
        
        
        # Select the best constant-leaf run in each fold
        # Multi-start (Section 5.1): within each fold, select the run
        # with the lowest training RMSE, separately for both leaf models
        simple_best = []
        
        for k in range(kkk):
            temp_df = final_df_simple.iloc[k*number_seeds:(k+1)*number_seeds,:]
            riga_minima = temp_df.loc[temp_df['RMSE train simple'].idxmin()].values
            simple_best.append(riga_minima)
        
        best_simple_df = pd.DataFrame(simple_best)
        best_simple_df.columns = final_df_simple.columns
        
        lr_best = []
        
        # Select the best linear-regression-leaf run in each fold
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
        
        
        # Save single-start results and fold-wise multi-start selections
        # Both CSV files include summary rows at the end
        final_merged = pd.concat([final_df_simple, final_df_linreg], axis=1)
        best_merged = pd.concat([best_simple_df, best_lr_df], axis=1)
        
        # Preserve the separate output naming for the 100% (no-selection) baseline
        if type_nystrom == None:
            final_merged.to_csv(data_name+'depth_'+str(depth)+'_solver_'+solver+'init_'+str(clust_init)+'_nystrom_None.csv',sep=';', index=False)
            best_merged.to_csv(data_name+'depth_'+str(depth)+'_solver_'+solver+'init_'+str(clust_init)+'_nystrom_None_best_over_'+str(number_seeds)+'.csv',sep=';', index=False)
        else:
            final_merged.to_csv(data_name+'depth_'+str(depth)+'_solver_'+solver+'init_'+str(clust_init)+'_nystrom_'+type_nystrom+'_type_mediods_'+str(type_med)+'perc_'+str(perc)+'.csv',sep=';', index=False)
            best_merged.to_csv(data_name+'depth_'+str(depth)+'_solver_'+solver+'init_'+str(clust_init)+'_nystrom_'+type_nystrom+'_type_mediods_'+str(type_med)+'perc_'+str(perc)+'_best_over'+str(number_seeds)+'.csv',sep=';', index=False)
