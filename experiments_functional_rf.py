# -*- coding: utf-8 -*-
"""Run Random Forest regression experiments on functional datasets (Section 5.3).

This script evaluates the Random Forest benchmark implemented with the R
randomForest package through rpy2. It uses five-fold cross-validation and
10 random seeds per fold, and records prediction errors and training times.

Input features are loaded from the functional dataset files; the distance
matrices used by DSRT are not required by the Random Forest benchmark.

Datasets are read from data/functional next to this script by default.
Set the DSRT_FUNCTIONAL_DATA_DIR environment variable to use another folder.
R and the R package randomForest must be installed and discoverable by rpy2.
"""

import os
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import time


# Configure R_HOME externally only if rpy2 cannot locate the R installation. 
#To set R_HOME to the directory where R is installed on your system simply write something like os.environ["R_HOME"] = r"C:\Program Files\R\R-4.2.1"
from rpy2 import robjects
from rpy2.robjects import numpy2ri,r
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
# Enable automatic NumPy-to-R conversion.
numpy2ri.activate()
# Load the randomForest implementation from R.
randomForest = importr('randomForest')

# Random Forest configuration: 500 trees and ten R seeds per fold.
n_estimators = 500
number_seeds= 10
seed = 56
    
df_path = Path(os.environ.get("DSRT_FUNCTIONAL_DATA_DIR", Path(__file__).resolve().parent / "data" / "functional"))
# Active dataset
list_dataset = ['energy','flood1','flood2','flood3','covid','iron','swissroll']
kkk = 5  # Number of folds


for data_name in list_dataset:

    # Load features and regression targets for the selected dataset.
    if data_name == 'covid':
        X = pd.read_csv(df_path / 'Covid3Month_DTW_train_n140_flatten_fixed_length_X.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'Covid3Month_DTW_train_n140_flatten_fixed_length_y.csv',sep=',').iloc[:,1:].values
    elif data_name == 'flood2':
        X = pd.read_csv(df_path / 'FloodModeling2_DTW_train_n466_flatten_fixed_length_X.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'FloodModeling2_DTW_train_n466_flatten_fixed_length_y.csv',sep=',').iloc[:,1:].values
    elif data_name == 'iron':
        X = pd.read_csv(df_path / 'IronConcentration_DTW_train_n427_flatten_fixed_length_X.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'IronConcentration_DTW_train_n427_flatten_fixed_length_y.csv',sep=',').iloc[:,1:].values
    elif data_name == 'flood3':
        X = pd.read_csv(df_path / 'FloodModeling3_DTW_train_n429_flatten_fixed_length_X.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'FloodModeling3_DTW_train_n429_flatten_fixed_length_y.csv',sep=',').iloc[:,1:].values
    elif data_name == 'flood1':
        X = pd.read_csv(df_path / 'FloodModeling1_DTW_train_n471_flatten_fixed_length_X.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'FloodModeling1_DTW_train_n471_flatten_fixed_length_y.csv',sep=',').iloc[:,1:].values
    elif data_name == 'energy':
        X = pd.read_csv(df_path / 'AppliancesEnergy_DTW_train_n95_flatten_fixed_length_X.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'AppliancesEnergy_DTW_train_n95_flatten_fixed_length_y.csv',sep=',').iloc[:,1:].values
    elif data_name == 'swissroll':
        
        X = pd.read_csv(df_path / 'swissroll_isomap_X_new.csv',sep=',').iloc[:,1:].values
        Y = pd.read_csv(df_path / 'swissroll_isomap_y_new.csv',sep=',').iloc[:,1].values  # Target is read as one column for Swiss roll
    else:
        print('dataset non presente')
    

    # Five-fold cross-validation with fixed folds (Section 5.1).
    kf = KFold(n_splits=kkk, shuffle=True, random_state=42)
    
    
    
    # Generate ten run seeds; the R RNG is set separately before each fit.
    np.random.seed(seed)
    random_seeds = np.random.randint(low=0, high=100, size=number_seeds)
    
    
    
    
    # Store training/test scores and fitting times for all 50 runs.
    accuracy_rf = []
    accuracy_train_rf = []
    mse_rf = []
    mse_train_rf = []
    timee = []


    
    
    
    kk = 1
    # Train and evaluate separately on each fold.
    for train_idx, test_idx in kf.split(X):
        # Split the features and scale using training-fold statistics only.
        

        X_train = X[train_idx]
        
        scaler_min = MinMaxScaler()
        X_train = scaler_min.fit_transform(X_train)
        
        X_test = X[test_idx]
        
        X_test = scaler_min.transform(X_test)
        
        y_train = Y[train_idx]
        y_test = Y[test_idx]
        
        # Standardize responses using training-fold statistics.
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
        y_test = scaler.transform(y_test.reshape(-1,1)).flatten()
        
        it = 1
        for sgl_seed in random_seeds:
        # Fit one Random Forest for each of the ten run seeds.
            
            print('Training Random Forest')
            # Convert the training features and response to an R data frame.
            data_dict = {f'V{i+1}': robjects.FloatVector(X_train[:, i]) for i in range(X_train.shape[1])}
            y_train_str = [str(i) for i in y_train]
            data_dict['y'] = robjects.FloatVector(y_train_str)
            
            # Initialize the R random generator for this run.
            r('set.seed('+str(sgl_seed)+')')
            
            # Assemble the training data frame.
            r_dataframe = robjects.r['data.frame'](**data_dict)
            # Regression formula: response as a function of all features.
            formula = Formula('y ~ .')
            # Measure model fitting time only.
            beg = time.time()
            rf_model = randomForest.randomForest(formula, data=r_dataframe, n_tree= n_estimators)
            end = time.time()
            
            timee.append(end-beg)
            # Construct test features with the same R column names.
            test_dict = {f'V{i+1}': robjects.FloatVector(X_test[:, i]) for i in range(X_test.shape[1])}
            r_test_df = robjects.r['data.frame'](**test_dict)
            # Evaluate on the test set.
            predicted_test =  np.array(robjects.r['predict'](rf_model, r_test_df))
            accuracy_rf.append(r2_score(y_test, predicted_test))
            mse_rf.append(np.sqrt(mean_squared_error(y_test,predicted_test)))
            print('Testing R quadro RF:',accuracy_rf[-1])
            print('Testing RMSE RF:',mse_rf[-1])
            
            
            # Evaluate on the training set.
            predicted_classes_tr =  np.array(robjects.r['predict'](rf_model, r_dataframe))
            accuracy_train_rf.append(r2_score(y_train, predicted_classes_tr))
            # Preserved: training column labeled RMSE contains MSE (no square root).
            mse_train_rf.append(mean_squared_error(y_train, predicted_classes_tr))
            
            
            
            
            it+=1
        kk+=1
    
    

    # Save per-run results followed by their means and standard deviations.
    df_simple = pd.DataFrame({
        "R square train RF": accuracy_train_rf,
        "R square test RF": accuracy_rf,
        "RMSE train RF": mse_train_rf,
        "RMSE test RF": mse_rf,
        "time": timee,
    })
    

    mean_simple_best = df_simple.mean(numeric_only=True)
    std_simple_best = df_simple.std(numeric_only=True)
    
    df_simple.loc[df_simple.shape[0]] = mean_simple_best
    df_simple.loc[df_simple.shape[0]] = std_simple_best
    
    df_simple.to_csv(data_name+'_rf.csv',sep=';', index=False)