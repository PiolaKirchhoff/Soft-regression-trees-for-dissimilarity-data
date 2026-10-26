# -*- coding: utf-8 -*-
"""Run Random Forest benchmarks on tabular regression datasets (Section 5.2).

This script evaluates the R randomForest model on the eight tabular
regression datasets considered in the paper. For each dataset, it performs
five-fold cross-validation with ten random seeds per fold, standardizes
responses using the training fold, and records training and test scores.
Results for all 50 runs per dataset, followed by summary statistics, are
saved to CSV files.

The original local dataset paths and the explicit R_HOME setting are
preserved. The code uses R through rpy2, not scikit-learn's Random Forest.
"""

import os
from pathlib import Path
# Set R_HOME to the directory where R is installed on your system.
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.2.1"
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


from rpy2 import robjects
from rpy2.robjects import numpy2ri,r
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
import rpy2.robjects.vectors as rvectors
# Enable the NumPy-to-R conversion used by the original script.
numpy2ri.activate()
# Load the randomForest package from R.
randomForest = importr('randomForest')

# Random Forest and experiment settings.
n_estimators = 500
number_seeds= 10
seed = 56
    
df_path = Path(os.environ.get("DSRT_FUNCTIONAL_DATA_DIR", Path(__file__).resolve().parent / "data" / "tabular"))
# All eight tabular regression datasets from Section 5.2.
list_dataset = ['abalone','yacht','auto-mpg','airfoil','housing','ailerons','kin8nm','elevators']


kkk = 5 # Number of cross-validation folds.


for data_name in list_dataset:

    if data_name == 'housing':
    # Load the dataset using the original source and path.
        data = pd.read_csv(df_path / 'housing.csv',sep=';')
    elif data_name == 'ailerons':
        data = pd.read_csv(df_path / 'ailerons.data',sep=',',header=None)
    elif data_name == 'kin8nm':
        kin = fetch_openml(data_id = 189, as_frame = True)
        X = kin.data
        y = kin.target
        data = pd.concat([X,y],axis=1)
    elif data_name == 'elevators':
        el = fetch_openml(name='elevators', version=1, as_frame=True)#(data_id = 225, as_frame = True)
        X = el.data
        y = el.target
        data = pd.concat([X,y],axis=1)
    elif data_name == 'abalone':
        ab = fetch_openml(name='abalone', version=1, as_frame=True)
        X = pd.get_dummies(ab.data, columns=['Sex'], drop_first=True).astype(float) 
        y = ab.target.astype(float)
        # The local CSV replaces the preceding OpenML-derived data (original behavior).
        data = pd.read_csv(df_path / 'abalone.csv',sep=';')
    elif data_name == 'airfoil':
        data = pd.read_csv('airfoil_self_noise.dat',sep='\t',header=None)
    elif data_name == 'auto-mpg':
        column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight","acceleration", "model_year", "origin", "car_name"]
        data = pd.read_csv(df_path / 'auto-mpg.data',delim_whitespace=True, names=column_names, na_values='?')
    
        # Remove the non-numeric car_name column.
        data.drop(columns=["car_name"], inplace=True)
        # Fill missing horsepower values with the column mean.
        data["horsepower"].fillna(data["horsepower"].mean(), inplace=True)
        # Move the regression target to the final column.
        mpg = data.pop("mpg")
        data["mpg"] = mpg
    elif data_name == 'yacht':
        data = pd.read_csv(df_path / 'yacht_hydrodynamics.data', delim_whitespace=True, header=None)
    else:
        print('dataset not found')
        
    X = np.asarray(data.iloc[:,:-1].values)
    Y = np.asarray(data.iloc[:,-1].values)

    
    # Random Forest uses input features; no distance matrix is required.

    # Reproducible five-fold cross-validation (Section 5.1).
    kf = KFold(n_splits=kkk, shuffle=True, random_state=42)
    
    
    
    # Generate the same ten NumPy seeds for every dataset and fold.
    np.random.seed(seed)
    random_seeds = np.random.randint(low=0, high=100, size=number_seeds)
    
    
    
    
    # Collect training and test metrics for the 50 single-start runs.
    accuracy_rf = []
    accuracy_train_rf = []
    mse_rf = []
    mse_train_rf = []
    


    
    
    
    kk = 1
    # Train and evaluate separately on each fold.
    for train_idx, test_idx in kf.split(X):
        # Select training and test observations for this fold.
        

        
        X_train = X[train_idx]
        
        # Fit scaling on training features only, then transform the test fold.
        scaler_min = MinMaxScaler()
        X_train = scaler_min.fit_transform(X_train)
        
        X_test = X[test_idx]
        
        X_test = scaler_min.transform(X_test)
        
        y_train = Y[train_idx]
        y_test = Y[test_idx]
        
        # Standardize the response using training-fold statistics.
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
        y_test = scaler.transform(y_test.reshape(-1,1)).flatten()
        
        it = 1
        for sgl_seed in random_seeds:
        # Fit the R Random Forest model for each random seed.
            
            print('Training Random Forest')
            # Build R-compatible feature columns and training responses.
            data_dict = {f'V{i+1}': robjects.FloatVector(X_train[:, i]) for i in range(X_train.shape[1])}

            y_train_str = [str(i) for i in y_train]
            data_dict['y'] = robjects.FloatVector(y_train_str)
            
            # Seed R independently for each training run.
            r('set.seed('+str(sgl_seed)+')')
            
            # Construct the R training data frame.
            r_dataframe = robjects.r['data.frame'](**data_dict)
            # Use the response-versus-features formula for regression.
            formula = Formula('y ~ .')
            # Train a Random Forest with the specified number of trees.
            rf_model = randomForest.randomForest(formula, data=r_dataframe, n_tree= n_estimators)
            
            # Use the same feature names for the R test data frame.
            test_dict = {f'V{i+1}': robjects.FloatVector(X_test[:, i]) for i in range(X_test.shape[1])}
            r_test_df = robjects.r['data.frame'](**test_dict)
            # Obtain test predictions and evaluate predictive accuracy.
            predicted_test =  np.array(robjects.r['predict'](rf_model, r_test_df))
            accuracy_rf.append(r2_score(y_test, predicted_test))
            mse_rf.append(np.sqrt(mean_squared_error(y_test,predicted_test)))
            print('Testing R quadro RF:',accuracy_rf[-1])
            print('Testing RMSE RF:',mse_rf[-1])
            
            
            predicted_classes_tr =  np.array(robjects.r['predict'](rf_model, r_dataframe))
            accuracy_train_rf.append(r2_score(y_train, predicted_classes_tr))
            # This stores training MSE, unlike the square-rooted test RMSE.
            mse_train_rf.append(mean_squared_error(y_train, predicted_classes_tr))
            
            
            
            
            it+=1
        kk+=1
    
    

    # Assemble single-start scores and append mean/std summary rows.
    df_simple = pd.DataFrame({
        "R square train RF": accuracy_train_rf,
        "R square test RF": accuracy_rf,
        "RMSE train RF": mse_train_rf,
        "RMSE test RF": mse_rf,
    })
    

    mean_simple_best = df_simple.mean(numeric_only=True)
    std_simple_best = df_simple.std(numeric_only=True)
    
    df_simple.loc[df_simple.shape[0]] = mean_simple_best
    df_simple.loc[df_simple.shape[0]] = std_simple_best
    
    # Save one CSV per dataset with all runs and the summary rows.
    df_simple.to_csv(data_name+'_rf.csv',sep=';', index=False)