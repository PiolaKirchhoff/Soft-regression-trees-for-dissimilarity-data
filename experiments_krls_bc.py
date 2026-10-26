# -*- coding: utf-8 -*-
"""Reproduce DSRT–KRLS and KronRLS comparisons on the Davis dataset.

This script implements the pair-input experiments corresponding to settings
B and C in Section 5.4 of ``Soft Regression Trees for Distance Data``.
It trains depth-two distance-based soft regression trees with Kronecker
regularized least-squares leaf models (``type_reg='kkls'``) and compares
predictions against the RLScore ``KronRLS`` baseline.

Five-fold cross-validation splits the *rows* of the routing matrix, with
ten random initializations per fold. ``setting_B=True`` routes drug rows;
``setting_B=False`` transposes the affinity matrix and routes target rows.
All counterpart entities remain available during each training fold.
The DSRT training routine is ``my_logistic_dec()``, not the different
``my_logistic_dec_new()`` variant. Prototype selection is disabled.

Input files are read from ``data/krls`` relative to this script.
CSV result files are saved in the process's current working directory,
using the original naming convention and original metric column names.

Command-line arguments (all positional):
    1. Dataset identifier: ``DT``.
    2. Optimizer: ``lbfgs``, ``tnc``, ``bfgs``, ``n_cg``, ``sqp``, etc.
    3. Clustering initialization: ``0`` or ``1``.
    4. Medoid type (retained for CLI compatibility; no prototype selection).
    5. Use similarity matrices: ``1``; ``0`` converts routing inputs to
       distances, while preserving the original matrices for the baseline.

Example: ``python experiments_krls_bc.py DT lbfgs 0 0 1``.
To run setting C, change ``setting_B`` to ``False`` below. The original
algorithm and cross-validation partitions are deliberately unchanged.
"""

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path
from rlscore.learner import KronRLS
from dsrt_krls import SSRT
from dti_eval_metrics import evaluate_setting

# Davis interaction and similarity matrices are stored next to the scripts.
data_dir = Path(__file__).resolve().parent / "data" / "krls"

def drug_drug_sim_to_dist(sim_matrix: np.ndarray) -> np.ndarray:
    """Convert an already-normalized drug similarity matrix to distances.

    The original transformation is ``sqrt(2 * (1 - similarity))`` after
    clipping similarities to [-1, 1]. The input is expected to have a
    unit diagonal. This function does not change any kernel used by
    the standalone KronRLS baseline, which receives separate copies.
    """
    sim_matrix = np.clip(sim_matrix, -1, 1)  # Preserve the original numerical clipping.
    dist_matrix = np.sqrt(2 * (1 - sim_matrix))
    return dist_matrix


def target_target_sim_to_dist(sim_matrix: np.ndarray) -> np.ndarray:
    """Normalize target similarities and convert them to distances.

    First apply diagonal normalization ``D^{-1/2} K D^{-1/2}``, replacing
    zero diagonal entries by 1e-12. Clip the normalized similarities
    to [-1, 1], then use ``sqrt(2 * (1 - similarity))``.
    """
    diag = np.diag(sim_matrix).copy()
    diag[diag == 0] = 1e-12  # Original safeguard for zero diagonal entries.
    D_inv_sqrt = np.diag(1.0 / np.sqrt(diag))
    K_normalized = D_inv_sqrt @ sim_matrix @ D_inv_sqrt
    K_normalized = np.clip(K_normalized, -1, 1)  # Preserve the original numerical clipping.
    dist_matrix = np.sqrt(2 * (1 - K_normalized))
    return dist_matrix


# Choose the entity used by the routing tree: drugs (True, setting B)
# or targets (False, setting C; the affinity matrix is transposed).
# Reference: https://staff.cs.utu.fi/~aatapa/software/RLScore/tutorial_kronecker.html
setting_B = True

# Enable the parallel objective and gradient evaluations in the BN step.
parallel= True
# Original experimental parameters: ten runs per fold, tree depth two.
number_seeds= 10
depth = 2
seed = 5
C = 1000.0
val = 70.0
# The command-line positions are retained to reproduce existing runs.
# Initialization can be disabled with 0; no default is silently changed.
clust_init = bool(int(sys.argv[3]))
# Parsed for compatibility with the original command line; prototype
# selection is disabled below, so this option is not exercised.
type_med = int(sys.argv[4])
# True: use supplied similarities in DSRT routing. False: convert them
# to distances, while the KronRLS comparison uses original similarities.
similarity_bool = bool(int(sys.argv[5]))

# All available training entities are used in branch-node problems.
type_nystrom = None



# Map the CLI choice to the optimizer name expected by the DSRT class.
if sys.argv[2] == 'lbfgs':
    solver = 'L-BFGS-B'
elif sys.argv[2] == 'tnc':
    solver = 'TNC'
elif sys.argv[2] == 'bfgs':
    solver = 'BFGS'
elif sys.argv[2] == 'n_cg':
    solver = 'Newton-CG'
elif sys.argv[2] == 'sqp':
    solver = 'SLSQP'
elif sys.argv[2] == 'ada_seq':
    solver = 'adam_seq'
elif sys.argv[2] == 'ada_vec':
    solver = 'adam_vec'
    


# Load the Davis affinity matrix and the two entity similarity matrices.
# Dataset selection follows the original single-dataset experiment.
if sys.argv[1] == 'DT':
    
    
    if setting_B:
        # Setting B: rows correspond to drugs, columns to targets.
        Y = np.loadtxt(data_dir / "drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt")
        X_base = np.loadtxt(data_dir / "drug-drug_similarities_2D.txt")
        X_aux = np.loadtxt(data_dir / "target-target_similarities_WS.txt")
        
        if not similarity_bool:
            # Retain the original similarity matrices for baseline KronRLS.
            X_base_kls = X_base.copy()
            X_aux_kls = X_aux.copy()
            X_base = drug_drug_sim_to_dist(X_base)
            X_aux = target_target_sim_to_dist(X_aux)          
        
        
    else:
        # Setting C: transpose affinities so rows correspond to targets.
        Y = np.loadtxt(data_dir / "drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt").T
        X_aux = np.loadtxt(data_dir / "drug-drug_similarities_2D.txt")
        X_base = np.loadtxt(data_dir / "target-target_similarities_WS.txt")
        
        if not similarity_bool:
            # Retain the original similarity matrices for baseline KronRLS.
            X_base_kls = X_base.copy()
            X_aux_kls = X_aux.copy()
            X_aux = drug_drug_sim_to_dist(X_aux)
            X_base = target_target_sim_to_dist(X_base) 

    list_idx = range(X_base.shape[0])
else:
    print('dataset non presente')
    

# Five shuffled folds with a fixed partition seed, shared by both models.
kf = KFold(n_splits=5, shuffle=True, random_state=42)



# Reuse the same ten initial seeds in every fold; do not alter this RNG.
np.random.seed(seed)
random_seeds = np.random.randint(low=0, high=100, size=number_seeds)




# Store metrics for each DSRT–KRLS fit (historical ``simple`` names
# refer to the DSRT model, not constant-valued leaf predictions).
accuracy_simple = []
rscore_simple = []
cindex_simple = []

accuracy_train_simple = []
rscore_train_simple = []
cindex_train_simple = []

time_simple = []

# Store metrics for the standalone KronRLS baseline (historical
# ``linear reg`` names are retained for CSV compatibility).
accuracy_linreg = []
rscore_linreg = []
cindex_linreg = []

accuracy_train_linreg = []
rscore_train_linreg = []
cindex_train_linreg = []

time_linreg = []



kk = 1
# Split the routed entity axis only; the counterpart axis is never split.
for train_idx, test_idx in kf.split(list_idx):
    
    sim_train = X_base[np.ix_(train_idx,train_idx)]
    sim_test = X_base[np.ix_(test_idx,train_idx)]
    
    if not similarity_bool:
        # The baseline is trained on the unmodified matrices even when
        # the DSRT uses transformed distances for routing.
        sim_train_kls = X_base_kls[np.ix_(train_idx,train_idx)]
        
        sim_test_kls = X_base_kls[np.ix_(test_idx,train_idx)]
    
    
    # Targets are interaction submatrices. For setting C, Y was already
    # transposed, so these slices still operate on the routed entity axis.
    y_train = Y[train_idx,:]
    y_test = Y[test_idx,:]
    
    
    
    it = 1
    for sgl_seed in random_seeds:
        # The original positive-index guard is retained verbatim.
        # It is always true for the configured folds and initializations.
        # Train the DSRT with KronRLS leaf regressors.
        if (kk-1)*number_seeds+it > -1:
            model = SSRT(depth)
            model.set_dataset(sim_train,y_train,'kkls',X_aux,1e-3)
            model.set_param_KLR(C=C,nu=1e-6,value=val,seed = sgl_seed)
            model.set_nystrom(typ=type_nystrom,percentage=0.01,ridge_leverage_lambda = 1)  # Inactive for typ=None.
            model.starting_point(sgl_seed)
            print('Parallel: ',parallel)
            print('Training tree:')
            beg = time.time()

            model.my_logistic_dec(n_iter = 10, mini_iter = 1000,seed=sgl_seed,eps1= 0.3, eps2=0.1,eps3= 0.4,decrement_rate = 0.9,init_flag = clust_init,num_clust= 100,gower=False,parallel=parallel,solver=solver)
            end = time.time()
            time_simple.append(end-beg)
        
        
            # Predict all training interactions using hard leaf assignments.
            y_pred_train = model.predict_kkls(sim_train,X_aux)
            
            temp_acc_train = evaluate_setting(y_train,y_pred_train)["RMSE"]
            accuracy_train_simple.append(temp_acc_train)
            cindex_train_simple.append(evaluate_setting(y_train,y_pred_train)["C_index"])
            rscore_train_simple.append(r2_score(y_train,y_pred_train))
            
            # Predict interactions of held-out routing entities with all
            # counterpart entities represented in the training matrix.
            y_pred = model.predict_kkls(sim_test,X_aux)
            temp_acc = evaluate_setting(y_test,y_pred)["RMSE"]
            print('Run:',(kk-1)*number_seeds+it)
            print('Accuracy simple:',temp_acc)
            accuracy_simple.append(temp_acc)
            rscore_simple.append(r2_score(y_test,y_pred))
            cindex_simple.append(evaluate_setting(y_test,y_pred)["C_index"])
            
            # Fit the standalone KronRLS baseline. The original code uses
            # the X1/X2 interface of RLScore (not the K1/K2 interface).
            if not similarity_bool:
                learner = KronRLS(X1=sim_train_kls, X2=X_aux_kls, Y=y_train)
            
                print('Training KLscore:')
                beg = time.time()
                learner.solve(1e-3)
                end = time.time()
                time_linreg.append(end-beg)
            
                P = learner.predict(sim_test_kls, X_aux_kls)
                P = P.reshape((sim_test_kls.shape[0], X_aux_kls.shape[0]), order='F')
            
                P_train = learner.predict(sim_train_kls, X_aux_kls)
                P_train = P_train.reshape((sim_train_kls.shape[0], X_aux_kls.shape[0]), order='F')
            else:
            
                learner = KronRLS(X1 = sim_train, X2 = X_aux, Y = y_train)
            
            
                print('Training KLscore:')
                beg = time.time()
                learner.solve(1e-3)
                end = time.time()
                time_linreg.append(end-beg)
            
                P = learner.predict(sim_test, X_aux)
                P = P.reshape((sim_test.shape[0], X_aux.shape[0]), order='F')
                
                P_train = learner.predict(sim_train, X_aux)
                P_train = P_train.reshape((sim_train.shape[0], X_aux.shape[0]), order='F')
            # Evaluate the KronRLS baseline on training interactions.
            temp_acc_train = evaluate_setting(y_train,P_train)["RMSE"]
            accuracy_train_linreg.append(temp_acc_train)
            cindex_train_linreg.append(evaluate_setting(y_train,P_train)["C_index"])
            rscore_train_linreg.append(r2_score(y_train,P_train))
            
            # Evaluate the baseline on exactly the same held-out rows.
            temp_acc = evaluate_setting(y_test,P)["RMSE"]
            print('Run:',(kk-1)*number_seeds+it)
            print('Accuracy linear regression:',temp_acc)
            accuracy_linreg.append(temp_acc)
            rscore_linreg.append(r2_score(y_test,P))
            cindex_linreg.append(evaluate_setting(y_test,P)["C_index"])
        
        
        it+=1
    kk+=1



# Preserve original column labels and metrics. Despite the ``mse`` column
# names, evaluate_setting()["RMSE"] produces root mean squared error.
df = pd.DataFrame({
    "mse train simple": accuracy_train_simple,
    "R quadro train simple": rscore_train_simple,
    "C index train simple": cindex_train_simple,
    "mse test simple": accuracy_simple,
    "R quadro test simple": rscore_simple,
    "C index test simple": cindex_simple,
    "time simple": time_simple,
    "mse train linear reg": accuracy_train_linreg,
    "R quadro train linear reg": rscore_train_linreg,
    "C index train linear reg": cindex_train_linreg,
    "mse test linear reg": accuracy_linreg,
    "R quadro test linear reg": rscore_linreg,
    "C index test linear reg": cindex_linreg,
    "time linear reg": time_linreg
    
})

# Append means of numeric columns using the original pandas behavior.
df.loc["media"] = df.mean()

# Keep the original result-file names to preserve existing workflows.
if type_nystrom == None:
    df.to_csv(sys.argv[1]+'depth_'+str(depth)+'_solver_'+solver+'init_proc'+str(clust_init)+'_nystrom_None_similarity_'+str(similarity_bool)+'settingB_'+str(setting_B)+'.csv',sep=';', index=False)
else:
    df.to_csv(sys.argv[1]+'depth_'+str(depth)+'_solver_'+solver+'init_proc'+str(clust_init)+'_nystrom_'+type_nystrom+'_type_mediods_'+str(type_med)+'settingB_'+str(setting_B)+'.csv',sep=';', index=False)
