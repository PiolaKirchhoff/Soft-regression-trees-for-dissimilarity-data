# -*- coding: utf-8 -*-
"""Evaluate DSRT with Kronecker regularized least-squares leaves: setting D.

This script reproduces the setting-D comparison in Section 5.4 of the
accompanying paper using the Davis drug-target interaction dataset. Both
entities of each test interaction are unseen during training: 20% of drugs
and 20% of targets are independently held out in each of ten random splits.

For each split and each of ten initialization seeds, two depth-two DSRTs
are fitted: a drug-routed tree and a target-routed tree. The prediction
for an interaction is the equally weighted (0.5/0.5) average of the two
predictions, with the target-routed matrix transposed to drug-by-target
orientation. Both trees use the original ``my_logistic_dec()`` procedure,
L-BFGS-B, and no prototype selection. An independent standalone KronRLS
baseline is fitted once per split using the original similarity matrices.

The script reports only root-mean-squared error (RMSE). Neither model
selection based on training RMSE nor best-of-start results are performed;
all ten initializations contribute equally to the reported test RMSE.
Results are written as four semicolon-delimited CSV files in the current
working directory:

* ``train_combined_dep2rmse.csv``: training RMSE for the averaged DSRTs.
* ``test_combined_dep2rmse.csv``: test RMSE for the averaged DSRTs.
* ``train_baseline_dep2rmse.csv``: training RMSE for standalone KronRLS.
* ``test_baseline_dep2rmse.csv``: test RMSE for standalone KronRLS.

Each CSV has ten rows corresponding to split seeds in their generated
order, and ten columns corresponding to initialization seeds. The DSRT
CSV entries represent independently trained DSRT pairs. Each baseline
row repeats the single KronRLS score for that split across ten columns
solely to preserve the original output alignment; these repeated values
do not represent additional baseline fits. To reproduce the overall test
RMSEs compared in Table 3, average the entries of the respective test
CSV; equivalently, average split-level means (each split has ten starts).

The input files are expected in ``data/krls`` relative to this script:

    drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt
    drug-drug_similarities_2D.txt
    target-target_similarities_WS.txt

Execute with ``python experiments_krls_d.py`` without command-line options.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from rlscore.learner import KronRLS

from dsrt_krls import SSRT

# Resolve input files relative to the script, not to the working directory.
data_dir = Path(__file__).resolve().parent / "data" / "krls"


def drug_drug_sim_to_dist(sim_matrix: np.ndarray) -> np.ndarray:
    """Convert normalized drug-drug similarities into kernel-induced distances.

    Clip similarities to [-1, 1], then compute sqrt(2 * (1 - similarity)).
    The transformation is used by the two DSRT routing and leaf models.
    """
    sim_matrix = np.clip(sim_matrix, -1, 1)
    dist_matrix = np.sqrt(2 * (1 - sim_matrix))
    return dist_matrix


def target_target_sim_to_dist(sim_matrix: np.ndarray) -> np.ndarray:
    """Normalize target similarities and compute kernel-induced distances.

    Replace zero diagonal entries by 1e-12, apply D^(-1/2) K D^(-1/2),
    clip to [-1, 1], and compute sqrt(2 * (1 - normalized similarity)).
    """
    diag = np.diag(sim_matrix).copy()
    diag[diag == 0] = 1e-12
    D_inv_sqrt = np.diag(1.0 / np.sqrt(diag))
    K_normalized = D_inv_sqrt @ sim_matrix @ D_inv_sqrt
    K_normalized = np.clip(K_normalized, -1, 1)
    dist_matrix = np.sqrt(2 * (1 - K_normalized))
    return dist_matrix


def load_davis():
    """Load Davis drug-target affinities and both entity similarity matrices.

    Returns
    -------
    XD : numpy.ndarray
        Drug-drug similarity matrix.
    XT : numpy.ndarray
        Target-target similarity matrix.
    Y : numpy.ndarray
        Drug-by-target affinity matrix.

    Notes
    -----
    Input files are read from ``data/krls`` relative to this script.
    """
    Y = np.loadtxt(data_dir / "drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt")
    XD = np.loadtxt(data_dir / "drug-drug_similarities_2D.txt")
    XT = np.loadtxt(data_dir / "target-target_similarities_WS.txt")
    return XD, XT, Y


def settingD_split(seed,perc =0.2):
    """Generate a setting-D split with unseen drugs AND unseen targets.

    Each entity axis is shuffled separately after setting NumPy's RNG seed.
    The fraction ``perc`` is held out on both axes; test interactions are
    restricted to pairs between held-out drugs and held-out targets.

    DSRT receives distances for both entity types, whereas the standalone
    KronRLS baseline uses the original similarity matrices. The ten returned
    arrays are ordered as routing train (drug, target), training affinities,
    routing test (drug, target), test affinities, baseline train (drug,
    target), and baseline test (drug, target).

    Parameters
    ----------
    seed : int
        Random seed controlling both entity permutations.
    perc : float, default=0.2
        Held-out fraction for each entity type.

    Returns
    -------
    tuple of numpy.ndarray
        The original ten split matrices, in their original order.
    """
    # Seed BOTH independent entity-axis permutations. This also resets the
    # global RNG as in the original experiment.
    np.random.seed(seed)
    XD_old, XT_old, Y = load_davis()

    # Distances are used for DSRT routing and its KRLS leaf regression.
    # The untouched similarities below are retained for the RLScore baseline.
    XD = drug_drug_sim_to_dist(XD_old)
    XT = target_target_sim_to_dist(XT_old)
    drug_ind = list(range(Y.shape[0]))
    target_ind = list(range(Y.shape[1]))
    len_drug = len(drug_ind)
    len_target = len(target_ind)
    np.random.shuffle(drug_ind)
    np.random.shuffle(target_ind)
    # Partition drugs and targets independently: neither held-out entity
    # type occurs in the training interaction submatrix.
    train_drug_ind = drug_ind[:int((1-perc)*len_drug)]
    test_drug_ind = drug_ind[int((1-perc)*len_drug):]
    train_target_ind = target_ind[:int((1-perc)*len_target)]
    test_target_ind = target_ind[int((1-perc)*len_target):]
    # Observed train-by-train interactions and held-out-by-held-out tests.
    Y_train = Y[np.ix_(train_drug_ind, train_target_ind)]
    Y_test = Y[np.ix_(test_drug_ind, test_target_ind)]
    # DSRT training matrices and test-to-training cross-distances.
    XD_train = XD[np.ix_(train_drug_ind,train_drug_ind)]
    XT_train = XT[np.ix_(train_target_ind,train_target_ind)]

    # Original, untransformed similarities for the KronRLS baseline.
    XD_train_krls = XD_old[np.ix_(train_drug_ind,train_drug_ind)]
    XT_train_krls = XT_old[np.ix_(train_target_ind,train_target_ind)]

    XD_test = XD[np.ix_(test_drug_ind,train_drug_ind)]
    XT_test = XT[np.ix_(test_target_ind,train_target_ind)]

    XD_test_krls = XD_old[np.ix_(test_drug_ind,train_drug_ind)]
    XT_test_krls = XT_old[np.ix_(test_target_ind,train_target_ind)]

    return XD_train, XT_train, Y_train, XD_test, XT_test, Y_test, XD_train_krls, XT_train_krls, XD_test_krls, XT_test_krls


# Equal-weight averaging of the drug-routed and target-routed DSRT outputs.
w1 = 0.5
w2 = 1 - w1
# Experimental design: ten independent drug/target splits, and ten initial
# tree parameter seeds evaluated on each split (100 pairs of DSRT models).
number_seeds = 10
# Preserve the original serial BN objective/gradient configuration.
parallel = False
depth = 2
seed_init = 5
C = 1000.0
val = 70.0
# Clustering-based initialization and prototype selection are both disabled.
clust_init = False
type_nystrom = None

# Generate split seeds FIRST, then initial-state seeds, without reseeding
# between the two draws. Preserve this order for reproducibility.
np.random.seed(seed_init)
list_seed_splits = np.random.randint(low=0, high=100, size=10)
random_seeds = np.random.randint(low=0, high=100, size=number_seeds)


# One RMSE entry per split and initialization for the combined DSRTs.
list_list_train_acc_overall = []
list_list_test_acc_overall = []

# KronRLS is fitted only once per train/test split and does not depend
# on the DSRT initialization seed. Its RMSE is repeated across the
# ten columns solely to match the structure of the DSRT output files.
# Baseline RMSE values may still differ between train/test splits.
list_train_baseline = []
list_test_baseline = []

# Outer loop: ten independent 80/20 splits of BOTH drug and target entities.
for seed in list_seed_splits:

    # Preserve the original random-state reset before generating each split.
    np.random.seed(seed)
    XD_train, XT_train, Y_train, XD_test, XT_test, Y_test, XD_train_krls, XT_train_krls, XD_test_krls, XT_test_krls = settingD_split(seed,perc = 0.2)

    list_train_acc_overall = []
    list_test_acc_overall = []

    # Fit both routing orientations for each of the ten initializations.
    # No best-model selection is performed; each pair contributes one RMSE.
    for sgl_seed in random_seeds:
        # Route by drug distances; use target distances in the KRLS leaves.
        model_d = SSRT(depth)
        model_d.set_dataset(XD_train,Y_train,'kkls',XT_train,1e-3)
        model_d.set_param_KLR(C=C,nu=1e-6,value=val,seed = sgl_seed)
        model_d.set_nystrom(typ=type_nystrom,percentage=0.01,ridge_leverage_lambda = 1)
        model_d.starting_point(sgl_seed)
        model_d.my_logistic_dec(n_iter = 10, mini_iter = 1000,seed=sgl_seed,eps1= 0.3, eps2=0.1,eps3= 0.4,decrement_rate = 0.9,init_flag = clust_init,num_clust= 100,gower=False,parallel=parallel,solver='L-BFGS-B')

        # Route by target distances and transpose the affinity matrix so
        # that each tree uses its routed entities along the first axis.
        model_t = SSRT(depth)
        model_t.set_dataset(XT_train,Y_train.T,'kkls',XD_train,1e-3)
        model_t.set_param_KLR(C=C,nu=1e-6,value=val,seed = sgl_seed)
        model_t.set_nystrom(typ=type_nystrom,percentage=0.01,ridge_leverage_lambda = 1)
        model_t.starting_point(sgl_seed)
        model_t.my_logistic_dec(n_iter = 10, mini_iter = 1000,seed=sgl_seed,eps1= 0.3, eps2=0.1,eps3= 0.4,decrement_rate = 0.9,init_flag = clust_init,num_clust= 100,gower=False,parallel=parallel,solver='L-BFGS-B')

        # Average predictions entrywise in drug-by-target orientation.
        # RMSE is calculated only after averaging, not averaged between trees.
        y_pred_train_d = model_d.predict_kkls(XD_train,XT_train)
        y_pred_train_t = model_t.predict_kkls(XT_train,XD_train)
        # Same-start ensemble: average current drug prediction with the
        # transposed current target prediction, then compute global RMSE.
        rmse_train_combined = np.sqrt(mean_squared_error(Y_train,(w1*y_pred_train_d)+(w2*y_pred_train_t).T))

        y_pred_test_d = model_d.predict_kkls(XD_test,XT_test.T)
        y_pred_test_t = model_t.predict_kkls(XT_test,XD_test.T)
        rmse_test_combined = np.sqrt(mean_squared_error(Y_test,(w1*y_pred_test_d)+(w2*y_pred_test_t).T))


        list_train_acc_overall.append(rmse_train_combined)
        list_test_acc_overall.append(rmse_test_combined)


    list_list_train_acc_overall.append(list_train_acc_overall)
    list_list_test_acc_overall.append(list_test_acc_overall)

    # One unweighted KronRLS baseline per split, using the unmodified
    # entity similarity matrices rather than the distances used by DSRT.
    lam = 1e-3

    learner = KronRLS(X1 = XD_train_krls, X2 = XT_train_krls, Y = Y_train)
    learner.solve(lam)

    P = learner.predict(XD_test_krls, XT_test_krls)
    P = P.reshape((XD_test_krls.shape[0], XT_test_krls.shape[0]), order='F')
    rmse_test_baseline = np.sqrt(mean_squared_error(Y_test,P))


    P_train = learner.predict(XD_train_krls, XT_train_krls)
    P_train = P_train.reshape((XD_train_krls.shape[0], XT_train_krls.shape[0]), order='F')
    rmse_train_baseline = np.sqrt(mean_squared_error(Y_train,P_train))

    # Repeat the scalar baseline RMSE across all starts for CSV alignment.
    list_train_baseline.append([rmse_train_baseline for i in range(len(list_test_acc_overall))])
    list_test_baseline.append([rmse_test_baseline for i in range(len(list_test_acc_overall))])

# Export only the quantities needed for the setting-D comparison.
# Rows: 10 splits; columns: 10 DSRT starts (repeated baseline scores).
# Outputs are written to the current working directory.
pd.DataFrame(list_list_train_acc_overall).to_csv('train_combined_dep'+str(depth)+'rmse.csv',sep=';',index=False)
pd.DataFrame(list_list_test_acc_overall).to_csv('test_combined_dep'+str(depth)+'rmse.csv',sep=';',index=False)
pd.DataFrame(list_train_baseline).to_csv('train_baseline_dep'+str(depth)+'rmse.csv',sep=';',index=False)
pd.DataFrame(list_test_baseline).to_csv('test_baseline_dep'+str(depth)+'rmse.csv',sep=';',index=False)
