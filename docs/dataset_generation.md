# Dataset generation and preparation

This document records how the functional benchmark data in *Soft Regression Trees for Dissimilarity Data* were generated. It supplements the [main README](../README.md), which describes the model, environments and experiments.

## Directory organization

The notebooks already present in the original repository should be moved, **without altering their contents**, from `dataset generation/` to `notebooks/dataset_generation/`. Their existing CSV outputs should be preserved under `generated_datasets/` (formerly `generated datasets/`), retaining the original `swiss_roll/` and `time_series/<dataset>/` subdirectories.

Those generated outputs are the **source data**. The experimental scripts require a separate, flat input layout at `data/functional/`; see the mapping below. Do not assume a change of filename alone proves equivalence between older output files and the experiment inputs.

## Swiss-roll benchmark

The Swiss-roll generation notebook uses `sklearn.datasets.make_swiss_roll` with **1,000 observations**, Gaussian noise **0.05**, and random seed **2020**. It constructs weighted neighborhood graphs for candidate neighbor counts **6, 8, ..., 30**, retains connected graphs, computes graph shortest-path distances, and obtains a two-dimensional Classical MDS embedding. The selected neighborhood size minimizes the residual variance

$$
\mathrm{RV}=1-\rho(D,D_X)^2,
$$

where the correlation compares the upper-triangular entries of the geodesic and embedded Euclidean distance matrices. The regression response is the unrolled intrinsic coordinate `t` returned by `make_swiss_roll`. The notebook was originally run with scikit-learn **1.6.1**.

The generated folder `generated_datasets/swiss_roll/` contains the geodesic distance matrix (`swissroll_isomap_D.csv`), embedded coordinates (`swissroll_isomap_X.csv`), responses (`swissroll_isomap_y.csv`) and neighborhood diagnostics (`swissroll_isomap_k_diagnostics.csv`).

## Time-series regression benchmarks

The time-series notebook loads the **training split** of six regression datasets via `aeon.datasets.load_regression`. Pairwise **Dynamic Time Warping (DTW)** distances are computed for DSRT; fixed-length observations are flattened to ordinary feature vectors for Random Forest. Continuous response values are exported unchanged. The benchmark datasets are:

| Dataset | Folder |
|---|---|
| Covid3Month | `covid` |
| FloodModeling1 | `flood1` |
| FloodModeling2 | `flood2` |
| FloodModeling3 | `flood3` |
| AppliancesEnergy | `energy` |
| IronConcentration | `iron` |

The outputs are stored under `generated_datasets/time_series/<folder>/`. Each contains a distance CSV (`*_D.csv`), feature CSV (`*_X.csv`), and response CSV (`*_y.csv`). The notebooks were developed in **Google Colab**, independently of the Conda environments supplied for experiments. They require notebook-specific dependencies such as `aeon` and `tslearn`; their precise original Colab environment has not been archived.

## Using generated data in the experiments

Both `experiments_functional_dsrt.py` and `experiments_functional_rf.py` expect their input files in **`data/functional/` next to the scripts**, unless `DSRT_FUNCTIONAL_DATA_DIR` is defined. The directory must contain the names below (with suffixes `_D.csv`, `_X.csv` and `_y.csv` for the six time-series datasets):

| Dataset selection | Expected filename stem |
|---|---|
| `covid` | `Covid3Month_DTW_train_n140_flatten_fixed_length` |
| `flood1` | `FloodModeling1_DTW_train_n471_flatten_fixed_length` |
| `flood2` | `FloodModeling2_DTW_train_n466_flatten_fixed_length` |
| `flood3` | `FloodModeling3_DTW_train_n429_flatten_fixed_length` |
| `energy` | `AppliancesEnergy_DTW_train_n95_flatten_fixed_length` |
| `iron` | `IronConcentration_DTW_train_n427_flatten_fixed_length` |

For `swissroll`, the experiment scripts expect `swissroll_isomap_D_new.csv`, `swissroll_isomap_X_new.csv` and `swissroll_isomap_y_new.csv` (the Random Forest script uses the last two). **These names differ from those listed in the older dataset-generation README.** Before staging or renaming files, check that their values, shapes, row ordering, headers and index columns match the experimental loaders; the scripts drop the first CSV column for feature/response files and expect the distance matrix in the shape they read. In particular, do not assume that an older `swissroll_isomap_D.csv` is numerically interchangeable with `swissroll_isomap_D_new.csv`.

Use `DSRT_FUNCTIONAL_DATA_DIR` to point the experiments to an existing directory of correctly formatted files, if preferred. The same documentation applies to RF features, which are generated from the time-series observations rather than from the dissimilarity matrices. Scaling is carried out **within the experimental cross-validation folds**, not by the generation notebooks.

## Reproducibility note

If the original Windows experimental environment is retained, it belongs at `docs/environments/experiments_environment.yml`; it is not a record of the notebooks' Colab runtimes. Use `environment.yml` for the core DSRT scripts, `environment_krls.yml` only for KRLS, and install the notebook dependencies separately if regenerating data. Keep original datasets and their attribution/licenses alongside any processed forms that you distribute.
