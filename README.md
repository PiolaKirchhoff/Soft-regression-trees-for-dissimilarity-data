# Soft Regression Trees for Dissimilarity Data

Research code and datasets accompanying *Soft Regression Trees for Dissimilarity Data*. The repository implements **Distance-based Soft Regression Trees (DSRTs)** with constant or linear-regression leaves, prototype selection for branch-node optimization, and an extension with **Kronecker regularized least squares (KRLS)** leaves for drug–target interactions.

## Repository structure

```text
.
├── README.md
├── environment.yml                  # Core DSRT dependencies (no RLScore)
├── environment_krls.yml             # Optional KRLS dependencies
├── setup_cython_dsrt.py             # Builds SMO, util and function extensions
├── SMO.pyx, util.pyx, function.pyx
├── dsrt.py                          # Standard DSRT class
├── dsrt_krls.py                     # DSRT extension with KRLS leaves
├── util_regression_KLR.py          # Routing and tree utilities
├── utils_numba_l2_nystrom.py       # Numba objectives and gradients
├── nystrom_utils_2.py              # Prototype selection
├── util_gower.py                    # Tabular dissimilarities
├── dti_eval_metrics.py             # Davis evaluation metrics
├── esperimenti_no_sparsity_tabular.py
├── esperimenti_sparsity_tabular.py
├── experiments_tabular_rf.py
├── experiments_functional_dsrt.py
├── experiments_functional_rf.py
├── experiments_krls_bc.py
├── experiments_krls_d.py
├── notebooks/
│   └── dataset_generation/         # Previously: "dataset generation"
├── generated_datasets/            # Previously: "generated datasets"; original notebook outputs
├── data/
│   ├── functional/                 # Experiment-ready, flat CSV layout (see below)
│   ├── tabular/                    # Locally stored tabular datasets (see loading note below)
│   └── krls/                       # Davis affinity and similarity matrices
└── docs/
    ├── dataset_generation.md       # Dataset provenance and notebook instructions
    └── environments/
        └── experiments_environment.yml  # Archived original Windows Conda export
```

Keep the model classes, utilities, experiment scripts and Cython sources **at the repository root**: their current imports and relative paths depend on this layout. The `data/` directory contains only `functional/`, `tabular/` and `krls/`. Generated notebook outputs are preserved separately at the repository root in `generated_datasets/`. The detailed dataset-generation documentation is in [docs/dataset_generation.md](docs/dataset_generation.md).

## Installation

The experiments were developed on Windows with **Python 3.11.11** (NumPy 2.1.2, SciPy 1.15.3, Numba 0.61.2, scikit-learn 1.6.1, Cython 3.1.1). The YAML files pin the principal versions, but are not complete platform-specific lockfiles.

Clone the repository and install the **core environment**, sufficient for tabular and functional DSRT experiments:

```bash
git clone https://github.com/PiolaKirchhoff/Soft-regression-trees-for-dissimilarity-data.git
cd Soft-regression-trees-for-dissimilarity-data
conda env create -f environment.yml
conda activate dsrt
python setup_cython_dsrt.py build_ext --inplace
python -c "import SMO, util, function; from dsrt import SSRT; print('DSRT ready')"
```

Compiling `SMO.pyx`, `util.pyx` and `function.pyx` requires a compatible C/C++ compiler (on Windows, Microsoft C++ Build Tools). Rebuild after editing these sources.

**KRLS only:** use the separate environment instead of the core one:

```bash
conda env create -f environment_krls.yml
conda activate dsrt-krls
python setup_cython_dsrt.py build_ext --inplace
python -c "from rlscore.learner import KronRLS; from dsrt_krls import SSRT; print('KRLS ready')"
```

`rlscore` is required **only** by `dsrt_krls.py`, `dti_eval_metrics.py` and the KRLS experiments. Installation can be problematic because of its compiled dependencies. The optional YAML uses the exact RLScore Git commit identified in the original working environment (`d3970ccce9affd715cd128188dbaebcbe07428ab`); Git and a suitable compiler are required. **RLScore is not needed to reproduce standard DSRT results.** The original, Windows-specific environment export is retained in `docs/environments/` for provenance; it is not the recommended installation file.

**Random Forest benchmarks (optional):** `experiments_tabular_rf.py` and `experiments_functional_rf.py` use **R**, `rpy2` and R's `randomForest` package, not scikit-learn's forest. Install a compatible R/`rpy2` pair separately (the original export records R 4.1.3 and `rpy2` 3.5.6), then install `randomForest` in R. The tabular script hard-codes a Windows `R_HOME` path to R 4.2.1; adapt it to your actual installation. These dependencies are not included in either YAML.

## Datasets

- **Tabular (Sections 5.1–5.2):** some datasets are retrieved from OpenML, while the scripts load others from files in the **current working directory**, such as `abalone.csv`, `ailerons.data` and `housing.csv`/`Book1anto.csv` (the filenames differ across scripts). The `data/tabular/` folder can hold these files, but scripts that use relative filenames must still find them in the current working directory unless their paths are adapted. Supply the exact filenames expected by each loader; first-time OpenML access requires an internet connection.
- **Functional (Section 5.3):** the scripts read `*_D.csv`, `*_X.csv` and `*_y.csv` from **`data/functional/`**, or a directory set with `DSRT_FUNCTIONAL_DATA_DIR`. The names and CSV layouts must match the loaders exactly. The older notebook outputs are preserved in `generated_datasets/` and may require staging/renaming and format verification; see [dataset preparation](docs/dataset_generation.md#using-generated-data-in-the-experiments).
- **Davis / KRLS (Section 5.4):** provide `drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt`, `drug-drug_similarities_2D.txt` and `target-target_similarities_WS.txt` in **`data/krls/`**. These are external inputs, not generated by the experiment scripts.

Preserve dataset attribution and comply with upstream licenses when redistributing files.

## Experiments

Run the following commands from the **repository root**, after activating the relevant environment. The scripts retain their published seeds and optimization parameters; change the dataset selections and experiment options inside the scripts when reproducing additional configurations.

| Experiment | Command | Notes |
|---|---|---|
| Tabular DSRT, full working set | `python esperimenti_no_sparsity_tabular.py lbfgs 0 0` | Five folds; ten initializations per fold |
| Tabular DSRT, prototype selection | `python esperimenti_sparsity_tabular.py lbfgs 0 0` | k-medoids (`kmediod`) or uniform selection; percentage configured in the script |
| Functional DSRT | `python experiments_functional_dsrt.py lbfgs 0 0` | Depth 2; default dataset `energy` |
| Tabular Random Forest | `python experiments_tabular_rf.py` | Requires R, `rpy2` and `randomForest` |
| Functional Random Forest | `python experiments_functional_rf.py` | Requires R, `rpy2` and `randomForest` |
| KRLS B (new drugs) | `python experiments_krls_bc.py DT lbfgs 0 0 1` | `setting_B=True`; drug-routed tree |
| KRLS C (new targets) | Same command after `setting_B=False` | Target-routed tree, transposed response matrix |
| KRLS D (new drugs and targets) | `python experiments_krls_d.py` | Two routed trees; equal-weight averaged predictions |

The KRLS experiments use `my_logistic_dec()` and **disable prototype selection**. In the B/C command, the final argument is `similarity_bool`: `1` uses similarity matrices and `0` converts the DSRT inputs to distances (the standalone KronRLS baseline still uses the original similarities). In the standard DSRT commands, `0` disables clustering initialization. See each script's opening docstring for further parameter details.

## Outputs and reproducibility

Results are CSV files written to the **current working directory**. Tabular and functional scripts record per-run metrics, training time and, where applicable, best-of-start summaries selected by **training** RMSE. KRLS B/C compare DSRT–KRLS with standalone KronRLS; some historical column labels say `mse` although the values are **RMSE**. KRLS D produces four RMSE tables (combined DSRT train/test and KronRLS train/test), each of shape **10 splits × 10 starts**. KronRLS is fitted once per split; its RMSE is duplicated across the ten columns solely for consistent output shapes. The setting-D paper comparison uses the mean of the test RMSE entries, **not** a best-start selection.

Exact results can depend on numerical libraries, compiler, hardware and parallel execution. The YAMLs are reconstructed from the reported working environment and have not been tested by repeating every experiment in a newly created environment.
