# Soft Regression Trees for Dissimilarity Data

## Overview

This repository contains material supporting the paper **“Soft Regression Trees for Dissimilarity Data.”**

The current public release focuses on the generation and storage of the datasets used in the functional-data experiments. It includes:

- the notebook used to generate the Swiss-roll benchmark;
- the notebook used to prepare the six time-series regression datasets;
- the generated CSV files used in the experiments;
- the complete Conda environment used for the numerical experiments in the paper.

The implementation of the Soft Distance-based Regression Tree (SDRT), the training algorithm, and the remaining experimental code are not included in the current release.

## Reference paper

**Soft Regression Trees for Dissimilarity Data**

Paper link: *to be added*

Full citation details will be added when available.

## Repository structure

```text
.
├── README.md
├── experiments_environment.yml
├── dataset_generation/
│   ├── swiss_roll_dataset_generation.ipynb
│   └── time_series_dataset_generation.ipynb
└── generated_datasets/
    ├── swiss_roll/
    │   ├── swissroll_isomap_D.csv
    │   ├── swissroll_isomap_X.csv
    │   ├── swissroll_isomap_y.csv
    │   └── swissroll_isomap_k_diagnostics.csv
    └── time_series/
        ├── covid/
        ├── flood1/
        ├── flood2/
        ├── flood3/
        ├── energy/
        └── iron/
```

The two main directories have distinct roles:

- `dataset_generation/` contains the code used to generate or preprocess the datasets;
- `generated_datasets/` contains the CSV files produced by the notebooks and used in the experiments.

## Experimental environment

The file `experiments_environment.yml` is a cleaned export of the complete Conda environment used for the numerical experiments in the paper.

With respect to the original export, only the following metadata were changed:

- the environment name was changed to `soft_dissimilarity_experiments`;
- the machine-specific local `prefix` was removed.

The package list, versions, build identifiers, channels, and pip dependencies were otherwise left unchanged.

Because the environment was exported on Windows and contains platform-specific packages and build identifiers, it is mainly intended to document and reproduce the original experimental software stack on a compatible Windows system.

The environment can be created with:

```bash
conda env create -f experiments_environment.yml
conda activate soft_dissimilarity_experiments
```

## Dataset-generation environment

The dataset-generation notebooks were developed and executed in Google Colab, separately from the Conda environment used for the complete experimental pipeline.

Therefore, `experiments_environment.yml` should not be interpreted as an exact export of the Colab runtime used to generate the datasets.

The notebooks contain the code required to generate the corresponding outputs. The time-series notebook installs its main additional dependencies directly in Colab:

```python
!pip install aeon tslearn
```

The Swiss-roll benchmark described in the paper was generated using `scikit-learn` version `1.6.1`.

## Swiss-roll benchmark

The Swiss-roll benchmark is generated using `sklearn.datasets.make_swiss_roll` with the following settings:

- number of observations: `1000`;
- Gaussian noise level: `0.05`;
- random seed: `2020`;
- candidate neighborhood sizes: `k ∈ {6, 8, ..., 30}`.

For each candidate value of `k`, the notebook:

1. constructs a weighted k-nearest-neighbor graph in the original three-dimensional space;
2. symmetrizes the graph using the union of the directed nearest-neighbor relations;
3. discards values of `k` producing a disconnected graph;
4. computes all-pairs shortest-path distances;
5. obtains a two-dimensional Classical MDS embedding;
6. selects the value of `k` minimizing the residual variance

$$
\mathrm{RV} = 1 - \rho(D,D_X)^2,
$$

where `D` is the geodesic distance matrix, `D_X` is the Euclidean distance matrix in the Classical MDS embedding, and `ρ` is the Pearson correlation computed over the upper-triangular distance entries.

The response variable used in the paper is the intrinsic unrolled coordinate returned by `make_swiss_roll`, namely `y_t = t`.

The generated files are stored in:

```text
generated_datasets/swiss_roll/
```

and include:

- `swissroll_isomap_D.csv`: pairwise geodesic distance matrix used by SDRT;
- `swissroll_isomap_X.csv`: two-dimensional Classical MDS representation used as tabular input for Random Forest;
- `swissroll_isomap_y.csv`: response variables, including `y_t`;
- `swissroll_isomap_k_diagnostics.csv`: connectivity and residual-variance values for the candidate neighborhood sizes.

## Time-series datasets

The following fixed-length regression datasets are considered:

| Dataset | Abbreviation |
|---|---|
| `Covid3Month` | `covid` |
| `FloodModeling1` | `flood1` |
| `FloodModeling2` | `flood2` |
| `FloodModeling3` | `flood3` |
| `AppliancesEnergy` | `energy` |
| `IronConcentration` | `iron` |

For each dataset, the notebook:

1. loads the training split with `aeon.datasets.load_regression(name, split="train")`;
2. computes the complete pairwise Dynamic Time Warping distance matrix;
3. flattens each fixed-length time series into a feature vector for Random Forest;
4. saves the distance matrix, feature matrix, and response vector as CSV files.

For multivariate time series, all channels and time points are concatenated. For univariate time series, the original time samples are used directly.

The generated files are organized as:

```text
generated_datasets/time_series/<abbreviation>/
```

with three CSV files for each dataset:

```text
<abbreviation>_D.csv
<abbreviation>_X.csv
<abbreviation>_y.csv
```

where:

- `*_D.csv` is the complete pairwise DTW distance matrix used by SDRT;
- `*_X.csv` is the fixed-dimensional feature matrix used by Random Forest;
- `*_y.csv` is the continuous response vector.

The scaling procedures used in the numerical experiments belong to the downstream experimental pipeline and are not applied by the dataset-generation notebooks.

## Notes on repository size

Pairwise distance matrices can be large. Before committing the generated CSV files, verify that they remain within the file-size limits of the hosting platform. For larger files, Git LFS or an external data archive may be more appropriate.
