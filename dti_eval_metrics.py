"""Evaluation utilities for the Davis drug–target KRLS experiments.

Used by the DSRT–KRLS experiments in settings B and C (Section 5.4 of
``Soft Regression Trees for Distance Data``). The affinity response is a
matrix whose entries correspond to drug–target interactions.

The experimental script uses ``evaluate_setting`` for both settings, after
orienting the response matrix so that its rows correspond to the entity
used for tree routing. ``evaluate_setting_C`` is an additional, optional
metric helper with an entity-wise concordance index; it is not called by
the B/C experimental script.

All metric formulas, handling of non-finite targets, and result keys are
preserved from the original implementation.
"""

from typing import Dict, Optional
import numpy as np
from scipy.stats import spearmanr
from rlscore.measure import cindex

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the RMSE over all finite reference affinity values.

    ``np.isfinite(y_true)`` defines the evaluation mask. Predictions are
    indexed with the same mask, without an additional finiteness check.
    Return NaN if no reference values are finite.
    """
    mask = np.isfinite(y_true)
    if not np.any(mask):
        return np.nan
    err = y_pred[mask] - y_true[mask]
    return float(np.sqrt(np.mean(err**2)))


def _spearman_per_entity(values_true: np.ndarray, values_pred: np.ndarray) -> float:
    """Compute the mean row-wise Spearman rank correlation.

    Parameters
    ----------
    values_true, values_pred : numpy.ndarray
        Two-dimensional interaction matrices shaped
        ``(n_entities, n_counterpart_entities)``. Each row is evaluated
        independently.

    Returns
    -------
    float
        Arithmetic mean of the finite per-row correlations, or NaN if
        none can be computed. Rows with fewer than two finite reference
        values or constant reference values are skipped.
    """
    rhos = []
    for i in range(values_true.shape[0]):
        y_t = values_true[i]
        y_p = values_pred[i]
        mask = np.isfinite(y_t)
        if np.sum(mask) < 2:
            continue
        # Spearman correlation is undefined for constant reference values.
        if np.nanmax(y_t[mask]) == np.nanmin(y_t[mask]):
            continue
        rho, _ = spearmanr(y_t[mask], y_p[mask])
        if np.isfinite(rho):
            rhos.append(float(rho))
    if len(rhos) == 0:
        return np.nan
    return float(np.mean(rhos))


def _c_index_per_entity(values_true: np.ndarray, values_pred: np.ndarray) -> float:
    """Compute the mean row-wise concordance index across entities.

    Each row is an entity, and its columns are interaction partners.
    The internal scorer enumerates distinct upper-triangular pairs for
    which the true responses differ. Equal predictions count as 0.5;
    ties in the true responses are excluded. Rows with no admissible
    pairs are omitted. Computation is quadratic in the row width.

    This is distinct from the external ``rlscore.measure.cindex`` used
    by ``evaluate_setting`` in the paper's B/C experiment script.
    """
    def cidx_row(y_t: np.ndarray, y_p: np.ndarray) -> Optional[float]:
        mask = np.isfinite(y_t)
        y_t = y_t[mask]
        y_p = y_p[mask]
        n = y_t.size
        if n < 2:
            return None

        yt_i = y_t[:, None]
        yt_j = y_t[None, :]
        yp_i = y_p[:, None]
        yp_j = y_p[None, :]

        # Keep each unordered pair once and exclude self-comparisons.
        triu = np.triu(np.ones((n, n), dtype=bool), k=1)

        # Only pairs with distinct reference values are comparable.
        perm = (yt_i != yt_j) & triu
        m = np.sum(perm)
        if m == 0:
            return None

        concordant = ((yt_i > yt_j) & (yp_i > yp_j) | (yt_i < yt_j) & (yp_i < yp_j)) & perm
        ties_pred  = (yp_i == yp_j) & perm

        score = np.sum(concordant) + 0.5 * np.sum(ties_pred)
        return float(score / m)

    scores = []
    for i in range(values_true.shape[0]):
        s = cidx_row(values_true[i], values_pred[i])
        if s is not None and np.isfinite(s):
            scores.append(s)
    if len(scores) == 0:
        return np.nan
    return float(np.mean(scores))

def evaluate_setting(Y_true: np.ndarray,
                        Y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate an interaction matrix in the orientation supplied by the caller.

    Used in both Davis settings B and C. The experiment selects the
    held-out entity rows before calling this function; for setting C,
    it first transposes the original drug–target matrix. No splitting
    or transposition occurs inside this helper.

    Parameters
    ----------
    Y_true, Y_pred : numpy.ndarray
        Two-dimensional matrices of true and predicted affinities with
        matching shapes. Missing reference values may be represented
        by NaN; RMSE and Spearman use the finite-reference mask.

    Returns
    -------
    dict[str, float]
        ``RMSE`` (global root mean squared error), ``Spearman_macro``
        (mean within-row rank correlation), and ``C_index`` (the result
        of passing both full arrays directly to RLScore's cindex).
    """
    return {
        "RMSE": _rmse(Y_true, Y_pred),
        "Spearman_macro": _spearman_per_entity(Y_true, Y_pred),
        "C_index": cindex(Y_true, Y_pred),
    }

def evaluate_setting_C(Y_true: np.ndarray,
                       Y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate a matrix using a macro-averaged concordance index.

    This optional helper does not perform any column selection or
    transposition. To measure rows as targets, the caller must supply
    the target-by-drug orientation, as in the main script's setting C.

    Returns the same RMSE and row-wise Spearman statistic as
    ``evaluate_setting``, but uses the internal row-wise concordance
    computation and the distinct key ``C_index_macro``.
    """
    return {
        "RMSE": _rmse(Y_true, Y_pred),
        "Spearman_macro": _spearman_per_entity(Y_true, Y_pred),
        "C_index_macro": _c_index_per_entity(Y_true, Y_pred),
    }
