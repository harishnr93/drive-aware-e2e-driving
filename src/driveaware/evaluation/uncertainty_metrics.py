"""Metrics for trajectory uncertainty: ranking quality, sparsification, NLL and coverage."""

from __future__ import annotations

import math

import numpy as np


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))


def sparsification(
    errors: np.ndarray, uncertainty: np.ndarray, fractions: np.ndarray | None = None
) -> dict:
    """Mean error of the samples kept after removing the most uncertain fraction.

    The oracle removes by true error. AUSE = area between the method's curve and the oracle's
    (both normalised by the full-set error), lower is better; 0 = perfect ranking.
    """
    fr = np.linspace(0.0, 0.98, 50) if fractions is None else fractions
    e = np.asarray(errors, dtype=np.float64)
    n = len(e)
    by_u = e[np.argsort(-np.asarray(uncertainty))]
    by_e = np.sort(e)[::-1]
    base = e.mean()
    curve_u = np.array([by_u[int(f * n) :].mean() for f in fr]) / base
    curve_o = np.array([by_e[int(f * n) :].mean() for f in fr]) / base
    return {
        "fractions": fr,
        "method": curve_u,
        "oracle": curve_o,
        "ause": float(np.trapezoid(curve_u - curve_o, fr)),
    }


def gaussian_nll(errors_xy: np.ndarray, var: np.ndarray, min_var: float = 1e-6) -> float:
    """Mean per-coordinate Gaussian NLL of residuals [.., 2] under variances [.., 2]."""
    v = np.maximum(var, min_var)
    return float(np.mean(0.5 * (errors_xy**2 / v + np.log(2 * math.pi * v))))


def coverage(errors_xy: np.ndarray, var: np.ndarray, k: float) -> float:
    """Fraction of coordinates whose |residual| <= k * sigma (Gaussian: 1 -> 68.3%, 2 -> 95.4%)."""
    return float(np.mean(np.abs(errors_xy) <= k * np.sqrt(np.maximum(var, 1e-12))))


def fit_sigma_scale(errors_xy: np.ndarray, var: np.ndarray, target: float = 0.6827) -> float:
    """Scalar s so that |e| <= s * sigma holds for ``target`` of the coordinates (on val)."""
    z = np.abs(errors_xy) / np.sqrt(np.maximum(var, 1e-12))
    return float(np.quantile(z, target))


def binned_uncertainty_vs_error(
    uncertainty: np.ndarray, errors: np.ndarray, bins: int = 10
) -> dict:
    """Equal-mass bins of predicted uncertainty: mean predicted vs mean actual error."""
    order = np.argsort(uncertainty)
    chunks = np.array_split(order, bins)
    return {
        "uncertainty": [float(np.mean(uncertainty[c])) for c in chunks],
        "error": [float(np.mean(errors[c])) for c in chunks],
    }
