"""Population Stability Index (PSI) for distribution drift.

Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from `expected`."""
    e, a = expected.dropna(), actual.dropna()
    if e.nunique() <= bins:  # discrete / binary: compare value proportions directly
        values = np.union1d(e.unique(), a.unique())
        pe = e.value_counts(normalize=True).reindex(values, fill_value=0).to_numpy()
        pa = a.value_counts(normalize=True).reindex(values, fill_value=0).to_numpy()
        pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
        return float(np.sum((pa - pe) * np.log(pa / pe)))
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_flag(value: float) -> str:
    return "stable" if value < 0.10 else ("moderate" if value < 0.25 else "major")
