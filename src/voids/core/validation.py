from __future__ import annotations

import warnings

import numpy as np

from voids.core.network import Network


RECOMMENDED_PORE_FIELDS = ("volume",)
RECOMMENDED_THROAT_FIELDS = ("volume", "length")


def assert_finite(name: str, arr: np.ndarray) -> None:
    """Validate that an array contains only finite values.

    Parameters
    ----------
    name : str
        Descriptive name of the array (for error messages).
    arr : np.ndarray
        Array to validate.

    Raises
    ------
    ValueError
        If the array contains non-finite values (NaN or inf).
    """
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")


def validate_network(net: Network, *, allow_parallel_throats: bool = True) -> None:
    tc = net.throat_conns
    if tc.ndim != 2 or tc.shape[1] != 2:
        raise ValueError("throat_conns must have shape (Nt, 2)")
    if net.pore_coords.ndim != 2 or net.pore_coords.shape[1] != 3:
        raise ValueError("pore_coords must have shape (Np, 3)")
    if np.isnan(net.pore_coords).any():
        raise ValueError("pore_coords contains NaNs")
    if (tc < 0).any() or (tc >= net.Np).any():
        raise ValueError("throat_conns contains out-of-range pore indices")
    if (tc[:, 0] == tc[:, 1]).any():
        raise ValueError("self-loop throats are not allowed in v0.1")

    if not allow_parallel_throats:
        edges = np.sort(tc, axis=1)
        uniq = np.unique(edges, axis=0)
        if uniq.shape[0] != tc.shape[0]:
            raise ValueError("parallel throats found")
    else:
        edges = np.sort(tc, axis=1)
        uniq = np.unique(edges, axis=0)
        if uniq.shape[0] != tc.shape[0]:
            warnings.warn("parallel throats detected", RuntimeWarning, stacklevel=2)

    for k, arr in net.pore.items():
        if arr.shape[0] != net.Np:
            raise ValueError(f"Pore field '{k}' has wrong first dimension")
        if np.issubdtype(arr.dtype, np.number):
            if np.isnan(arr).any():
                raise ValueError(f"Pore field '{k}' contains NaNs")
            if (
                k in {"volume", "area", "diameter_inscribed", "radius_inscribed", "length"}
                and (arr < 0).any()
            ):
                raise ValueError(f"Pore field '{k}' contains negative values")
    for k, arr in net.throat.items():
        if arr.shape[0] != net.Nt:
            raise ValueError(f"Throat field '{k}' has wrong first dimension")
        if np.issubdtype(arr.dtype, np.number):
            if np.isnan(arr).any():
                raise ValueError(f"Throat field '{k}' contains NaNs")
            if (
                k in {"volume", "area", "diameter_inscribed", "radius_inscribed"}
                and (arr < 0).any()
            ):
                raise ValueError(f"Throat field '{k}' contains negative values")
            if k in {"length", "core_length", "pore1_length", "pore2_length"} and (arr <= 0).any():
                raise ValueError(f"Throat field '{k}' contains nonpositive values")

    for label, mask in net.pore_labels.items():
        if mask.shape != (net.Np,):
            raise ValueError(f"Pore label '{label}' has wrong shape")
    for label, mask in net.throat_labels.items():
        if mask.shape != (net.Nt,):
            raise ValueError(f"Throat label '{label}' has wrong shape")

    if net.sample is not None:
        # Only sanity-check if provided
        try:
            bv = net.sample.resolved_bulk_volume()
            if bv <= 0:
                raise ValueError("sample bulk volume must be positive")
        except ValueError:
            pass

    for name in RECOMMENDED_PORE_FIELDS:
        if name not in net.pore:
            warnings.warn(f"Recommended pore field missing: '{name}'", RuntimeWarning, stacklevel=2)
    for name in RECOMMENDED_THROAT_FIELDS:
        if name not in net.throat:
            warnings.warn(
                f"Recommended throat field missing: '{name}'", RuntimeWarning, stacklevel=2
            )
