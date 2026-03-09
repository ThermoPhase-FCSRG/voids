from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, gmres, spsolve


def _import_pyamg():
    """Import PyAMG lazily so the dependency remains easy to diagnose."""

    try:
        import pyamg
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyAMG preconditioning requires the 'pyamg' package to be installed."
        ) from exc
    return pyamg


def _build_preconditioner(
    A: sparse.csr_matrix,
    *,
    solver_parameters: dict[str, Any] | None,
) -> tuple[Any | None, dict[str, Any]]:
    """Build an optional Krylov preconditioner from solver parameters."""

    parameters = dict(solver_parameters or {})
    name = parameters.get("preconditioner")
    if name is None:
        return None, {}
    if name != "pyamg":
        raise ValueError(f"Unknown preconditioner '{name}'")

    pyamg = _import_pyamg()
    amg_kind = str(parameters.get("pyamg_solver", "smoothed_aggregation"))
    amg_kwargs = parameters.get("pyamg_kwargs", {})
    if not isinstance(amg_kwargs, dict):
        raise ValueError("pyamg_kwargs must be a dictionary")

    matrix = sparse.csr_matrix(A, dtype=float)
    if amg_kind == "smoothed_aggregation":
        hierarchy = pyamg.smoothed_aggregation_solver(matrix, **amg_kwargs)
    elif amg_kind == "rootnode":
        hierarchy = pyamg.rootnode_solver(matrix, **amg_kwargs)
    elif amg_kind == "ruge_stuben":
        hierarchy = pyamg.ruge_stuben_solver(matrix, **amg_kwargs)
    else:
        raise ValueError(
            f"Unknown pyamg_solver '{amg_kind}'. Expected 'smoothed_aggregation', "
            "'rootnode', or 'ruge_stuben'."
        )
    return (
        hierarchy.aspreconditioner(),
        {
            "preconditioner": "pyamg",
            "pyamg_solver": amg_kind,
            "pyamg_levels": int(len(hierarchy.levels)),
            "pyamg_operator_complexity": float(hierarchy.operator_complexity()),
        },
    )


def solve_linear_system(
    A: sparse.csr_matrix,
    b: np.ndarray,
    *,
    method: str = "direct",
    solver_parameters: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a sparse linear system with one of the supported SciPy backends.

    Parameters
    ----------
    A :
        Sparse system matrix.
    b :
        Right-hand-side vector.
    method :
        Solver backend. Supported values are ``"direct"``, ``"cg"``,
        and ``"gmres"``.
    solver_parameters :
        Optional backend-specific solver options. For SciPy Krylov methods this
        maps directly to supported keyword arguments such as ``rtol``,
        ``atol``, ``restart``, and ``maxiter``. Setting
        ``{"preconditioner": "pyamg"}`` attaches a PyAMG preconditioner to
        ``cg`` or ``gmres``.

    Returns
    -------
    numpy.ndarray
        Solution vector.
    dict[str, Any]
        Solver metadata containing the method name and the iterative solver
        status code ``info``.

    Raises
    ------
    ValueError
        If ``method`` is not recognized.
    """

    if method == "direct":
        x = spsolve(A, b)
        return np.asarray(x, dtype=float), {"method": method, "info": 0}
    if method == "cg":
        parameters = dict(solver_parameters or {})
        preconditioner, preconditioner_info = _build_preconditioner(A, solver_parameters=parameters)
        cg_kwargs = {
            key: parameters[key] for key in ("rtol", "atol", "maxiter", "M") if key in parameters
        }
        if preconditioner is not None and "M" not in cg_kwargs:
            cg_kwargs["M"] = preconditioner
        x, info = cg(A, b, **cg_kwargs)
        return np.asarray(x, dtype=float), {
            "method": method,
            "info": int(info),
            **preconditioner_info,
        }
    if method == "gmres":
        parameters = dict(solver_parameters or {})
        preconditioner, preconditioner_info = _build_preconditioner(A, solver_parameters=parameters)
        gmres_kwargs = {
            key: parameters[key]
            for key in ("rtol", "atol", "restart", "maxiter", "M")
            if key in parameters
        }
        if preconditioner is not None and "M" not in gmres_kwargs:
            gmres_kwargs["M"] = preconditioner
        x, info = gmres(A, b, **gmres_kwargs)
        return np.asarray(x, dtype=float), {
            "method": method,
            "info": int(info),
            **preconditioner_info,
        }
    raise ValueError(f"Unknown solver method '{method}'")
