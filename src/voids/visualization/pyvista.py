from __future__ import annotations

from typing import Any

import numpy as np

from voids.core.network import Network


def _require_pyvista():
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyVista is not installed. Use the 'default' or 'test' pixi environment or install pyvista."
        ) from exc
    return pv


def _line_cells_from_conns(conns: np.ndarray) -> np.ndarray:
    conns = np.asarray(conns, dtype=np.int64)
    if conns.ndim != 2 or conns.shape[1] != 2:
        raise ValueError("throat connections must have shape (Nt, 2)")
    cells = np.empty((conns.shape[0], 3), dtype=np.int64)
    cells[:, 0] = 2
    cells[:, 1:] = conns
    return cells.ravel()


def network_to_pyvista_polydata(
    net: Network,
    *,
    point_scalars: str | np.ndarray | None = None,
    cell_scalars: str | np.ndarray | None = None,
    include_all_numeric_fields: bool = False,
):
    """Return a ``pyvista.PolyData`` representing pores (points) and throats (lines).

    Parameters
    ----------
    point_scalars, cell_scalars
        Either a field name (e.g. ``"volume"``) from ``net.pore``/``net.throat`` or an array.
    include_all_numeric_fields
        If True, attach all 1D numeric pore/throat arrays as point/cell data when lengths match.
    """
    pv = _require_pyvista()

    points = np.asarray(net.pore_coords, dtype=float)
    line_cells = _line_cells_from_conns(net.throat_conns)
    poly = pv.PolyData(points, lines=line_cells)

    # Canonical identifiers for easier debugging and selections
    poly.point_data["pore.id"] = np.arange(net.Np, dtype=np.int64)
    poly.cell_data["throat.id"] = np.arange(net.Nt, dtype=np.int64)

    if include_all_numeric_fields:
        for k, v in net.pore.items():
            a = np.asarray(v)
            if a.ndim == 1 and a.shape[0] == net.Np and np.issubdtype(a.dtype, np.number):
                poly.point_data[f"pore.{k}"] = a
        for k, v in net.throat.items():
            a = np.asarray(v)
            if a.ndim == 1 and a.shape[0] == net.Nt and np.issubdtype(a.dtype, np.number):
                poly.cell_data[f"throat.{k}"] = a

    if isinstance(point_scalars, str):
        if point_scalars not in net.pore:
            raise KeyError(f"Missing pore field '{point_scalars}'")
        poly.point_data["pore.scalar"] = np.asarray(net.pore[point_scalars])
        poly.set_active_scalars("pore.scalar", preference="point")
    elif point_scalars is not None:
        arr = np.asarray(point_scalars)
        if arr.shape != (net.Np,):
            raise ValueError("point_scalars array must have shape (Np,)")
        poly.point_data["pore.scalar"] = arr
        poly.set_active_scalars("pore.scalar", preference="point")

    if isinstance(cell_scalars, str):
        if cell_scalars not in net.throat:
            raise KeyError(f"Missing throat field '{cell_scalars}'")
        poly.cell_data["throat.scalar"] = np.asarray(net.throat[cell_scalars])
    elif cell_scalars is not None:
        arr = np.asarray(cell_scalars)
        if arr.shape != (net.Nt,):
            raise ValueError("cell_scalars array must have shape (Nt,)")
        poly.cell_data["throat.scalar"] = arr

    return poly


def plot_network_pyvista(
    net: Network,
    *,
    point_scalars: str | np.ndarray | None = None,
    cell_scalars: str | np.ndarray | None = None,
    show_points: bool = True,
    show_lines: bool = True,
    line_width: float = 3.0,
    point_size: float = 9.0,
    render_tubes: bool = False,
    tube_radius: float | None = None,
    off_screen: bool = False,
    screenshot: str | None = None,
    show_axes: bool = True,
    notebook: bool | None = None,
    **add_mesh_kwargs: Any,
):
    """Plot the pore network using PyVista and return ``(plotter, polydata)``.

    The function is intentionally lightweight and works as an optional convenience wrapper.
    Use ``network_to_pyvista_polydata`` for advanced visualization workflows.
    """
    pv = _require_pyvista()
    poly = network_to_pyvista_polydata(
        net,
        point_scalars=point_scalars,
        cell_scalars=cell_scalars,
        include_all_numeric_fields=True,
    )

    if notebook is None:
        try:
            notebook = False
        except Exception:
            notebook = False
    pl = pv.Plotter(off_screen=off_screen, notebook=notebook)

    # Prefer explicit throat/cell scalars for lines, then fall back to pore/point scalars.
    line_scalars_name = "throat.scalar" if "throat.scalar" in poly.cell_data else None
    point_scalars_name = "pore.scalar" if "pore.scalar" in poly.point_data else None
    if line_scalars_name is None and point_scalars_name is not None:
        line_scalars_name = point_scalars_name

    if show_lines and net.Nt > 0:
        line_mesh = poly
        if render_tubes:
            kwargs = {}
            if tube_radius is not None:
                kwargs["radius"] = float(tube_radius)
            try:
                line_mesh = poly.tube(**kwargs)
            except Exception:
                line_mesh = poly
        pl.add_mesh(
            line_mesh,
            scalars=line_scalars_name,
            line_width=line_width,
            render_lines_as_tubes=not render_tubes,
            show_scalar_bar=(line_scalars_name is not None),
            **add_mesh_kwargs,
        )

    if show_points and net.Np > 0:
        pl.add_mesh(
            poly,
            style="points",
            point_size=point_size,
            render_points_as_spheres=True,
            scalars=point_scalars_name,
            show_scalar_bar=(point_scalars_name is not None and not show_lines),
        )

    if show_axes:
        pl.add_axes()

    if screenshot is not None:
        pl.show(auto_close=False)
        pl.screenshot(screenshot)
    return pl, poly
