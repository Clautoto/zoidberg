"""Unit tests for components used to generate FCI grids.

These tests were written by an AI assistant (Claude). They are marked with
the ``ai_generated`` pytest marker, so they can be selected with
``pytest -m ai_generated`` or excluded with ``pytest -m "not ai_generated"``.
"""

import numpy as np
import pytest
from matplotlib.path import Path
from scipy.sparse import csr_matrix

from . import boundary, field, grid, poloidal_grid, rzline, weights, zoidberg

pytestmark = pytest.mark.ai_generated


def make_shift_maps(nx, ny, nz, mxg, dx, dz):
    """Maps where every field line is shifted by (dx, dz) in index space
    per parallel step: forward to (x + dx, z + dz), backward to (x - dx, z - dz)"""
    xind, _, zind = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    return {
        "R": xind.astype(float),
        "MXG": mxg,
        "forward_xt_prime": xind + dx,
        "forward_zt_prime": zind + dz,
        "backward_xt_prime": xind - dx,
        "backward_zt_prime": zind - dz,
    }


def operator_matrix(ops, direction):
    """Assemble the CSR destination operator for a direction"""
    return csr_matrix(
        (
            ops[f"{direction}_weights"],
            ops[f"{direction}_columns"],
            ops[f"{direction}_rows"],
        ),
        shape=(ops[f"n_{direction}_legs"], ops["total_cells"]),
    )


def cell_vector(ops, func, shape):
    """Evaluate func(x, y, z) on all radial interior and X-boundary cells"""
    xind, yind, zind = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    values = np.zeros(ops["total_cells"])
    values[ops["cell_number"]] = func(xind, yind, zind)
    return values


def test_catmull_rom_weights():
    offsets = np.array([-1.0, 0.0, 1.0, 2.0])

    # Interpolating at the stencil points returns that point exactly
    assert np.allclose(weights.catmull_rom_weights(0.0), [0, 1, 0, 0])
    assert np.allclose(weights.catmull_rom_weights(1.0), [0, 0, 1, 0])

    for u in np.linspace(0.0, 1.0, 11):
        w = weights.catmull_rom_weights(u)
        # Weights reproduce polynomials up to quadratic
        assert np.isclose(np.sum(w), 1.0)
        assert np.isclose(w @ offsets, u)
        assert np.isclose(w @ offsets**2, u**2)


def test_parallel_operator_identity_maps():
    nx, ny, nz, mxg = 8, 3, 6, 2
    maps = make_shift_maps(nx, ny, nz, mxg, 0.0, 0.0)
    ops = weights.calculate_parallel_map_operators(maps)

    # All field lines stay inside the domain: one interior leg per evolving cell
    n_evolving = (nx - 2 * mxg) * ny * nz
    assert ops["total_cells"] == nx * ny * nz
    assert ops["n_forward_legs"] == n_evolving
    assert ops["n_backward_legs"] == n_evolving
    assert np.all(ops["forward_cell_number"] == -1)
    assert np.allclose(ops["forward_leg_weights"], 1.0)

    rng = np.random.default_rng(42)
    values = rng.normal(size=ops["total_cells"])
    cell = ops["cell_number"]
    for direction, offset in (("forward", 1), ("backward", -1)):
        result = operator_matrix(ops, direction) @ values
        legs = ops[f"{direction}_leg_interior_number"]
        for x in range(mxg, nx - mxg):
            for y in range(ny):
                for z in range(nz):
                    # Value of the same (x, z) cell on the next / previous plane
                    expected = values[cell[x, (y + offset) % ny, z]]
                    assert np.isclose(result[legs[x, y, z]], expected)


def test_parallel_operator_fractional_shift():
    nx, ny, nz, mxg = 14, 3, 12, 2
    dx, dz = 0.3, 0.25
    maps = make_shift_maps(nx, ny, nz, mxg, dx, dz)
    ops = weights.calculate_parallel_map_operators(maps)

    def func(x, y, z):
        # Quadratic in x and z, so reproduced exactly by Catmull-Rom
        return 1.0 + 0.5 * x - 0.2 * x**2 - 0.3 * z + 0.1 * z**2 + y

    values = cell_vector(ops, func, (nx, ny, nz))

    for direction, sign in (("forward", 1), ("backward", -1)):
        matrix = operator_matrix(ops, direction)
        # Interpolation weights of each leg sum to one
        assert np.allclose(np.asarray(matrix.sum(axis=1)).ravel(), 1.0)

        result = matrix @ values
        legs = ops[f"{direction}_leg_interior_number"]
        checked = 0
        for x in range(mxg, nx - mxg):
            for y in range(ny):
                for z in range(nz):
                    xt, yt, zt = x + sign * dx, y + sign, z + sign * dz
                    # Only check stencils that avoid the X boundaries and
                    # the periodic wrap in y and z
                    x_base, z_base = int(np.floor(xt)), int(np.floor(zt))
                    if not (mxg <= x_base - 1 and x_base + 2 < nx - mxg):
                        continue
                    if not (0 <= z_base - 1 and z_base + 2 < nz):
                        continue
                    if not 0 <= yt < ny:
                        continue
                    assert np.isclose(result[legs[x, y, z]], func(xt, yt, zt))
                    checked += 1
        assert checked > 0


def test_parallel_operator_boundary_legs():
    nx, ny, nz, mxg = 8, 2, 4, 2
    maps = make_shift_maps(nx, ny, nz, mxg, 0.0, 0.0)
    # Field lines starting at the innermost evolving x leave the domain
    # in the forward direction
    maps["forward_xt_prime"][mxg, :, :] = -1
    ops = weights.calculate_parallel_map_operators(maps)

    n_evolving = (nx - 2 * mxg) * ny * nz
    n_hits = ny * nz
    # One forward parallel boundary cell per field line hitting the boundary
    assert ops["total_cells"] == nx * ny * nz + n_hits
    assert np.all(ops["forward_cell_number"][mxg] >= nx * ny * nz)
    assert np.all(ops["forward_cell_number"][mxg + 1 :] == -1)
    assert np.all(ops["backward_cell_number"] == -1)
    # Each evolving cell still has exactly one forward leg
    assert ops["n_forward_legs"] == n_evolving

    matrix = operator_matrix(ops, "forward").toarray()
    for y in range(ny):
        for z in range(nz):
            # Boundary legs pick out their parallel boundary cell
            assert ops["forward_leg_interior_number"][mxg, y, z] == -1
            row = matrix[ops["forward_leg_boundary_number"][mxg, y, z]]
            col = ops["forward_cell_number"][mxg, y, z]
            assert row[col] == 1.0
            assert np.count_nonzero(row) == 1


def test_rectangular_boundary():
    bndry = boundary.RectangularBoundaryXZ(xmin=-1.0, xmax=2.0, zmin=0.0, zmax=1.0)
    x = np.array([0.0, -1.5, 2.5, 0.5, 0.5, 1.9])
    z = np.array([0.5, 0.5, 0.5, -0.1, 1.2, 0.9])
    expected = [False, True, True, True, True, False]
    assert np.array_equal(bndry.outside(x, 0.0, z), expected)

    # NoBoundary never marks points as outside, for any shape of input
    x2d = np.full((3, 4), 1e6)
    assert not np.any(boundary.NoBoundary().outside(x2d, 0.0, x2d))


def test_polygon_boundary_concave():
    # L-shaped (concave) polygon
    xarr = np.array([0.0, 2.0, 2.0, 1.0, 1.0, 0.0])
    zarr = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    bndry = boundary.PolygonBoundaryXZ(xarr, zarr)

    rng = np.random.default_rng(1)
    x = rng.uniform(-0.5, 2.5, (20, 30))
    z = rng.uniform(-0.5, 2.5, (20, 30))

    inside = Path(np.column_stack((xarr, zarr))).contains_points(
        np.column_stack((x.ravel(), z.ravel()))
    )
    outside = bndry.outside(x, 0.0, z)
    assert outside.shape == x.shape
    assert np.array_equal(outside.ravel(), ~inside)


def test_rectangular_poloidal_grid():
    nx, nz, mxg = 9, 6, 2
    Lx, Lz, Rcentre, Zcentre = 2.0, 3.0, 5.0, -1.0
    pol = poloidal_grid.RectangularPoloidalGrid(
        nx, nz, Lx, Lz, Rcentre=Rcentre, Zcentre=Zcentre, MXG=mxg
    )

    dR = Lx / (nx - 2 * mxg)
    dZ = Lz / nz
    assert pol.R.shape == (nx, nz)
    assert np.allclose(np.diff(pol.R, axis=0), dR)
    assert np.allclose(np.diff(pol.Z, axis=1), dZ)
    # The domain is centred, with the radial boundaries half-way
    # between the guard cells and the first/last evolving cells
    assert np.isclose(np.mean(pol.R), Rcentre)
    assert np.isclose(np.mean(pol.Z), Zcentre)
    R_inner, _ = pol.getCoordinate(mxg - 0.5, 0.0)
    R_outer, _ = pol.getCoordinate(nx - mxg - 0.5, 0.0)
    assert np.isclose(R_inner, Rcentre - Lx / 2)
    assert np.isclose(R_outer, Rcentre + Lx / 2)

    # findIndex is the inverse of getCoordinate, including fractional indices
    rng = np.random.default_rng(3)
    xind = rng.uniform(0, nx - 1, 50)
    zind = rng.uniform(0, nz - 1, 50)
    xfound, zfound = pol.findIndex(*pol.getCoordinate(xind, zind))
    assert np.allclose(xfound, xind)
    assert np.allclose(zfound, zind)

    # First derivatives with respect to index
    dRdx, dZdx = pol.getCoordinate(xind, zind, dx=1)
    dRdz, dZdz = pol.getCoordinate(xind, zind, dz=1)
    assert np.allclose(dRdx, dR) and np.allclose(dZdx, 0.0)
    assert np.allclose(dRdz, 0.0) and np.allclose(dZdz, dZ)

    metric = pol.metric()
    assert np.isclose(metric["dx"], dR) and np.isclose(metric["dz"], dZ)
    assert metric["g_xx"] == metric["g_zz"] == 1.0 and metric["g_xz"] == 0.0


def test_grid_annulus_concentric_circles():
    nx, nz = 5, 16
    inner = rzline.circle(R0=3.0, r=1.0, n=64)
    outer = rzline.circle(R0=3.0, r=2.0, n=64)
    pol = poloidal_grid.grid_annulus(inner, outer, nx, nz, show=False)

    assert isinstance(pol, poloidal_grid.StructuredPoloidalGrid)
    assert pol.R.shape == (nx, nz)

    # Radial lines are straight between the boundaries: the radius is linear
    # in x index, and the angle is equally spaced in z index
    radius = np.hypot(pol.R - 3.0, pol.Z)
    expected_radius = np.linspace(1.0, 2.0, nx)[:, None]
    assert np.allclose(radius, expected_radius, atol=1e-4)

    angle = np.arctan2(pol.Z, pol.R - 3.0)
    expected_angle = np.linspace(0.0, 2 * np.pi, nz, endpoint=False)[None, :]
    dangle = np.angle(np.exp(1j * (angle - expected_angle)))
    assert np.allclose(dangle, 0.0, atol=1e-4)


def test_structured_poloidal_grid_metric():
    nx, nz = 9, 32
    inner = rzline.circle(R0=0.0, r=1.0, n=64)
    outer = rzline.circle(R0=0.0, r=2.0, n=64)
    pol = poloidal_grid.grid_annulus(inner, outer, nx, nz, show=False)
    metric = pol.metric()

    # Covariant and contravariant metrics are inverses of each other
    g = np.array([[metric["g_xx"], metric["g_xz"]], [metric["g_xz"], metric["g_zz"]]])
    ginv = np.array([[metric["gxx"], metric["gxz"]], [metric["gxz"], metric["gzz"]]])
    product = np.einsum("ij...,jk...->ik...", g, ginv)
    assert np.allclose(product, np.eye(2)[:, :, None, None])
    assert np.all(metric["g_xx"] > 0) and np.all(metric["g_zz"] > 0)

    # Integrating the area element over the grid gives the annulus area.
    # x is not periodic (trapezoidal rule), z is periodic.
    area_element = (
        np.sqrt(metric["g_xx"] * metric["g_zz"] - metric["g_xz"] ** 2)
        * metric["dx"]
        * metric["dz"]
    )
    wx = np.ones(nx)
    wx[[0, -1]] = 0.5
    area = np.sum(wx[:, None] * area_element)
    assert np.isclose(area, np.pi * (2.0**2 - 1.0**2), rtol=1e-4)


def test_grid_metric_and_get_metric_cylindrical():
    nx, ny, nz = 6, 8, 5
    Ly = 2 * np.pi
    rect = grid.rectangular_grid(
        nx, ny, nz, Lx=0.5, Ly=Ly, Lz=0.4, xcentre=0.0, yperiodic=True
    )

    metric = rect.metric()
    assert metric["dy"].shape == (nx, ny, nz)
    assert np.allclose(metric["dy"], Ly / ny)
    assert np.allclose(metric["dx"], 0.5 / (nx - 4))
    assert np.allclose(metric["dz"], 0.4 / nz)
    # Cartesian y metric before any field is taken into account
    assert np.allclose(metric["g_yy"], 1.0)

    # In cylindrical coordinates (y is the toroidal angle), g_yy is
    # R^2 (B / B_y)^2 so that sqrt(g_yy) dy is the parallel length
    Rmaj, By, Bz = 4.0, 1.0, 0.5
    curved = field.CurvedSlab(By=By, Bz=Bz, Bzprime=0.0, Rmaj=Rmaj)
    rect = grid.rectangular_grid(nx, ny, nz, Lx=0.5, Ly=Ly, Lz=0.4, yperiodic=True)
    metric, _ = zoidberg.get_metric(rect, curved)

    B = np.hypot(By, Bz)
    assert np.allclose(metric["B"], B)
    assert np.allclose(metric["g_yy"], Rmaj**2 * (B / By) ** 2)
    assert np.allclose(metric["gyy"], (By / B) ** 2 / Rmaj**2)
