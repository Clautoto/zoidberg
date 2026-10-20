"""Unit tests for components used to generate FCI grids.

These tests were written by an AI assistant (Claude). They are marked with
the ``ai_generated`` pytest marker, so they can be selected with
``pytest -m ai_generated`` or excluded with ``pytest -m "not ai_generated"``.

Tests marked ``xfail(strict=True)`` document known bugs: they fail now, and
will report XPASS (failing the run) once the bug is fixed, so the marker
can be removed.
"""

import numpy as np
import pytest
from matplotlib.path import Path
from scipy.sparse import csr_matrix

from . import boundary, field, grid, poloidal_grid, rzline, weights, zoidberg

pytestmark = pytest.mark.ai_generated

# Catmull-Rom spline in matrix form: p(u) = [1, u, u^2, u^3] @ CR @ p[-1:3]
# (independent of the polynomial form used in weights.py)
CATMULL_ROM_MATRIX = 0.5 * np.array(
    [
        [0.0, 2.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [2.0, -5.0, 4.0, -1.0],
        [-1.0, 3.0, -3.0, 1.0],
    ]
)


def reference_catmull_rom(u):
    return np.array([1.0, u, u**2, u**3]) @ CATMULL_ROM_MATRIX


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


def row_entries(matrix, row):
    """Dictionary {column: weight} of the non-zero entries in a row"""
    start, end = matrix.indptr[row], matrix.indptr[row + 1]
    return {
        int(c): w
        for c, w in zip(matrix.indices[start:end], matrix.data[start:end])
        if abs(w) > 1e-14
    }


def test_catmull_rom_weights():
    offsets = np.array([-1.0, 0.0, 1.0, 2.0])

    # Interpolating at the stencil points returns that point exactly
    assert np.allclose(weights.catmull_rom_weights(0.0), [0, 1, 0, 0])
    assert np.allclose(weights.catmull_rom_weights(1.0), [0, 0, 1, 0])

    # Explicit values at a non-symmetric point (at u = 0.5 Catmull-Rom
    # coincides with Lagrange cubic interpolation)
    assert np.allclose(
        weights.catmull_rom_weights(0.25),
        [-0.0703125, 0.8671875, 0.2265625, -0.0234375],
    )
    # Lagrange cubic weights differ at the same point
    assert not np.allclose(
        weights.catmull_rom_weights(0.25),
        [-0.0546875, 0.8203125, 0.2734375, -0.0390625],
    )

    for u in np.linspace(0.0, 1.0, 11):
        w = weights.catmull_rom_weights(u)
        assert np.allclose(w, reference_catmull_rom(u))
        # Weights reproduce polynomials up to quadratic
        assert np.isclose(np.sum(w), 1.0)
        assert np.isclose(w @ offsets, u)
        assert np.isclose(w @ offsets**2, u**2)

    # The derivative at the stencil points is the central difference
    h = 1e-6
    dw0 = (weights.catmull_rom_weights(h) - weights.catmull_rom_weights(-h)) / (2 * h)
    dw1 = (weights.catmull_rom_weights(1 + h) - weights.catmull_rom_weights(1 - h)) / (
        2 * h
    )
    assert np.allclose(dw0, [-0.5, 0.0, 0.5, 0.0], atol=1e-6)
    assert np.allclose(dw1, [0.0, -0.5, 0.0, 0.5], atol=1e-6)

    # Cubics are not reproduced: Catmull-Rom is not cubic Lagrange interpolation
    assert not np.isclose(weights.catmull_rom_weights(0.25) @ offsets**3, 0.25**3)


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
    assert np.all(ops["backward_cell_number"] == -1)
    assert np.allclose(ops["forward_leg_weights"], 1.0)
    assert np.allclose(ops["backward_leg_weights"], 1.0)

    # Cell space ordering: evolving cells first in (x, y, z) order, then the
    # inner X boundary, then the outer X boundary
    cell = ops["cell_number"]
    xs = list(range(mxg, nx - mxg)) + list(range(mxg)) + list(range(nx - mxg, nx))
    expected_order = [cell[x, y, z] for x in xs for y in range(ny) for z in range(nz)]
    assert expected_order == list(range(nx * ny * nz))

    rng = np.random.default_rng(42)
    values = rng.normal(size=ops["total_cells"])
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
    nx, ny, nz, mxg = 14, 3, 32, 2
    dx, dz = 0.3, 0.25
    maps = make_shift_maps(nx, ny, nz, mxg, dx, dz)
    ops = weights.calculate_parallel_map_operators(maps)
    cell = ops["cell_number"]

    # Interior is mxg <= x <= nx - mxg - 1, so only the outermost (forward)
    # and innermost (backward) evolving columns leave the domain
    for direction, x_bndry in (("forward", nx - mxg - 1), ("backward", mxg)):
        interior = ops[f"{direction}_leg_interior_number"]
        bndry = ops[f"{direction}_leg_boundary_number"]
        for x in range(mxg, nx - mxg):
            hits = x == x_bndry
            assert np.all((interior[x] < 0) == hits), (direction, x)
            assert np.all((bndry[x] >= 0) == hits), (direction, x)
        assert np.all(ops[f"{direction}_cell_number"][x_bndry] >= 0)

    def func(x, y, z):
        # Quadratic in x, so reproduced exactly; smooth and periodic in z,
        # so the z wrap is exercised
        return 1.0 + 0.5 * x - 0.2 * x**2 + y + np.sin(2 * np.pi * z / nz)

    values = cell_vector(ops, func, (nx, ny, nz))
    guard_columns = {int(c) for c in cell[:mxg].ravel()} | {
        int(c) for c in cell[nx - mxg :].ravel()
    }
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
                    row = legs[x, y, z]
                    if row < 0:
                        continue
                    entries = row_entries(matrix, row)
                    # Guard cells are never used in the interpolation
                    assert not guard_columns & entries.keys()

                    xt, zt = x + sign * dx, z + sign * dz
                    if zt < 0:
                        # Negative z is tested in test_parallel_operator_negative_z
                        continue
                    yt = (y + sign) % ny
                    x_base = int(np.floor(xt))
                    if mxg <= x_base - 1 and x_base + 2 < nx - mxg:
                        checked += 1
                        # Stencil in x inside the domain: interpolation error only
                        # from the sin(z) term, O(h^3) for Catmull-Rom (~1e-4 here)
                        assert np.isclose(
                            result[row], func(xt, yt, zt % nz), atol=1e-3
                        ), (direction, x, y, z)
        assert checked > 0

    # Next to the X boundaries the weight on the guard cell is moved onto the
    # nearest evolving cell: check a stencil explicitly. Backward from
    # x = mxg + 1 maps to xt = mxg + 0.7, with stencil points
    # mxg - 1 (guard), mxg, mxg + 1, mxg + 2
    matrix = operator_matrix(ops, "backward")
    x, y, z = mxg + 1, 1, 5
    entries = row_entries(matrix, ops["backward_leg_interior_number"][x, y, z])
    wx = reference_catmull_rom(0.7)
    wz = reference_catmull_rom(0.75)
    expected_wx = {mxg: wx[0] + wx[1], mxg + 1: wx[2], mxg + 2: wx[3]}
    expected = {}
    for xs, w1 in expected_wx.items():
        for zo, w2 in zip((-1, 0, 1, 2), wz):
            expected[int(cell[xs, y - 1, z - 1 + zo])] = w1 * w2
    assert entries.keys() == expected.keys()
    for col, w in expected.items():
        assert np.isclose(entries[col], w)


@pytest.mark.xfail(
    strict=True,
    reason="weights.py uses int() instead of floor for the z stencil base, "
    "so negative z extrapolates from the wrong points",
)
def test_parallel_operator_negative_z():
    nx, ny, nz, mxg = 8, 2, 16, 2
    maps = make_shift_maps(nx, ny, nz, mxg, 0.0, 0.25)
    ops = weights.calculate_parallel_map_operators(maps)
    cell = ops["cell_number"]
    matrix = operator_matrix(ops, "backward")

    # Backward from z = 0 maps to z = -0.25, which lies between z = -1
    # (i.e. nz - 1) and 0, so the stencil is z = -2, -1, 0, 1
    x, y = 3, 1
    entries = row_entries(matrix, ops["backward_leg_interior_number"][x, y, 0])
    expected_z = [nz - 2, nz - 1, 0, 1]
    expected = {
        int(cell[x, y - 1, zs]): w
        for zs, w in zip(expected_z, reference_catmull_rom(0.75))
    }
    assert entries.keys() == expected.keys()
    for col, w in expected.items():
        assert np.isclose(entries[col], w)


def test_parallel_operator_boundary_legs():
    nx, ny, nz, mxg = 8, 2, 4, 2
    maps = make_shift_maps(nx, ny, nz, mxg, 0.0, 0.0)
    # Field lines starting at the innermost evolving x leave the domain in
    # the forward direction, at the outermost evolving x in the backward one
    maps["forward_xt_prime"][mxg, :, :] = -1
    maps["backward_xt_prime"][nx - mxg - 1, :, :] = nx
    ops = weights.calculate_parallel_map_operators(maps)

    n_cells = nx * ny * nz
    n_evolving = (nx - 2 * mxg) * ny * nz
    n_hits = ny * nz
    # One parallel boundary cell per field line hitting the boundary
    assert ops["total_cells"] == n_cells + 2 * n_hits
    # Each evolving cell still has exactly one leg in each direction
    assert ops["n_forward_legs"] == n_evolving
    assert ops["n_backward_legs"] == n_evolving

    # Parallel boundary cells come after the X-boundary cells, numbered in
    # (x, y, z) order of the source cell
    for direction, x_bndry in (("forward", mxg), ("backward", nx - mxg - 1)):
        numbers = ops[f"{direction}_cell_number"]
        assert np.all(np.delete(numbers, x_bndry, axis=0) == -1)
        assert np.all(numbers[x_bndry] >= n_cells)
        assert np.all(np.diff(numbers[x_bndry].ravel()) == 1)

        matrix = operator_matrix(ops, direction).toarray()
        for y in range(ny):
            for z in range(nz):
                # Boundary legs pick out their parallel boundary cell
                assert ops[f"{direction}_leg_interior_number"][x_bndry, y, z] == -1
                row = matrix[ops[f"{direction}_leg_boundary_number"][x_bndry, y, z]]
                col = numbers[x_bndry, y, z]
                assert row[col] == 1.0
                assert np.count_nonzero(row) == 1


def test_parallel_operator_subsamples():
    nx, ny, nz, mxg = 8, 2, 4, 2
    maps = make_shift_maps(nx, ny, nz, mxg, 0.0, 0.0)
    nsub = 4
    subcell_weights = np.array([0.1, 0.2, 0.3, 0.4])
    for direction in ("forward", "backward"):
        for c in "xz":
            prime = maps.pop(f"{direction}_{c}t_prime")
            maps[f"{direction}_{c}t_primes"] = np.repeat(
                prime[..., None], nsub, axis=-1
            ).astype(float)
    maps["subcell_weights"] = subcell_weights

    # One cell where sub-samples 0 and 2 hit the boundary, and 1 and 3 map
    # to different interior cells
    x, y, z = 3, 0, 1
    maps["forward_xt_primes"][x, y, z] = [-1.0, 3.0, nx, 4.0]
    maps["forward_zt_primes"][x, y, z] = [0.0, 1.0, 0.0, 2.0]

    ops = weights.calculate_parallel_map_operators(maps)
    cell = ops["cell_number"]
    n_evolving = (nx - 2 * mxg) * ny * nz

    # The cell has both an interior and a boundary leg in the forward direction
    assert ops["n_forward_legs"] == n_evolving + 1
    assert ops["n_backward_legs"] == n_evolving
    interior = ops["forward_leg_interior_number"][x, y, z]
    bndry = ops["forward_leg_boundary_number"][x, y, z]
    assert interior >= 0 and bndry >= 0

    # Leg weights are the fractions of the sub-cell weights in each leg
    leg_weights = ops["forward_leg_weights"]
    assert np.isclose(leg_weights[interior], 0.2 + 0.4)
    assert np.isclose(leg_weights[bndry], 0.1 + 0.3)
    others = np.ones(len(leg_weights), dtype=bool)
    others[[interior, bndry]] = False
    assert np.allclose(leg_weights[others], 1.0)
    assert np.allclose(ops["backward_leg_weights"], 1.0)

    # The interior stencil is normalised by the interior weight
    matrix = operator_matrix(ops, "forward")
    assert row_entries(matrix, interior) == pytest.approx(
        {
            int(cell[3, y + 1, 1]): 0.2 / 0.6,
            int(cell[4, y + 1, 2]): 0.4 / 0.6,
        }
    )
    assert row_entries(matrix, bndry) == {int(ops["forward_cell_number"][x, y, z]): 1.0}


def test_rectangular_boundary():
    bndry = boundary.RectangularBoundaryXZ(xmin=-1.0, xmax=2.0, zmin=0.0, zmax=1.0)
    x = np.array([0.0, -1.5, 2.5, 0.5, 0.5, 1.9])
    z = np.array([0.5, 0.5, 0.5, -0.1, 1.2, 0.9])
    expected = [False, True, True, True, True, False]
    assert np.array_equal(bndry.outside(x, 0.0, z), expected)

    # Points exactly on the edges are inside
    x_edge = np.array([-1.0, 2.0, 0.5, 0.5, -1.0])
    z_edge = np.array([0.5, 0.5, 0.0, 1.0, 1.0])
    assert not np.any(bndry.outside(x_edge, 0.0, z_edge))

    # NoBoundary never marks points as outside, for any shape of input
    x2d = np.full((3, 4), 1e6)
    assert not np.any(boundary.NoBoundary().outside(x2d, 0.0, x2d))


# L-shaped (concave) polygon. The closing segment (last to first vertex) is
# horizontal, so it matters: vertical segments never intersect a vertical ray
L_SHAPE_X = np.array([2.0, 2.0, 1.0, 1.0, 0.0, 0.0])
L_SHAPE_Z = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 0.0])


def test_polygon_boundary_concave():
    bndry = boundary.PolygonBoundaryXZ(L_SHAPE_X, L_SHAPE_Z)

    rng = np.random.default_rng(1)
    x = rng.uniform(-0.5, 2.5, (20, 30))
    z = rng.uniform(-0.5, 2.5, (20, 30))

    inside = Path(np.column_stack((L_SHAPE_X, L_SHAPE_Z))).contains_points(
        np.column_stack((x.ravel(), z.ravel()))
    )
    outside = bndry.outside(x, 0.0, z)
    assert outside.shape == x.shape
    assert np.array_equal(outside.ravel(), ~inside)

    # Deterministic points: inside each arm, in the notch, and beyond the
    # polygon in each direction (not in line with a vertex)
    x_pts = np.array([0.5, 1.5, 0.5, 1.5, -0.5, 2.5, 0.5, 0.5])
    z_pts = np.array([0.5, 0.5, 1.5, 1.5, 0.5, 0.5, -0.5, 2.5])
    expected = [False, False, False, True, True, True, True, True]
    assert np.array_equal(bndry.outside(x_pts, 0.0, z_pts), expected)


@pytest.mark.xfail(
    strict=True,
    reason="PolygonBoundaryXZ counts a crossing twice when the ray passes "
    "through a polygon vertex",
)
def test_polygon_boundary_points_below_vertices():
    bndry = boundary.PolygonBoundaryXZ(L_SHAPE_X, L_SHAPE_Z)
    # (1.0, 0.5) is inside, with vertices (1, 1) and (1, 2) directly above.
    # (2.0, -0.2) and (1.0, -0.5) are outside, below vertices
    x_pts = np.array([1.0, 2.0, 1.0])
    z_pts = np.array([0.5, -0.2, -0.5])
    assert np.array_equal(bndry.outside(x_pts, 0.0, z_pts), [False, True, True])


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


@pytest.mark.parametrize("inner_offset", [0.0, 0.4])
def test_structured_poloidal_grid_metric(inner_offset):
    # Annulus between a unit circle centred at (inner_offset, 0) and a circle
    # of radius 2 centred at the origin, joined by straight lines. With an
    # offset the grid is not orthogonal, so g_xz is non-zero.
    nx, nz = 9, 32
    inner = rzline.circle(R0=inner_offset, r=1.0, n=256)
    outer = rzline.circle(R0=0.0, r=2.0, n=256)
    pol = poloidal_grid.grid_annulus(inner, outer, nx, nz, show=False)
    metric = pol.metric()

    # Analytic derivatives. With s = x / (nx - 1) and theta = 2 pi z / nz:
    #   R = (1 - s) (offset + cos theta) + 2 s cos theta
    #   Z = (1 - s) sin theta + 2 s sin theta
    # metric() uses index spacings dx = 1 / nx and dz = 1 / nz
    s = np.linspace(0.0, 1.0, nx)[:, None]
    theta = np.linspace(0.0, 2 * np.pi, nz, endpoint=False)[None, :]
    dsdx = 1.0 / (nx - 1)
    dthetadz = 2 * np.pi / nz
    s, theta = np.broadcast_arrays(s, theta)
    dRds = 2 * np.cos(theta) - (inner_offset + np.cos(theta))
    dZds = np.sin(theta)
    dRdtheta = -(1 + s) * np.sin(theta)
    dZdtheta = (1 + s) * np.cos(theta)

    ex = np.array([dRds, dZds]) * dsdx * nx
    ez = np.array([dRdtheta, dZdtheta]) * dthetadz * nz
    g_xx = np.sum(ex * ex, axis=0)
    g_xz = np.sum(ex * ez, axis=0)
    g_zz = np.sum(ez * ez, axis=0)

    assert np.allclose(metric["dx"], 1.0 / nx)
    assert np.allclose(metric["dz"], 1.0 / nz)
    assert np.allclose(metric["g_xx"], g_xx, rtol=1e-3, atol=1e-3)
    assert np.allclose(metric["g_xz"], g_xz, rtol=1e-3, atol=1e-3)
    assert np.allclose(metric["g_zz"], g_zz, rtol=1e-3, atol=1e-3)
    if inner_offset != 0.0:
        assert np.max(np.abs(g_xz)) > 1.0

    # Contravariant components are the inverse of the analytic metric
    det = g_xx * g_zz - g_xz**2
    assert np.allclose(metric["gxx"], g_zz / det, rtol=1e-3)
    assert np.allclose(metric["gxz"], -g_xz / det, rtol=1e-3, atol=1e-5)
    assert np.allclose(metric["gzz"], g_xx / det, rtol=1e-3)

    # Integrating the area element over the grid gives the annulus area
    # (trapezoidal rule in x, periodic in z)
    area_element = (
        np.sqrt(metric["g_xx"] * metric["g_zz"] - metric["g_xz"] ** 2)
        * metric["dx"]
        * metric["dz"]
    )
    wx = np.ones(nx)
    wx[[0, -1]] = 0.5
    area = np.sum(wx[:, None] * area_element)
    assert np.isclose(area, np.pi * (2.0**2 - 1.0**2), rtol=1e-4)


class CylindricalTestField(field.MagneticField):
    """Cylindrical field varying with R, Z and the toroidal angle"""

    def __init__(self):
        self.attributes = {}

    def Bxfunc(self, x, z, phi):
        return 0.1 * z

    def Byfunc(self, x, z, phi):
        return (1.0 + 0.2 * np.sin(phi)) * 3.0 / x

    def Bzfunc(self, x, z, phi):
        return 0.3 + 0.1 * x

    def Rfunc(self, x, z, phi):
        return x


def expected_cylindrical_metric(rect, fld):
    """g_yy = R^2 (B / B_y)^2 and B evaluated independently on each plane"""
    g_yy = np.zeros(rect.shape)
    B = np.zeros(rect.shape)
    for j, y in enumerate(rect.ycoords):
        pol, _ = rect.getPoloidalGrid(j)
        R, Z = pol.R, pol.Z
        Bx, By, Bz = fld.Bxfunc(R, Z, y), fld.Byfunc(R, Z, y), fld.Bzfunc(R, Z, y)
        B[:, j, :] = np.sqrt(Bx**2 + By**2 + Bz**2)
        g_yy[:, j, :] = R**2 * (B[:, j, :] / By) ** 2
    return g_yy, B


def test_grid_metric_and_get_metric_cylindrical():
    nx, ny, nz = 6, 8, 5
    Ly = 2 * np.pi
    rect = grid.rectangular_grid(
        nx, ny, nz, Lx=0.5, Ly=Ly, Lz=0.4, xcentre=3.0, yperiodic=True
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
    fld = CylindricalTestField()
    rect = grid.rectangular_grid(
        nx, ny, nz, Lx=0.5, Ly=Ly, Lz=0.4, xcentre=3.0, yperiodic=True
    )
    metric, BJg = zoidberg.get_metric(rect, fld)

    g_yy, B = expected_cylindrical_metric(rect, fld)
    assert np.allclose(metric["B"], B)
    assert np.allclose(metric["g_yy"], g_yy)
    assert np.allclose(metric["gyy"], 1 / g_yy)
    # B J / sqrt(g_yy) with the Jacobian of the (identity) poloidal metric
    assert np.allclose(BJg, B)


@pytest.mark.xfail(
    strict=True,
    reason="get_metric modifies the metric cached by Grid.metric() in place",
)
def test_get_metric_does_not_modify_grid_metric():
    fld = CylindricalTestField()
    rect = grid.rectangular_grid(6, 4, 5, Lx=0.5, Lz=0.4, xcentre=3.0, yperiodic=True)
    g_yy, _ = expected_cylindrical_metric(rect, fld)

    first, _ = zoidberg.get_metric(rect, fld)
    first_g_yy = first["g_yy"].copy()
    second, _ = zoidberg.get_metric(rect, fld)

    assert np.allclose(first_g_yy, g_yy)
    assert np.allclose(second["g_yy"], g_yy)
    assert np.allclose(rect.metric()["g_yy"], 1.0)
