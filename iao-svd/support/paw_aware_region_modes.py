"""Support routines retained from the established project implementation."""
from __future__ import annotations
import numpy as np
from paw import nonlq, pawpotcar
from scipy.interpolate import CubicSpline
from sph_harm import sph_r

def atom_projector_matrix(qproj: nonlq, atom_index: int) -> np.ndarray:
    type_index = qproj.element_idx[atom_index]
    radial_angular = qproj.qproj[type_index] * qproj.cqfak[type_index][:, None]
    return np.asarray(
        radial_angular * qproj.crexp[:, atom_index][None, :], dtype=np.complex128
    )


def validate_g_mapping(
    gvec: np.ndarray,
    shape: tuple[int, int, int],
    label: str = "FFT reconstruction",
) -> dict[str, object]:
    """Reject reciprocal vectors that alias onto one FFT-grid index."""

    vectors = np.asarray(gvec)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"{label}: G vectors must have shape (n, 3), got {vectors.shape}")
    if vectors.dtype.kind not in "iu":
        if not np.all(np.isfinite(vectors)) or not np.all(vectors == np.rint(vectors)):
            raise ValueError(f"{label}: G vectors must be finite integers")
        vectors = vectors.astype(np.int64)
    grid = tuple(int(value) for value in shape)
    if len(grid) != 3 or any(value <= 0 for value in grid):
        raise ValueError(f"{label}: FFT shape must contain three positive integers")
    mapped = np.column_stack(
        [vectors[:, axis] % grid[axis] for axis in range(3)]
    )
    linear = np.ravel_multi_index(mapped.T, grid)
    unique_count = int(np.unique(linear).size)
    collision_count = int(len(vectors) - unique_count)
    if collision_count:
        raise ValueError(
            f"{label}: {collision_count} reciprocal-vector collisions on FFT grid {grid}"
        )
    return {
        "grid": list(grid),
        "g_vector_count": int(len(vectors)),
        "unique_grid_index_count": unique_count,
        "collision_count": 0,
        "max_abs_G": (
            np.max(np.abs(vectors), axis=0).astype(int).tolist()
            if len(vectors)
            else [0, 0, 0]
        ),
    }


def read_raw_coefficients(wav, ik: int, bands: np.ndarray) -> np.ndarray:
    return np.stack([wav.coeffs(ik, int(ib)) for ib in bands]).astype(np.complex128)


def radial_difference_values(
    pp: pawpotcar,
    radial_index: int,
    l_value: int,
    radii: np.ndarray,
    spline_cache: dict[tuple[str, int], CubicSpline],
) -> np.ndarray:
    key = (pp.element, radial_index)
    difference = pp.paw_ae_wfc[radial_index] - pp.paw_ps_wfc[radial_index]
    if key not in spline_cache:
        spline_cache[key] = CubicSpline(pp.rgrid, difference, bc_type="natural")
    result = np.zeros_like(radii)
    regular = radii >= pp.rgrid[0]
    result[regular] = spline_cache[key](radii[regular]) / radii[regular]
    small = ~regular
    if np.any(small):
        coefficient = float(np.median(difference[:6] / pp.rgrid[:6] ** (l_value + 1)))
        result[small] = coefficient * radii[small] ** l_value
    return result


def periodic_sphere_grid_points(
    shape: tuple[int, int, int],
    center_frac: np.ndarray,
    lattice: np.ndarray,
    radius: float,
    inverse_lattice: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exactly the periodic FFT points inside one Cartesian sphere.

    If ``|r| <= radius``, every fractional component is bounded by
    ``radius * ||inverse_lattice[:, j]||``.  This gives a rigorous small
    candidate box; the final Cartesian-radius test is identical to the former
    full-grid scan.
    """

    grid = tuple(int(value) for value in shape)
    if len(grid) != 3 or any(value <= 0 for value in grid):
        raise ValueError("FFT shape must contain three positive integers")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError(f"Sphere radius must be finite and non-negative, got {radius}")
    lattice_array = np.asarray(lattice, dtype=np.float64)
    center = np.asarray(center_frac, dtype=np.float64)
    if lattice_array.shape != (3, 3) or center.shape != (3,):
        raise ValueError("Expected lattice shape (3,3) and fractional center shape (3,)")
    inverse = (
        np.linalg.inv(lattice_array)
        if inverse_lattice is None
        else np.asarray(inverse_lattice, dtype=np.float64)
    )
    if inverse.shape != (3, 3) or not np.isfinite(inverse).all():
        raise ValueError("Inverse lattice must be a finite 3x3 matrix")

    component_bounds = radius * np.linalg.norm(inverse, axis=0)
    axis_indices: list[np.ndarray] = []
    axis_deltas: list[np.ndarray] = []
    for axis, size in enumerate(grid):
        indices = np.arange(size, dtype=np.int64)
        delta = indices.astype(np.float64) / size - center[axis]
        delta -= np.rint(delta)
        tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0, float(component_bounds[axis])
        )
        keep = np.abs(delta) <= component_bounds[axis] + tolerance
        axis_indices.append(indices[keep])
        axis_deltas.append(delta[keep])
    if any(len(indices) == 0 for indices in axis_indices):
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    index_mesh = np.meshgrid(*axis_indices, indexing="ij")
    delta_mesh = np.meshgrid(*axis_deltas, indexing="ij")
    fractional_delta = np.stack(
        [values.ravel() for values in delta_mesh], axis=1
    )
    cartesian_delta = fractional_delta @ lattice_array
    radii = np.linalg.norm(cartesian_delta, axis=1)
    inside = radii <= radius * (1.0 + 1e-12)
    flat_indices = np.ravel_multi_index(
        tuple(values.ravel() for values in index_mesh), grid
    )
    return (
        np.asarray(flat_indices[inside], dtype=np.int64),
        np.asarray(cartesian_delta[inside], dtype=np.float64),
        np.asarray(radii[inside], dtype=np.float64),
    )
