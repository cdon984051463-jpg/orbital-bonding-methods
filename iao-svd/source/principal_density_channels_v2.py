#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""Principal density-coherence channels between two atomic IAO blocks (v2).

The channels are the SVD of one block of the density matrix,

    C = D[A, B] = U diag(s) V^T,

where D is the density matrix in an orthonormal (Lowdin IAO) basis and A, B are
two disjoint sets of basis labels. Everything reported here -- populations,
environment leakage, star totals, canonical correlations -- is another block of
the same D, so the whole method is one matrix plus a label partition.

The eigen-coordinate form, which v1 and the q214 chain used, is an equivalent
rewriting and is still accepted as input. For ANY orthonormal complete basis X
with D = X^T Lambda X, selecting columns with the 0/1 selector E_A gives

    X_A^T Lambda X_B = E_A^T (X^T Lambda X) E_B = D[A, B],

so those two same-cell inputs define the same cross block, singular values and
same channel subspaces -- the diagonalisation is a coordinate choice, not part
of the definition. ``self_test`` case 11 checks this to machine precision, and
``--config`` accepts weighted ``density_matrices_k`` plus lattice images
(preferred for periodic bonds), a same-cell ``density_matrix``, or the legacy
four-array eigen-coordinate form.

v2 additions over v1, all driven by the single-bond target use case:

  * strict complex handling (assert, never silently coerce)
  * canonical (population-normalised) correlations as an independent pair spectrum
  * pair share, selected/full A/B stars, environment leakage, and a context
    decomposition over an enlarged shell, so multicentre character is visible
    without ever moving the channels off the target pair
  * structure-independent sign gauge fixed on the small (B-side) subspace,
    with an explicit tie margin so an ambiguous gauge is reported, not hidden
  * absolute rank floor in addition to the relative one
  * per-channel coherence-density weights for real-space cube assembly
  * cross-structure channel matching in fixed IAO-label coordinates
  * atomic output via a staging directory

v1 (`principal_density_channels.py`) is deliberately left untouched: its
reference data remain in the original project archive.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from periodic_density import density_for_translations, self_test as periodic_density_self_test


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------



def _path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _load_real(path: Path, name: str, tolerance: float) -> np.ndarray:
    """Load an array and require it to be real.

    The upstream periodic chain realifies the complex Bloch/IAO overlaps
    (02_run_multi_k_iao_pilot.realify) before anything reaches this program, so
    a genuinely complex array here means the convention was broken somewhere.
    v1 used ``np.asarray(..., dtype=float)``, which discards the imaginary part
    with only a ComplexWarning. That silent path is removed.
    """
    array = np.load(path, allow_pickle=False)
    if np.iscomplexobj(array):
        worst = float(np.max(np.abs(array.imag))) if array.size else 0.0
        if worst > tolerance:
            raise ValueError(
                f"{name} is complex with max |Im| = {worst:.3e} > {tolerance:.3e}. "
                "This core is real-only; realify upstream (see 02_run_multi_k_iao_pilot.realify) "
                "or extend the core to complex128. Refusing to discard the imaginary part."
            )
        array = array.real
    return np.ascontiguousarray(array, dtype=np.float64)


def _column_range(spec: dict, width: int, what: str) -> np.ndarray:
    try:
        raw = spec["column_range"]
        start, stop = raw
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{what} needs column_range=[start, stop), stop excluded") from exc
    for value in (start, stop):
        # bool is an int subclass; floats that happen to be integral are also rejected.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{what} column_range entries must be integers, got {value!r}")
    if start < 0 or stop <= start or stop > width:
        raise ValueError(f"{what}: invalid column range [{start}, {stop}) for atomic width {width}")
    return np.arange(start, stop, dtype=int)


def _resolve_columns(spec: dict, width: int, what: str) -> np.ndarray:
    """Column selection for one atom: an explicit list or a contiguous range.

    ``columns`` exists for the density-matrix input, where an atom's IAO labels
    are read from iao_labels.csv and need not be assumed contiguous.
    """
    if "columns" in spec:
        raw = spec["columns"]
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{what} columns must be a non-empty list")
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{what} columns entries must be integers, got {value!r}")
        columns = np.asarray(raw, dtype=int)
        if np.unique(columns).size != columns.size:
            raise ValueError(f"{what} columns contain duplicates")
        if columns.min() < 0 or columns.max() >= width:
            raise ValueError(f"{what}: columns outside [0, {width})")
        return columns
    return _column_range(spec, width, what)


def _lattice_translation(spec: dict, what: str) -> np.ndarray:
    raw = np.asarray(spec.get("lattice_translation", [0, 0, 0]))
    if raw.shape != (3,) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{what} lattice_translation must contain three integers")
    if not np.isfinite(raw).all() or not np.allclose(raw, np.rint(raw)):
        raise ValueError(f"{what} lattice_translation must contain three integers")
    return np.rint(raw).astype(int)


def _coordinate_labels(spec: dict, count: int, what: str) -> np.ndarray | None:
    raw = spec.get("coordinate_labels")
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != count or not all(
        isinstance(label, str) and label for label in raw
    ):
        raise ValueError(f"{what} coordinate_labels must contain one non-empty string per column")
    return np.asarray(raw)


def _density_support(matrix: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors spanning a positive-semidefinite block."""
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if values.min() < -floor:
        raise ValueError(f"atomic density block is not positive semidefinite: {values.min():.3e}")
    keep = values > floor
    return values[keep], vectors[:, keep]


def _relative_spectral_gaps(values: np.ndarray) -> list[float | None]:
    gaps: list[float | None] = []
    for index, value in enumerate(values):
        neighbours = []
        if index > 0:
            neighbours.append(values[index - 1] - value)
        if index + 1 < values.size:
            neighbours.append(value - values[index + 1])
        gaps.append(float(min(neighbours) / value) if neighbours and value > 0.0 else None)
    return gaps


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def principal_angles(basis_p: np.ndarray, basis_q: np.ndarray) -> np.ndarray:
    """Principal angles (degrees, ascending) between two orthonormal column spans.

    Uses the sine formulation for small angles (Knyazev-Argentati). Taking
    arccos of a singular value close to 1 loses half the available digits --
    precisely the regime this diagnostic exists to measure, since a
    well-transported atomic subspace should give angles of a few degrees.

    Complex bases are handled with the Hermitian inner product. Without the
    conjugation a complex pair of nearly identical subspaces reports ~85
    degrees, which is how this was found: the density-matrix side is real, but
    the occupied-space amplitudes <psi_s|a_i> used by the basis-convergence
    check are not.
    """
    if basis_p.shape != basis_q.shape:
        raise ValueError(f"Subspaces must have equal shape, got {basis_p.shape} and {basis_q.shape}")
    left, cosines, right_t = np.linalg.svd(basis_p.conj().T @ basis_q)
    cosines = np.clip(cosines, -1.0, 1.0)

    # principal vectors: p_k = (P U)_k, q_k = (Q V)_k, with p_k . q_k = cos theta_k
    rotated_q = basis_q @ right_t.conj().T
    residual = rotated_q - basis_p @ (basis_p.conj().T @ rotated_q)
    sines = np.clip(np.linalg.norm(residual, axis=0), 0.0, 1.0)

    angles = np.where(cosines > np.sqrt(0.5), np.arcsin(sines), np.arccos(cosines))
    return np.degrees(np.sort(angles))


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------



def _load_density_matrix_form(config: dict, config_path: Path, tolerance: float,
                              imaginary_tolerance: float) -> dict:
    """Input form 1: the density matrix itself, in fixed IAO label coordinates.

    Nothing has to be orthogonalised, reordered or diagonalised here; an atom is
    just a set of columns. The only structural requirements are that D is real,
    symmetric and positive semi-definite, which is what makes its blocks a
    density and not an arbitrary matrix.
    """
    path = _path(config_path, config["density_matrix"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")
    paths = {"density_matrix": path}

    matrix = _load_real(path, "density_matrix", imaginary_tolerance)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"density_matrix must be square, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("density_matrix contains NaN or infinity")
    scale = max(float(np.max(np.abs(matrix))), 1.0)
    symmetry = float(np.max(np.abs(matrix - matrix.T)))
    if symmetry > tolerance * scale:
        raise ValueError(
            f"density_matrix is not symmetric: max |D - D^T| = {symmetry:.3e} "
            f"> {tolerance:.3e} x {scale:.3e}"
        )
    gram = 0.5 * (matrix + matrix.T)
    smallest = float(np.linalg.eigvalsh(gram)[0])
    floor = float(config.get("density_negative_eigenvalue_tolerance", 1.0e-8)) * scale
    if smallest < -floor:
        raise ValueError(
            f"density_matrix is not positive semi-definite: smallest eigenvalue "
            f"{smallest:.3e} < -{floor:.3e}"
        )
    return {
        "input_mode": "density_matrix",
        "paths": paths,
        "gram": gram,
        "selectable_width": gram.shape[0],
        "eigenvalues": None,
        "atomic": None,
        "environment": None,
        "full": None,
        "basis_metrics": {"density_matrix_symmetry_max_abs_error": symmetry},
        "diagnostics": {
            "density_matrix_smallest_eigenvalue": smallest,
            "density_matrix_trace": float(np.trace(gram)),
        },
    }


def _load_periodic_density_form(config: dict, config_path: Path, tolerance: float) -> dict:
    """Input form 2: weighted D(k) blocks plus one lattice image per column."""
    keys = ("density_matrices_k", "kpoint_fractional_coordinates")
    if "column_translations" not in config:
        raise ValueError("periodic density input requires column_translations")
    paths = {key: _path(config_path, config[key]) for key in keys}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + "; ".join(missing))
    density_k = np.load(paths["density_matrices_k"], allow_pickle=False)
    kpoints = np.load(paths["kpoint_fractional_coordinates"], allow_pickle=False)
    column_translations = np.asarray(config["column_translations"])
    gram = density_for_translations(density_k, kpoints, column_translations)
    scale = max(float(np.max(np.abs(gram))), 1.0)
    smallest = float(np.linalg.eigvalsh(gram)[0])
    floor = float(config.get("density_negative_eigenvalue_tolerance", 1.0e-8)) * scale
    if smallest < -floor:
        raise ValueError(
            f"translated density_matrix is not positive semi-definite: smallest eigenvalue "
            f"{smallest:.3e} < -{floor:.3e}"
        )
    return {
        "input_mode": "periodic_density",
        "paths": paths,
        "gram": gram,
        "selectable_width": gram.shape[0],
        "eigenvalues": None,
        "atomic": None,
        "environment": None,
        "full": None,
        "basis_metrics": {"density_matrix_symmetry_max_abs_error": 0.0},
        "diagnostics": {
            "density_matrix_smallest_eigenvalue": smallest,
            "density_matrix_trace": float(np.trace(gram)),
        },
        "column_translations": np.rint(column_translations).astype(int),
    }


def _load_eigenbasis_form(config: dict, config_path: Path, tolerance: float,
                          imaginary_tolerance: float) -> dict:
    """Input form 2: eigenvalues plus an orthonormal complete basis (v1 legacy).

    Equivalent to form 1 by X_A^T Lambda X_B = D[A, B] with D = X^T Lambda X;
    the coverage and orthonormality checks below are exactly what makes that
    rewriting exact, so they are kept where they belong -- on this input form.
    """
    keys = ("density_eigenvalues", "atomic_basis", "environment_basis", "full_basis")
    paths = {key: _path(config_path, config[key]) for key in keys}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + "; ".join(missing))


    eigenvalues = _load_real(paths["density_eigenvalues"], "density_eigenvalues", imaginary_tolerance)
    atomic = _load_real(paths["atomic_basis"], "atomic_basis", imaginary_tolerance)
    environment = _load_real(paths["environment_basis"], "environment_basis", imaginary_tolerance)
    full = _load_real(paths["full_basis"], "full_basis", imaginary_tolerance)

    if eigenvalues.ndim != 1:
        raise ValueError("density_eigenvalues must be one-dimensional")
    n = eigenvalues.size
    if any(array.ndim != 2 for array in (atomic, environment, full)):
        raise ValueError("All basis arrays must be two-dimensional")
    if any(array.shape[0] != n for array in (atomic, environment, full)):
        raise ValueError("Every basis row count must equal the density dimension")
    if full.shape != (n, n) or atomic.shape[1] + environment.shape[1] != n:
        raise ValueError("atomic_basis plus environment_basis must fully cover the n-dimensional space")
    if not all(np.isfinite(array).all() for array in (eigenvalues, atomic, environment, full)):
        raise ValueError("Inputs contain NaN or infinity")
    if np.any(eigenvalues < 0.0):
        raise ValueError("density_eigenvalues must be non-negative")

    identity = np.eye(n)
    metrics = {
        "atomic_orthonormality_max_abs_error": float(
            np.max(np.abs(atomic.T @ atomic - np.eye(atomic.shape[1])))
        ),
        "environment_orthonormality_max_abs_error": float(
            np.max(np.abs(environment.T @ environment - np.eye(environment.shape[1])))
        )
        if environment.shape[1]
        else 0.0,
        "atomic_environment_overlap_max_abs": float(np.max(np.abs(atomic.T @ environment)))
        if environment.shape[1]
        else 0.0,
        "full_orthonormality_max_abs_error": float(np.max(np.abs(full.T @ full - identity))),
        "combined_vs_full_max_abs_error": float(
            np.max(np.abs(np.column_stack((atomic, environment)) - full))
        ),
    }
    failures = {key: value for key, value in metrics.items() if value > tolerance}
    if failures:
        raise ValueError(f"Full-coverage basis validation failed at tolerance {tolerance}: {failures}")

    # D in the coordinates of the supplied basis columns. Column mu of `full` is
    # basis vector mu, so gram[mu, nu] = X_mu^T Lambda X_nu, and any block of it
    # is the corresponding block of the density.
    gram = full.T @ (eigenvalues[:, None] * full)
    gram = 0.5 * (gram + gram.T)
    return {
        "input_mode": "eigenbasis",
        "paths": paths,
        "gram": gram,
        # atom specs index the atomic block, which `combined_vs_full` pins to the
        # leading columns of `full`, so they index the gram unchanged
        "selectable_width": atomic.shape[1],
        "eigenvalues": eigenvalues,
        "atomic": atomic,
        "environment": environment,
        "full": full,
        "basis_metrics": metrics,
        "diagnostics": {
            "density_matrix_trace": float(eigenvalues.sum()),
            "density_matrix_smallest_eigenvalue": float(eigenvalues.min()),
        },
    }


def load_problem(config_path: Path) -> dict:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))

    tolerance = float(config.get("orthogonality_tolerance", 1.0e-10))
    if not np.isfinite(tolerance) or tolerance <= 0.0 or tolerance > 1.0e-3:
        raise ValueError(f"orthogonality_tolerance must be finite in (0, 1e-3], got {tolerance}")
    imaginary_tolerance = float(config.get("imaginary_tolerance", 1.0e-12))

    forms = [
        key for key in ("density_matrix", "density_matrices_k", "density_eigenvalues")
        if key in config
    ]
    if len(forms) != 1:
        raise ValueError(
            "config mixes input forms or gives none; give exactly one: density_matrix, density_matrices_k, "
            "or density_eigenvalues"
        )
    if forms[0] == "density_matrix":
        loaded = _load_density_matrix_form(config, config_path, tolerance, imaginary_tolerance)
    elif forms[0] == "density_matrices_k":
        loaded = _load_periodic_density_form(config, config_path, tolerance)
    else:
        loaded = _load_eigenbasis_form(config, config_path, tolerance, imaginary_tolerance)

    width = loaded["selectable_width"]
    total = loaded["gram"].shape[0]
    atom_a, atom_b = config["atom_a"], config["atom_b"]
    columns_a = _resolve_columns(atom_a, width, "atom_a")
    columns_b = _resolve_columns(atom_b, width, "atom_b")
    coordinate_labels_a = _coordinate_labels(atom_a, columns_a.size, "atom_a")
    coordinate_labels_b = _coordinate_labels(atom_b, columns_b.size, "atom_b")
    translation_a = _lattice_translation(atom_a, "atom_a")
    translation_b = _lattice_translation(atom_b, "atom_b")
    if np.intersect1d(columns_a, columns_b).size:
        raise ValueError("atom_a and atom_b column ranges must be disjoint")

    # Enlarged shell. These atoms never enter the channel definition; they only
    # supply the denominators and the multicentre diagnostics.
    context = []
    used = set(columns_a.tolist()) | set(columns_b.tolist())
    for spec in config.get("context_atoms", []):
        columns = _resolve_columns(spec, width, f"context atom {spec.get('label', '?')}")
        overlap = used & set(columns.tolist())
        if overlap:
            raise ValueError(f"context atom {spec.get('label', '?')} reuses columns {sorted(overlap)}")
        used |= set(columns.tolist())
        context.append({
            "label": str(spec.get("label", "context")),
            "columns": columns,
            "translation": _lattice_translation(spec, f"context atom {spec.get('label', '?')}")
        })

    selected_images = [("atom_a", columns_a, translation_a),
                       ("atom_b", columns_b, translation_b)] + [
        (entry["label"], entry["columns"], entry["translation"]) for entry in context
    ]
    if loaded["input_mode"] == "periodic_density":
        for label, columns, translation in selected_images:
            if not np.all(loaded["column_translations"][columns] == translation):
                raise ValueError(f"{label} lattice_translation disagrees with column_translations")
    elif any(np.any(translation) for _, _, translation in selected_images):
        raise ValueError(
            f"{loaded['input_mode']} can represent only R=(0,0,0); use density_matrices_k "
            "for lattice-image atoms"
        )

    # "Atomic" columns are the ones an atom label can be selected from; the rest
    # of the density is the environment. In the eigenbasis form that split is
    # given by the supplied atomic/environment bases; with a density matrix the
    # selectable space is the whole IAO set, so every column that no named atom
    # claims is environment.
    if loaded["input_mode"] != "eigenbasis":
        atomic_columns = np.asarray(sorted(used), dtype=int)
    else:
        atomic_columns = np.arange(width, dtype=int)
    environment_columns = np.setdiff1d(np.arange(total, dtype=int), atomic_columns)

    return {
        "config": config,
        "config_path": config_path,
        "columns_a": columns_a,
        "columns_b": columns_b,
        "coordinate_labels_a": coordinate_labels_a,
        "coordinate_labels_b": coordinate_labels_b,
        "context": context,
        "atomic_columns": atomic_columns,
        "environment_columns": environment_columns,
        "label_a": str(atom_a.get("label", "atom_a")),
        "label_b": str(atom_b.get("label", "atom_b")),
        "tolerance": tolerance,
        **loaded,
    }


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------

def compute(problem: dict) -> dict:
    gram = problem["gram"]
    columns_a = problem["columns_a"]
    columns_b = problem["columns_b"]

    def block(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        return gram[np.ix_(rows, columns)]

    cross = block(columns_a, columns_b)
    u, singular_values, vt = np.linalg.svd(cross, full_matrices=False)
    v = vt.T.copy()

    # ---- sign gauge -------------------------------------------------------
    # v1 fixed the sign on the A side (Ru, 6 columns), where the two largest
    # components differed by only 4.4% in q214: a small geometry change flips
    # the pivot and the whole partner cube. v2 fixes it on the *smaller* B-side
    # subspace (O 2p, 3 columns), where sigma/pi separation makes the dominant
    # component well defined, and reports the tie margin so an ambiguous gauge
    # is visible instead of silent. An optional local-frame rotation makes the
    # rule identical across structures.
    frame = problem["config"].get("b_side_frame")
    if frame is not None:
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape != (v.shape[0], v.shape[0]):
            raise ValueError(f"b_side_frame must be {v.shape[0]}x{v.shape[0]}")
        if np.max(np.abs(frame.T @ frame - np.eye(frame.shape[0]))) > 1.0e-10:
            raise ValueError("b_side_frame must be orthogonal")
        gauge_vectors = frame.T @ v
    else:
        # must not alias v: the flip below updates u, v and gauge_vectors once each,
        # and an alias would flip v twice and break U diag(s) V^T = C
        gauge_vectors = v.copy()

    def _margins(vectors: np.ndarray) -> tuple[list[int], list[float]]:
        pivots, margins = [], []
        for channel in range(vectors.shape[1]):
            column = np.abs(vectors[:, channel])
            order = np.argsort(column)[::-1]
            largest = float(column[order[0]])
            second = float(column[order[1]]) if column.size > 1 else 0.0
            pivots.append(int(order[0]))
            margins.append((largest - second) / largest if largest > 0.0 else 1.0)
        return pivots, margins

    # Margins on both sides are reported so the fragile gauge is visible. In
    # q214 the A side ties at 4.4% and the B side at 1.2%: neither is safe for a
    # structure series without a local bond frame, and v1 reported neither.
    gauge_pivots, gauge_margins = _margins(gauge_vectors)
    _, gauge_margins_other = _margins(u)
    gauge_flips = []
    for channel in range(singular_values.size):
        flip = gauge_vectors[gauge_pivots[channel], channel] < 0.0
        gauge_flips.append(bool(flip))
        if flip:
            u[:, channel] *= -1.0
            v[:, channel] *= -1.0
            gauge_vectors[:, channel] *= -1.0

    # Channel partners as coefficient vectors over ALL basis labels: u_i on the
    # A columns, zero elsewhere. Orthonormality and A/B orthogonality are then
    # structural (disjoint supports), and every reported number is a quadratic
    # form of D, which is what makes the whole method one matrix plus a
    # partition. The v1 arrays x_a @ u were the same vectors written in density
    # eigen coordinates; nothing downstream consumed that representation.
    total = gram.shape[0]
    partner_a = np.zeros((total, singular_values.size))
    partner_b = np.zeros((total, singular_values.size))
    partner_a[columns_a] = u
    partner_b[columns_b] = v
    phi_plus = (partner_a + partner_b) / np.sqrt(2.0)
    phi_minus = (partner_a - partner_b) / np.sqrt(2.0)

    def quadratic(vectors: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ji->i", vectors.T, gram @ vectors)

    n_a = quadratic(partner_a)
    n_b = quadratic(partner_b)
    n_plus = quadratic(phi_plus)
    n_minus = quadratic(phi_minus)

    squared = singular_values**2
    pair_total = float(squared.sum())
    fractions = squared / pair_total if pair_total else np.zeros_like(squared)

    # post-hoc normalisation (v1's c_i) -- kept for continuity
    normalized = np.divide(
        singular_values,
        np.sqrt(n_a * n_b),
        out=np.zeros_like(singular_values),
        where=(n_a * n_b) > 0.0,
    )
    # Canonical correlations are a separate pair-level spectrum. They are not
    # indexed versions of the raw SVD channels above.
    block_aa = block(columns_a, columns_a)
    block_bb = block(columns_b, columns_b)
    floor = float(problem["config"].get("block_density_floor", 1.0e-10))
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError(f"block_density_floor must be finite and positive, got {floor}")
    support_values_a, support_a = _density_support(block_aa, floor)
    support_values_b, support_b = _density_support(block_bb, floor)
    support_cross = support_a.T @ cross @ support_b
    if support_values_a.size and support_values_b.size:
        whitened_cross = (
            support_cross
            / np.sqrt(support_values_a)[:, None]
            / np.sqrt(support_values_b)[None, :]
        )
        canonical_u, canonical, canonical_vt = np.linalg.svd(whitened_cross, full_matrices=False)
        canonical_v = canonical_vt.T
        canonical_directions_a = support_a @ (canonical_u / np.sqrt(support_values_a)[:, None])
        canonical_directions_b = support_b @ (canonical_v / np.sqrt(support_values_b)[:, None])
    else:
        canonical = np.empty(0)
        canonical_u = np.empty((support_values_a.size, 0))
        canonical_v = np.empty((support_values_b.size, 0))
        canonical_directions_a = np.empty((columns_a.size, 0))
        canonical_directions_b = np.empty((columns_b.size, 0))
    projector_a = support_a @ support_a.T
    projector_b = support_b @ support_b.T
    canonical_null_coupling = float(np.max(np.abs(cross - projector_a @ cross @ projector_b)))
    block_population_max = max(
        float(support_values_a.max()) if support_values_a.size else 0.0,
        float(support_values_b.max()) if support_values_b.size else 0.0,
        floor,
    )
    canonical_null_tolerance = float(
        np.sqrt(floor * block_population_max) + 10.0 * problem["tolerance"]
    )

    # ---- context / global denominators -----------------------------------
    # Every denominator below is the Frobenius norm of another block of D, so
    # they close exactly against ||D[A, not-A]||^2 rather than approximately.
    environment_columns = problem["environment_columns"]
    leak_a = (
        np.linalg.norm(u.T @ block(columns_a, environment_columns), axis=1)
        if environment_columns.size
        else np.zeros(singular_values.size)
    )
    leak_b = (
        np.linalg.norm(v.T @ block(columns_b, environment_columns), axis=1)
        if environment_columns.size
        else np.zeros(singular_values.size)
    )

    star = []
    selected_context_total_a = 0.0
    selected_context_total_b = 0.0
    for entry in problem["context"]:
        block_ac = block(columns_a, entry["columns"])
        block_bc = block(columns_b, entry["columns"])
        squared_ac = float(np.linalg.norm(block_ac) ** 2)
        squared_bc = float(np.linalg.norm(block_bc) ** 2)
        star.append(
            {
                "label": entry["label"],
                "pair_squared_coherence_with_atom_a": squared_ac,
                "pair_squared_coherence_with_atom_b": squared_bc,
                "per_channel_coherence_from_atom_a": np.linalg.norm(u.T @ block_ac, axis=1),
                "per_channel_coherence_from_atom_b": np.linalg.norm(v.T @ block_bc, axis=1),
            }
        )
        selected_context_total_a += squared_ac
        selected_context_total_b += squared_bc
    environment_total_a = (
        float(np.linalg.norm(block(columns_a, environment_columns)) ** 2)
        if environment_columns.size
        else 0.0
    )
    environment_total_b = (
        float(np.linalg.norm(block(columns_b, environment_columns)) ** 2)
        if environment_columns.size
        else 0.0
    )
    selected_columns = set(problem["columns_a"].tolist()) | set(problem["columns_b"].tolist())
    for entry in problem["context"]:
        selected_columns.update(entry["columns"].tolist())
    # Atomic columns that were supplied but not named as atom_a/atom_b/context.
    # With a density matrix every unnamed column is environment, so this term is
    # structurally zero; with the eigenbasis form the atomic basis may be wider
    # than the named set, and this is what makes
    # full_star = selected_star + unselected + environment a real closure.
    unselected_columns = np.asarray(
        [column for column in problem["atomic_columns"].tolist() if column not in selected_columns],
        dtype=int,
    )
    if unselected_columns.size:
        unselected_atomic_total_a = float(np.linalg.norm(block(columns_a, unselected_columns)) ** 2)
        unselected_atomic_total_b = float(np.linalg.norm(block(columns_b, unselected_columns)) ** 2)
    else:
        unselected_atomic_total_a = unselected_atomic_total_b = 0.0
    selected_star_total_a = pair_total + selected_context_total_a
    selected_star_total_b = pair_total + selected_context_total_b
    full_star_total_a = selected_star_total_a + unselected_atomic_total_a + environment_total_a
    full_star_total_b = selected_star_total_b + unselected_atomic_total_b + environment_total_b
    all_columns = np.arange(gram.shape[0])
    complement_a = np.setdiff1d(all_columns, columns_a)
    complement_b = np.setdiff1d(all_columns, columns_b)
    full_star_closure_error_a = abs(
        full_star_total_a - float(np.linalg.norm(block(columns_a, complement_a)) ** 2)
    )
    full_star_closure_error_b = abs(
        full_star_total_b - float(np.linalg.norm(block(columns_b, complement_b)) ** 2)
    )
    selected_star_fractions_a = (
        squared / selected_star_total_a if selected_star_total_a else np.zeros_like(squared)
    )
    selected_star_fractions_b = (
        squared / selected_star_total_b if selected_star_total_b else np.zeros_like(squared)
    )

    # ---- pair natural orbitals -------------------------------------------
    # The SVD answers "which directions carry the A-B coherence". It does not
    # say how many electrons sit in a bonding versus an antibonding combination,
    # because it never looks at the diagonal blocks. Diagonalising the joint
    # block D[A+B, A+B] does: its eigenvectors are the natural orbitals of the
    # pair and its eigenvalues are their occupations, which is the number a
    # chemist reads first. The two views are linked exactly, without any new
    # approximation, because the off-diagonal block of the same eigen expansion
    # is the cross block the channels diagonalise:
    #
    #   D[A,B] = sum_p n_p phi_(p,A) phi_(p,B)^T
    #   =>  s_i = sum_p n_p (u_i.phi_(p,A)) (phi_(p,B).v_i)  =:  sum_p g_(i,p)
    #
    # so g_(i,p) is how much natural orbital p contributes to channel i, with a
    # sign: positive means its two atomic lobes are in phase across the bond.
    pair_columns = np.concatenate((columns_a, columns_b))
    pair_block = block(pair_columns, pair_columns)
    natural_values, natural_vectors = np.linalg.eigh(0.5 * (pair_block + pair_block.T))
    natural_order = np.argsort(natural_values)[::-1]
    natural_values = natural_values[natural_order]
    natural_vectors = natural_vectors[:, natural_order]
    # deterministic phase: largest component positive
    pivots = np.argmax(np.abs(natural_vectors), axis=0)
    natural_vectors *= np.sign(natural_vectors[pivots, np.arange(natural_vectors.shape[1])])[None, :]
    natural_a = natural_vectors[: columns_a.size]
    natural_b = natural_vectors[columns_a.size :]
    natural_weight_a = np.sum(natural_a**2, axis=0)
    natural_weight_b = np.sum(natural_b**2, axis=0)
    # g_(i,p); rows sum to the singular values, the whole table sums to the
    # trace norm sum_i s_i
    natural_channel_table = natural_values[None, :] * (u.T @ natural_a) * (v.T @ natural_b)
    natural_bonding_index = natural_channel_table.sum(axis=0)
    natural_sum_rule_error = float(
        np.max(np.abs(natural_channel_table.sum(axis=1) - singular_values))
    ) if singular_values.size else 0.0
    # occupations must account for every electron on the two atoms
    natural_population_error = float(abs(natural_values.sum() - np.trace(pair_block)))
    natural_reconstruction_error = float(
        np.max(np.abs((natural_a * natural_values[None, :]) @ natural_b.T - cross))
    )

    # ---- rank ------------------------------------------------------------
    relative = float(problem["config"].get("rank_relative_tolerance", 1.0e-12))
    absolute = float(problem["config"].get("rank_absolute_floor", 1.0e-14))
    top = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(relative * top, absolute)
    interpretable = singular_values > threshold

    reconstruction_error = float(np.max(np.abs(cross - u @ np.diag(singular_values) @ v.T)))
    cross_diagonal_error = float(
        np.max(np.abs(partner_a.T @ gram @ partner_b - np.diag(singular_values)))
    )
    occupancy_errors = np.abs((n_plus - n_minus) - 2.0 * singular_values)

    # spectral separation: how safely channels can be tracked across structures
    gaps = _relative_spectral_gaps(singular_values)

    return {
        "cross": cross,
        "u": u,
        "v": v,
        "partner_a": partner_a,
        "partner_b": partner_b,
        "phi_plus": phi_plus,
        "phi_minus": phi_minus,
        "singular_values": singular_values,
        "squared": squared,
        "fractions": fractions,
        "selected_star_fractions_a": selected_star_fractions_a,
        "selected_star_fractions_b": selected_star_fractions_b,
        "pair_total": pair_total,
        "selected_context_total_a": selected_context_total_a,
        "selected_context_total_b": selected_context_total_b,
        "selected_star_total_a": selected_star_total_a,
        "selected_star_total_b": selected_star_total_b,
        "environment_total_a": environment_total_a,
        "environment_total_b": environment_total_b,
        "unselected_atomic_total_a": unselected_atomic_total_a,
        "unselected_atomic_total_b": unselected_atomic_total_b,
        "full_star_total_a": full_star_total_a,
        "full_star_total_b": full_star_total_b,
        "full_star_closure_error_a": full_star_closure_error_a,
        "full_star_closure_error_b": full_star_closure_error_b,
        "n_a": n_a,
        "n_b": n_b,
        "n_plus": n_plus,
        "n_minus": n_minus,
        "normalized": normalized,
        "canonical": canonical,
        "canonical_u_support": canonical_u,
        "canonical_v_support": canonical_v,
        "canonical_directions_a": canonical_directions_a,
        "canonical_directions_b": canonical_directions_b,
        "canonical_support_a": support_a,
        "canonical_support_b": support_b,
        "canonical_support_values_a": support_values_a,
        "canonical_support_values_b": support_values_b,
        "canonical_null_coupling": canonical_null_coupling,
        "canonical_null_tolerance": canonical_null_tolerance,
        "leak_a": leak_a,
        "leak_b": leak_b,
        "star": star,
        "natural_occupations": natural_values,
        "natural_vectors": natural_vectors,
        "natural_weight_a": natural_weight_a,
        "natural_weight_b": natural_weight_b,
        "natural_channel_table": natural_channel_table,
        "natural_bonding_index": natural_bonding_index,
        "natural_sum_rule_error": natural_sum_rule_error,
        "natural_population_error": natural_population_error,
        "natural_reconstruction_error": natural_reconstruction_error,
        "pair_columns": pair_columns,
        "gauge_pivots": gauge_pivots,
        "gauge_flips": gauge_flips,
        "gauge_margins": np.asarray(gauge_margins),
        "gauge_margins_other_side": np.asarray(gauge_margins_other),
        "gauge_side": "b" if frame is None else "b_framed",
        "gauge_frame_supplied": frame is not None,
        "relative_gaps": gaps,
        "threshold": threshold,
        "interpretable": interpretable,
        "rank": int(np.count_nonzero(interpretable)),
        "reconstruction_error": reconstruction_error,
        "cross_diagonal_error": cross_diagonal_error,
        "occupancy_errors": occupancy_errors,
    }


def validate_result(problem: dict, result: dict) -> dict:
    tolerance = problem["tolerance"]
    r = result["singular_values"].size
    identity = np.eye(r)
    canonical_rank = result["canonical"].size
    canonical_identity = np.eye(canonical_rank)
    canonical_cross = (
        result["canonical_directions_a"].T
        @ result["cross"]
        @ result["canonical_directions_b"]
    )
    metrics = {
        **problem["basis_metrics"],
        "cross_block_reconstruction_max_abs_error": result["reconstruction_error"],
        "channel_cross_diagonalization_max_abs_error": result["cross_diagonal_error"],
        "partner_a_orthonormality_max_abs_error": float(
            np.max(np.abs(result["partner_a"].T @ result["partner_a"] - identity))
        ),
        "partner_b_orthonormality_max_abs_error": float(
            np.max(np.abs(result["partner_b"].T @ result["partner_b"] - identity))
        ),
        "partner_a_b_overlap_max_abs": float(
            np.max(np.abs(result["partner_a"].T @ result["partner_b"]))
        ),
        "maximum_occupancy_identity_abs_error": float(np.max(result["occupancy_errors"])),
        "squared_coherence_closure_abs_error": float(
            abs(result["pair_total"] - np.linalg.norm(result["cross"]) ** 2)
        ),
        "full_atom_a_star_closure_abs_error": result["full_star_closure_error_a"],
        "full_atom_b_star_closure_abs_error": result["full_star_closure_error_b"],
        "canonical_atom_a_population_normalisation_max_abs_error": float(
            np.max(
                np.abs(
                    result["canonical_directions_a"].T
                    @ (result["canonical_support_a"]
                       @ np.diag(result["canonical_support_values_a"])
                       @ result["canonical_support_a"].T)
                    @ result["canonical_directions_a"]
                    - canonical_identity
                )
            )
        ) if canonical_rank else 0.0,
        "canonical_atom_b_population_normalisation_max_abs_error": float(
            np.max(
                np.abs(
                    result["canonical_directions_b"].T
                    @ (result["canonical_support_b"]
                       @ np.diag(result["canonical_support_values_b"])
                       @ result["canonical_support_b"].T)
                    @ result["canonical_directions_b"]
                    - canonical_identity
                )
            )
        ) if canonical_rank else 0.0,
        "canonical_cross_diagonalization_max_abs_error": float(
            np.max(np.abs(canonical_cross - np.diag(result["canonical"])))
        ) if canonical_rank else 0.0,
        "canonical_correlation_excess_above_one": float(
            max(float(result["canonical"].max()) - 1.0, 0.0)
        ) if canonical_rank else 0.0,
        "canonical_null_support_coupling_max_abs": result["canonical_null_coupling"],
        "pair_natural_orbital_cross_block_max_abs_error": result["natural_reconstruction_error"],
        "pair_natural_orbital_channel_sum_rule_max_abs_error": result["natural_sum_rule_error"],
        "pair_natural_orbital_population_closure_abs_error": result["natural_population_error"],
    }
    checks = {
        key: value <= tolerance
        for key, value in metrics.items()
        if key != "canonical_null_support_coupling_max_abs"
    }
    checks["canonical_null_support_coupling_max_abs"] = (
        result["canonical_null_coupling"] <= result["canonical_null_tolerance"]
    )
    if not all(checks.values()):
        failed = {key: metrics[key] for key, passed in checks.items() if not passed}
        raise AssertionError(f"Channel validation failed at tolerance {tolerance}: {failed}")
    return {"metrics": metrics, "checks": checks}


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def channel_rows(problem: dict, result: dict) -> list[dict]:
    rows = []
    for index, singular_value in enumerate(result["singular_values"]):
        row = {
            "channel": index + 1,
            "interpretable": bool(result["interpretable"][index]),
            # --- intensity -------------------------------------------------
            "singular_value_s": float(singular_value),
            "singular_value_squared": float(result["squared"][index]),
            "coherence_density_weight_2s": float(2.0 * singular_value),
            # --- share -----------------------------------------------------
            "fraction_of_pair_squared_coherence": float(result["fractions"][index]),
            "fraction_of_selected_atom_a_star_squared_coherence": float(
                result["selected_star_fractions_a"][index]
            ),
            "fraction_of_selected_atom_b_star_squared_coherence": float(
                result["selected_star_fractions_b"][index]
            ),
            # --- populations / normalised strength -------------------------
            "n_atom_a": float(result["n_a"][index]),
            "n_atom_b": float(result["n_b"][index]),
            "normalized_coherence_c": float(result["normalized"][index]),
            # --- environment / multicentre ---------------------------------
            "environment_leak_atom_a": float(result["leak_a"][index]),
            "environment_leak_atom_b": float(result["leak_b"][index]),
            # --- gauge / trackability --------------------------------------
            "gauge_pivot_b_side_1based": result["gauge_pivots"][index] + 1,
            "gauge_margin": float(result["gauge_margins"][index]),
            "gauge_margin_a_side": float(result["gauge_margins_other_side"][index]),
            "relative_spectral_gap": result["relative_gaps"][index],
            # --- identities -------------------------------------------------
            "n_plus": float(result["n_plus"][index]),
            "n_minus": float(result["n_minus"][index]),
            "n_plus_minus_n_minus": float(result["n_plus"][index] - result["n_minus"][index]),
            "occupancy_identity_abs_error": float(result["occupancy_errors"][index]),
        }
        for entry in result["star"]:
            row[f"coherence_a_with_{entry['label']}"] = float(
                entry["per_channel_coherence_from_atom_a"][index]
            )
            row[f"coherence_b_with_{entry['label']}"] = float(
                entry["per_channel_coherence_from_atom_b"][index]
            )
        rows.append(row)
    return rows


def natural_orbital_character(problem: dict, result: dict) -> tuple[np.ndarray, float]:
    """bonding / antibonding / nonbonding, with a threshold rather than a sign.

    Most of the natural orbitals of a Ru-O pair are neither: they are Ru t2g
    states that live almost entirely on one atom and carry |bonding_index| of a
    few percent. Splitting purely on the sign puts them in one camp or the other
    and makes the summed occupations jump when a nearly zero index changes sign
    between structures, which is exactly what happened for the 1OHot top bond
    (4/5 there against 3/6 everywhere else). Anything below a fraction of the
    total index mass is therefore reported as nonbonding.
    """
    index = result["natural_bonding_index"]
    fraction = float(problem["config"].get("nonbonding_index_fraction", 0.05))
    if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError(f"nonbonding_index_fraction must be in [0, 1), got {fraction}")
    threshold = fraction * float(np.abs(index).sum())
    character = np.where(
        np.abs(index) <= threshold, "nonbonding",
        np.where(index > 0.0, "bonding", "antibonding"),
    )
    return character, threshold


def natural_orbital_rows(problem: dict, result: dict) -> list[dict]:
    character, _ = natural_orbital_character(problem, result)
    rows = []
    for index, occupation in enumerate(result["natural_occupations"]):
        row = {
            "natural_orbital": index + 1,
            "occupation": float(occupation),
            "weight_atom_a": float(result["natural_weight_a"][index]),
            "weight_atom_b": float(result["natural_weight_b"][index]),
            # signed: positive means the two lobes are in phase across the bond.
            # The column sums to sum_i s_i, the trace norm of the cross block.
            "bonding_index": float(result["natural_bonding_index"][index]),
            "character": str(character[index]),
        }
        for channel in range(result["singular_values"].size):
            row[f"contribution_to_channel_{channel + 1}"] = float(
                result["natural_channel_table"][channel, index]
            )
        rows.append(row)
    return rows


def build_report(problem: dict, result: dict, validation: dict) -> dict:
    rows = channel_rows(problem, result)
    natural_character, natural_threshold = natural_orbital_character(problem, result)
    gauge_floor = float(problem["config"].get("gauge_margin_warning", 0.05))
    fragile = [
        {
            "channel": index + 1,
            "gauge_margin_b_side": float(result["gauge_margins"][index]),
            "gauge_margin_a_side": float(result["gauge_margins_other_side"][index]),
        }
        for index in range(result["singular_values"].size)
        if result["gauge_margins"][index] < gauge_floor
    ]
    degenerate = [
        {"channel": index + 1, "relative_spectral_gap": float(result["relative_gaps"][index])}
        for index in range(result["singular_values"].size)
        if result["relative_gaps"][index] is not None
        and result["relative_gaps"][index]
        < float(problem["config"].get("degeneracy_warning", 0.10))
    ]
    return {
        "status": "pass",
        "method": "PAW-IAO principal density-coherence channels (PAW-IAO-PDC v2)",
        "definition": "channels are the SVD of the density block D[A, B]",
        "input_mode": problem["input_mode"],
        "scope": f"{problem['label_a']}--{problem['label_b']}",
        "pair_lattice_translation": (
            np.asarray(problem["config"]["atom_b"].get("lattice_translation", [0, 0, 0]))
            - np.asarray(problem["config"]["atom_a"].get("lattice_translation", [0, 0, 0]))
        ).tolist(),
        "context_lattice_translations": {
            str(spec.get("label", "context")): spec.get("lattice_translation", [0, 0, 0])
            for spec in problem["config"].get("context_atoms", [])
        },
        "density_dimension": int(problem["gram"].shape[0]),
        "atomic_basis_dimension": int(problem["atomic_columns"].size),
        "environment_basis_dimension": int(problem["environment_columns"].size),
        "density_matrix_diagnostics": problem["diagnostics"],
        "partner_coordinates": (
            "coefficients over the basis labels of the input density; a partner is u_i on "
            "the atom's columns and zero elsewhere"
        ),
        "channels_retained": int(result["singular_values"].size),
        "numerical_rank": result["rank"],
        "rank_threshold": result["threshold"],
        "pair_squared_coherence_total": result["pair_total"],
        "selected_atom_a_context_squared_coherence_total": result["selected_context_total_a"],
        "selected_atom_b_context_squared_coherence_total": result["selected_context_total_b"],
        "selected_atom_a_star_squared_coherence_total": result["selected_star_total_a"],
        "selected_atom_b_star_squared_coherence_total": result["selected_star_total_b"],
        "environment_atom_a_squared_coherence_total": result["environment_total_a"],
        "environment_atom_b_squared_coherence_total": result["environment_total_b"],
        "unselected_atomic_atom_a_squared_coherence_total": result["unselected_atomic_total_a"],
        "unselected_atomic_atom_b_squared_coherence_total": result["unselected_atomic_total_b"],
        "full_atom_a_star_squared_coherence_total": result["full_star_total_a"],
        "full_atom_b_star_squared_coherence_total": result["full_star_total_b"],
        "context_atoms": [entry["label"] for entry in problem["context"]],
        "context_pair_table": [
            {
                "label": entry["label"],
                "squared_coherence_with_atom_a": entry["pair_squared_coherence_with_atom_a"],
                "squared_coherence_with_atom_b": entry["pair_squared_coherence_with_atom_b"],
                "fraction_of_selected_atom_a_star": (
                    entry["pair_squared_coherence_with_atom_a"] / result["selected_star_total_a"]
                    if result["selected_star_total_a"]
                    else 0.0
                ),
                "fraction_of_selected_atom_b_star": (
                    entry["pair_squared_coherence_with_atom_b"] / result["selected_star_total_b"]
                    if result["selected_star_total_b"]
                    else 0.0
                ),
            }
            for entry in result["star"]
        ],
        "input_paths": {key: str(path) for key, path in problem["paths"].items()},
        "checks": validation["checks"],
        "metrics": validation["metrics"],
        "channels": rows,
        "pair_natural_orbitals": {
            "definition": (
                "eigenvectors and occupations of the joint block D[A+B, A+B]; "
                "D[A,B] = sum_p n_p phi_(p,A) phi_(p,B)^T, so s_i = sum_p g_(i,p) with "
                "g_(i,p) = n_p (u_i.phi_(p,A))(phi_(p,B).v_i)"
            ),
            "occupation_scale": "electrons per orbital, spin summed (0 to 2)",
            "total_occupation": float(result["natural_occupations"].sum()),
            "bonding_occupation": float(
                result["natural_occupations"][natural_character == "bonding"].sum()
            ),
            "antibonding_occupation": float(
                result["natural_occupations"][natural_character == "antibonding"].sum()
            ),
            "nonbonding_occupation": float(
                result["natural_occupations"][natural_character == "nonbonding"].sum()
            ),
            "net_bonding_occupation": float(
                result["natural_occupations"][natural_character == "bonding"].sum()
                - result["natural_occupations"][natural_character == "antibonding"].sum()
            ),
            "nonbonding_index_threshold": natural_threshold,
            "nonbonding_index_fraction": float(
                problem["config"].get("nonbonding_index_fraction", 0.05)
            ),
            # Threshold-free summary: every orbital is weighted by how much
            # bonding character it actually carries, so an orbital whose index is
            # near zero contributes near zero instead of being pushed into one
            # camp. Units are electrons; the classified sums above are easier to
            # read, this one is the safe number to compare across structures.
            "occupation_weighted_bonding_index": float(
                (result["natural_occupations"] * result["natural_bonding_index"]).sum()
                / np.abs(result["natural_bonding_index"]).sum()
            ) if np.abs(result["natural_bonding_index"]).sum() > 0 else 0.0,
            "bonding_index_total_equals_trace_norm": float(
                result["natural_bonding_index"].sum()
            ),
            "trace_norm_sum_of_singular_values": float(result["singular_values"].sum()),
            "orbitals": natural_orbital_rows(problem, result),
            "note": (
                "Occupations are populations of the two atoms' joint IAO block, not of "
                "isolated bonds: a natural orbital with weight on both atoms carries the "
                "shared density. The label is the sign and size of the cross term against "
                "the channel directions, not an energy ordering; orbitals carrying less "
                "than nonbonding_index_fraction of the total index mass are reported as "
                "nonbonding because their sign is not stable across structures."
            ),
        },
        "canonical_analysis": {
            "canonical_correlations": result["canonical"].tolist(),
            "atom_a_support_rank": int(result["canonical_support_values_a"].size),
            "atom_b_support_rank": int(result["canonical_support_values_b"].size),
            "atom_a_block_dimension": int(result["canonical_support_a"].shape[0]),
            "atom_b_block_dimension": int(result["canonical_support_b"].shape[0]),
            "block_density_floor": float(problem["config"].get("block_density_floor", 1.0e-10)),
            "null_support_coupling_max_abs": result["canonical_null_coupling"],
            "null_support_coupling_tolerance": result["canonical_null_tolerance"],
            "directions_file": "channel_coefficients.npz",
            "note": (
                "Independent pair-level spectrum. Canonical directions are not raw PDC channels "
                "and must not be joined to raw channel rows by index."
            ),
        },
        "gauge": {
            "side": result["gauge_side"],
            "local_frame_supplied": result["gauge_frame_supplied"],
            "warning_threshold": gauge_floor,
            "fragile_channels": fragile,
            "note": (
                "The sign of each (a_i, b_i) pair is fixed by the dominant component of the "
                "gauge-side singular vector. A small margin means a minor geometry change can "
                "flip the signed partner and plus/minus combination fields; the simultaneous "
                "sign flip leaves the product field invariant. For a structure "
                "series supply b_side_frame (local bond frame) so the rule is identical "
                "everywhere, and check that no channel is listed as fragile."
            ),
        },
        "near_degenerate_channels": degenerate,
        "interpretation_boundary": (
            "Density-coherence channels only. s^2 is an active-space IAO Wiberg-type coherence "
            "index whose absolute scale depends on the active space, the band window and the "
            "orthogonalisation set; it is not a bond order, bond energy, bond strength or "
            "rupture criterion. Channel count is capped by the smaller atomic subspace."
        ),
    }


def save_outputs(problem: dict, result: dict, validation: dict, output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".pdc_v2.", dir=output.parent))
    try:
        np.save(staged / "density_cross_block.npy", result["cross"])
        arrays = dict(
            atom_a_columns=problem["columns_a"],
            atom_b_columns=problem["columns_b"],
            atom_a_svd_coefficients=result["u"],
            atom_b_svd_coefficients=result["v"],
            atom_a_partners=result["partner_a"],
            atom_b_partners=result["partner_b"],
            phi_plus=result["phi_plus"],
            phi_minus=result["phi_minus"],
            singular_values=result["singular_values"],
            interpretable=result["interpretable"],
            coherence_density_weights=2.0 * result["singular_values"],
            canonical_correlations=result["canonical"],
            canonical_atom_a_directions=result["canonical_directions_a"],
            canonical_atom_b_directions=result["canonical_directions_b"],
            canonical_atom_a_support_basis=result["canonical_support_a"],
            canonical_atom_b_support_basis=result["canonical_support_b"],
            canonical_atom_a_support_eigenvalues=result["canonical_support_values_a"],
            canonical_atom_b_support_eigenvalues=result["canonical_support_values_b"],
            canonical_atom_a_support_rank=result["canonical_support_values_a"].size,
            canonical_atom_b_support_rank=result["canonical_support_values_b"].size,
            pair_columns=result["pair_columns"],
            natural_occupations=result["natural_occupations"],
            natural_orbitals=result["natural_vectors"],
            natural_channel_contributions=result["natural_channel_table"],
        )
        if problem["coordinate_labels_a"] is not None:
            arrays["atom_a_coordinate_labels"] = problem["coordinate_labels_a"]
            arrays["atom_b_coordinate_labels"] = problem["coordinate_labels_b"]
        np.savez_compressed(staged / "channel_coefficients.npz", **arrays)
        rows = channel_rows(problem, result)
        with (staged / "channel_table.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        natural_rows = natural_orbital_rows(problem, result)
        with (staged / "pair_natural_orbitals.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(natural_rows[0]))
            writer.writeheader()
            writer.writerows(natural_rows)

        report = build_report(problem, result, validation)
        (staged / "density-channel-validation.json").write_text(
            json_text(report), encoding="utf-8"
        )
        lines = [
            "# PAW-IAO-PDC v2 validation",
            "",
            f"- pair: `{problem['label_a']}--{problem['label_b']}`",
            f"- input: `{problem['input_mode']}`",
            f"- coverage: `{problem['atomic_columns'].size} + {problem['environment_columns'].size} "
            f"= {problem['gram'].shape[0]}`",
            f"- channels: `{result['singular_values'].size}` (numerical rank `{result['rank']}`)",
            f"- singular values: `{', '.join(f'{value:.15g}' for value in result['singular_values'])}`",
            f"- pair shares: `{', '.join(f'{value:.6%}' for value in result['fractions'])}`",
            f"- selected A-star shares: `{', '.join(f'{value:.6%}' for value in result['selected_star_fractions_a'])}`",
            f"- pair natural occupations: "
            f"`{', '.join(f'{value:.4f}' for value in result['natural_occupations'])}`",
            f"- bonding / antibonding / nonbonding occupation: "
            f"`{report['pair_natural_orbitals']['bonding_occupation']:.4f}"
            f" / {report['pair_natural_orbitals']['antibonding_occupation']:.4f}"
            f" / {report['pair_natural_orbitals']['nonbonding_occupation']:.4f}`",
            f"- canonical pair spectrum: `{', '.join(f'{value:.6f}' for value in result['canonical'])}`",
            f"- canonical support ranks A/B: `{result['canonical_support_values_a'].size}/{result['canonical_support_values_b'].size}`",
            f"- minimum gauge margin: `{float(result['gauge_margins'].min()):.4f}`",
            "- validation: `PASS`",
            "",
            "Density-coherence channels only; not bond energies or bond orders.",
        ]
        (staged / "validation-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (staged / "run_status.json").write_text(
            json_text(
                {
                    "status": "complete",
                    "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "channels": int(result["singular_values"].size),
                }
            ),
            encoding="utf-8",
        )
        os.replace(staged, output)
    except BaseException:
        for path in sorted(staged.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        staged.rmdir()
        raise


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def _toy(columns_a, columns_b, eigenvalues, atomic, environment, config=None):
    """Eigen-coordinate toy problem: X = [atomic, environment], D = X^T Lambda X."""
    full = np.column_stack((atomic, environment))
    gram = full.T @ (eigenvalues[:, None] * full)
    return {
        "config": config or {},
        "input_mode": "eigenbasis",
        "gram": 0.5 * (gram + gram.T),
        "eigenvalues": eigenvalues,
        "atomic": atomic,
        "environment": environment,
        "full": full,
        "columns_a": np.asarray(columns_a),
        "columns_b": np.asarray(columns_b),
        "context": [],
        "atomic_columns": np.arange(atomic.shape[1]),
        "environment_columns": np.arange(atomic.shape[1], full.shape[1]),
        "label_a": "A",
        "label_b": "B",
        "basis_metrics": {"seed": 0.0},
        "diagnostics": {},
        "tolerance": 1.0e-10,
    }


def _toy_density(columns_a, columns_b, density, context=None, config=None):
    """Density-matrix toy problem: D given directly in label coordinates."""
    density = 0.5 * (np.asarray(density, dtype=float) + np.asarray(density, dtype=float).T)
    context = context or []
    named = sorted(
        set(np.asarray(columns_a).tolist())
        | set(np.asarray(columns_b).tolist())
        | {column for entry in context for column in np.asarray(entry["columns"]).tolist()}
    )
    return {
        "config": config or {},
        "input_mode": "density_matrix",
        "gram": density,
        "eigenvalues": None,
        "atomic": None,
        "environment": None,
        "full": None,
        "columns_a": np.asarray(columns_a),
        "columns_b": np.asarray(columns_b),
        "context": [
            {"label": entry["label"], "columns": np.asarray(entry["columns"])} for entry in context
        ],
        "atomic_columns": np.asarray(named, dtype=int),
        "environment_columns": np.setdiff1d(np.arange(density.shape[0]), np.asarray(named, dtype=int)),
        "label_a": "A",
        "label_b": "B",
        "basis_metrics": {"seed": 0.0},
        "diagnostics": {},
        "tolerance": 1.0e-10,
    }


def self_test() -> None:
    periodic_density_self_test()
    rng = np.random.default_rng(214)

    # 1. rank-1 reference (identical to the v1 self-test)
    root_two = np.sqrt(2.0)
    atomic = np.column_stack(
        (
            np.array([1.0, 0.0, 1.0, 0.0]) / root_two,
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, -1.0, 0.0]) / root_two,
        )
    )
    eigenvalues = np.array([1.8, 1.2, 0.8, 0.4])
    environment = np.array([[0.0], [0.0], [0.0], [1.0]])
    problem = _toy([0, 1], [2], eigenvalues, atomic, environment)
    result = compute(problem)
    validate_result(problem, result)
    assert result["singular_values"].shape == (1,)
    assert np.isclose(result["singular_values"][0], 0.5)
    assert result["rank"] == 1

    # 2. invariance under rotation inside each atomic subspace
    basis, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    lam = np.sort(rng.uniform(0.05, 2.0, size=12))[::-1]
    prob = _toy(range(4), range(4, 7), lam, basis[:, :7], basis[:, 7:])
    base = compute(prob)
    rot_a, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rot_b, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    spun = basis[:, :7].copy()
    spun[:, :4] = spun[:, :4] @ rot_a
    spun[:, 4:7] = spun[:, 4:7] @ rot_b
    spun_result = compute(_toy(range(4), range(4, 7), lam, spun, basis[:, 7:]))
    assert np.allclose(base["singular_values"], spun_result["singular_values"], atol=1e-12), (
        "singular values must depend only on the two spans"
    )
    assert np.allclose(base["canonical"], spun_result["canonical"], atol=1e-12)

    # 3. zero cross block
    zero_atomic = np.zeros((6, 3))
    zero_atomic[0, 0] = zero_atomic[1, 1] = 1.0
    zero_atomic[2, 2] = 1.0
    lam0 = np.array([1.0, 1.0, 0.0, 0.5, 0.5, 0.5])
    env0 = np.zeros((6, 3))
    env0[3, 0] = env0[4, 1] = env0[5, 2] = 1.0
    zero_result = compute(_toy([0, 1], [2], lam0, zero_atomic, env0))
    assert zero_result["rank"] == 0
    assert np.allclose(zero_result["fractions"], 0.0)

    # 3b. the sign gauge must never break U diag(s) V^T = C. Random cases are
    # used because a fixed toy can silently avoid ever needing a flip, which is
    # exactly how an aliasing bug in the gauge survived the first draft.
    flips_exercised = False
    for seed in range(25):
        gen = np.random.default_rng(1000 + seed)
        frame_basis, _ = np.linalg.qr(gen.normal(size=(14, 14)))
        lam_r = gen.uniform(0.02, 2.0, size=14)
        prob_r = _toy(range(5), range(5, 8), lam_r, frame_basis[:, :8], frame_basis[:, 8:])
        res = compute(prob_r)
        rebuilt = res["u"] @ np.diag(res["singular_values"]) @ res["v"].T
        assert np.allclose(rebuilt, res["cross"], atol=1e-12), f"gauge broke the SVD (seed {seed})"
        cross_diag = res["partner_a"].T @ prob_r["gram"] @ res["partner_b"]
        assert np.allclose(cross_diag, np.diag(res["singular_values"]), atol=1e-12)
        if any(res["gauge_flips"]):
            flips_exercised = True
    assert flips_exercised, "sign-gauge test never exercised a flip"

    # 4. non-zero near-degenerate pair is reported as such
    cross = np.diag([0.300, 0.295, 0.050])
    density = np.block([[np.eye(3), cross], [cross, np.eye(3)]])
    deg_values, deg_vectors = np.linalg.eigh(density)
    deg = compute(_toy(range(3), range(3, 6), deg_values, deg_vectors.T, np.empty((6, 0))))
    assert deg["relative_gaps"][0] is not None and deg["relative_gaps"][0] < 0.02

    # 5. complex input must raise, not silently truncate
    with tempfile.TemporaryDirectory() as name:
        path = Path(name) / "complex.npy"
        np.save(path, np.array([1.0 + 1.0j, 2.0]))
        try:
            _load_real(path, "probe", 1.0e-12)
        except ValueError as exc:
            assert "complex" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("complex input was not rejected")

    # 6. malformed column ranges
    for bad in ({"column_range": [0, 0]}, {"column_range": [1.0, 3]}, {"column_range": [0, 99]}):
        try:
            _column_range(bad, 6, "probe")
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"bad column_range accepted: {bad}")

    # 7. semidefinite population blocks use their support instead of returning NaN
    density = np.array([[1.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 1.0]])
    psd_values, psd_vectors = np.linalg.eigh(density)
    psd = compute(_toy([0, 1], [2], psd_values, psd_vectors.T, np.empty((3, 0))))
    assert psd["canonical_support_values_a"].size == 1
    assert psd["canonical_support_values_b"].size == 1
    assert np.allclose(psd["canonical"], [0.5], atol=1e-12)
    validate_result(_toy([0, 1], [2], psd_values, psd_vectors.T, np.empty((3, 0))), psd)

    # 8. undefined single-channel spectral gap is JSON null, never Infinity
    assert channel_rows(problem, result)[0]["relative_spectral_gap"] is None
    json.dumps(channel_rows(problem, result), allow_nan=False)

    # 9. fixed-label matcher recovers permutation and simultaneous sign
    with tempfile.TemporaryDirectory() as name:
        first, second = Path(name) / "a", Path(name) / "b"
        first.mkdir()
        second.mkdir()
        eye = np.eye(3)
        permutation = [1, 0, 2]
        signs = np.array([-1.0, 1.0, -1.0])
        common = {
            "atom_a_coordinate_labels": np.array(["Ru:s:1", "Ru:d:2", "Ru:d:3"]),
            "atom_b_coordinate_labels": np.array(["O:p:1", "O:p:2", "O:p:3"]),
            "singular_values": np.array([0.30, 0.20, 0.10]),
            "interpretable": np.ones(3, dtype=bool),
        }
        np.savez(first / "channel_coefficients.npz", atom_a_svd_coefficients=eye,
                 atom_b_svd_coefficients=eye, atom_a_columns=np.arange(3),
                 atom_b_columns=np.arange(3), **common)
        moved = eye[:, permutation] * signs[None, :]
        np.savez(second / "channel_coefficients.npz", atom_a_svd_coefficients=moved,
                 atom_b_svd_coefficients=moved, atom_a_columns=np.arange(10, 13),
                 atom_b_columns=np.arange(20, 23), **common)
        matched = match_channels(first, second)
        assert [entry["run_b_channel"] for entry in matched["matches"]] == [2, 1, 3]

    # 10. principal-angle helper remains for reference-subspace checks
    span = np.linalg.qr(rng.normal(size=(20, 4)))[0]
    assert np.allclose(principal_angles(span, span), 0.0, atol=1e-8)
    other = np.linalg.qr(rng.normal(size=(20, 20)))[0]
    assert np.allclose(principal_angles(other[:, :4], other[:, 4:8]), 90.0, atol=1e-8)
    # complex bases need the Hermitian inner product: a subspace compared with a
    # phase-rotated copy of itself is the SAME subspace and must give 0 degrees
    complex_span = np.linalg.qr(rng.normal(size=(20, 3)) + 1j * rng.normal(size=(20, 3)))[0]
    phased = complex_span * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=3))[None, :]
    assert np.allclose(principal_angles(complex_span, phased), 0.0, atol=1e-8), (
        "complex principal angles must use the conjugate transpose"
    )
    mixed = complex_span @ np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]
    assert np.allclose(principal_angles(complex_span, mixed), 0.0, atol=1e-8)

    # 11. the two input forms are one problem (executable equivalence proof).
    # For any orthonormal complete X with D = X^T Lambda X and the 0/1 selector
    # E_A, X_A^T Lambda X_B = E_A^T (X^T Lambda X) E_B = D[A, B]. So the cross
    # block, the singular values, both channel subspaces and every star total
    # must agree to machine precision, and the eigendecomposition is a
    # coordinate choice rather than a step of the method. This is the check that
    # licenses stating the method directly on D[A, B].
    gen = np.random.default_rng(2140)
    size = 17
    root = gen.normal(size=(size, size))
    density = root @ root.T / size
    density = 0.5 * (density + density.T)
    values, vectors = np.linalg.eigh(density)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    x_basis = vectors.T  # column mu holds the coordinates of label mu
    pair_a, pair_b = np.arange(0, 5), np.arange(5, 8)
    shell = [{"label": "C1", "columns": np.arange(8, 11)},
             {"label": "C2", "columns": np.arange(11, 13)}]

    eigen_problem = _toy(pair_a, pair_b, values, x_basis[:, :13], x_basis[:, 13:])
    eigen_problem["context"] = [dict(entry) for entry in shell]
    assert np.max(np.abs(eigen_problem["gram"] - density)) < 1e-13, "X^T Lambda X != D"
    eigen_result = compute(eigen_problem)
    validate_result(eigen_problem, eigen_result)

    density_problem = _toy_density(pair_a, pair_b, density, context=shell)
    density_result = compute(density_problem)
    validate_result(density_problem, density_result)

    assert np.allclose(
        eigen_result["cross"], density[np.ix_(pair_a, pair_b)], atol=1e-13
    ), "X_A^T Lambda X_B != D[A, B]"
    assert np.allclose(eigen_result["singular_values"], density_result["singular_values"], atol=1e-13)
    assert np.allclose(eigen_result["u"], density_result["u"], atol=1e-12)
    assert np.allclose(eigen_result["v"], density_result["v"], atol=1e-12)
    assert np.allclose(eigen_result["canonical"], density_result["canonical"], atol=1e-12)
    for key in ("pair_total", "selected_star_total_a", "selected_star_total_b",
                "environment_total_a", "environment_total_b",
                "full_star_total_a", "full_star_total_b"):
        assert abs(eigen_result[key] - density_result[key]) < 1e-12, key
    for key in ("n_a", "n_b", "n_plus", "n_minus", "leak_a", "leak_b", "normalized"):
        assert np.allclose(eigen_result[key], density_result[key], atol=1e-12), key
    # the star closure is now an exact block identity, not a near-miss
    assert density_result["full_star_closure_error_a"] < 1e-12
    assert density_result["full_star_closure_error_b"] < 1e-12

    # 11b. pair natural orbitals: exact identities, and the textbook two-level
    # case must come out as one filled bonding and one empty antibonding orbital.
    for probe in (eigen_problem, density_problem):
        outcome = compute(probe)
        assert np.allclose(
            outcome["natural_channel_table"].sum(axis=1), outcome["singular_values"], atol=1e-12
        ), "natural orbitals must resum to the singular values"
        assert abs(
            outcome["natural_bonding_index"].sum() - outcome["singular_values"].sum()
        ) < 1e-12, "signed bonding indices must resum to the trace norm"
        assert outcome["natural_reconstruction_error"] < 1e-12
        assert outcome["natural_population_error"] < 1e-12
        assert np.allclose(
            outcome["natural_weight_a"] + outcome["natural_weight_b"], 1.0, atol=1e-12
        )

    # H2-like: bonding orbital doubly occupied, antibonding empty. In an
    # orthonormal two-orbital basis that is D = [[1, 1], [1, 1]].
    two_level = _toy_density([0], [1], np.array([[1.0, 1.0], [1.0, 1.0]]))
    two = compute(two_level)
    assert np.allclose(two["natural_occupations"], [2.0, 0.0], atol=1e-12)
    assert np.allclose(two["natural_weight_a"], [0.5, 0.5], atol=1e-12)
    assert two["natural_bonding_index"][0] > 0 > two["natural_bonding_index"][1] - 1e-15
    assert np.isclose(two["singular_values"][0], 1.0)
    # and the antibonding partner filled instead: D = [[1, -1], [-1, 1]] has the
    # same |s| but the in-phase combination is the empty one
    anti = compute(_toy_density([0], [1], np.array([[1.0, -1.0], [-1.0, 1.0]])))
    assert np.allclose(anti["natural_occupations"], [2.0, 0.0], atol=1e-12)
    assert np.isclose(anti["singular_values"][0], 1.0)

    # a partially filled antibonding level must show up as antibonding occupation
    mixed = compute(_toy_density([0], [1], np.array([[1.5, 0.5], [0.5, 1.5]])))
    assert np.allclose(mixed["natural_occupations"], [2.0, 1.0], atol=1e-12)
    assert mixed["natural_bonding_index"][1] < 0.0

    # 12. the density-matrix loader rejects what it must
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        np.save(folder / "density.npy", density)
        config = {
            "density_matrix": "density.npy",
            "atom_a": {"label": "A", "columns": pair_a.tolist()},
            "atom_b": {"label": "B", "column_range": [5, 8]},
            "context_atoms": [{"label": "C1", "columns": np.arange(8, 11).tolist()}],
        }
        (folder / "config.json").write_text(json_text(config), encoding="utf-8")
        loaded = load_problem(folder / "config.json")
        assert loaded["input_mode"] == "density_matrix"
        assert np.allclose(compute(loaded)["singular_values"], density_result["singular_values"])
        assert loaded["environment_columns"].size == size - 11

        np.save(folder / "density_k.npy", density[None, :, :].astype(complex))
        np.save(folder / "kpoints.npy", np.zeros((1, 3)))
        periodic_config = {
            **{key: value for key, value in config.items() if key != "density_matrix"},
            "density_matrices_k": "density_k.npy",
            "kpoint_fractional_coordinates": "kpoints.npy",
            "column_translations": np.zeros((size, 3), dtype=int).tolist(),
        }
        (folder / "periodic.json").write_text(json_text(periodic_config), encoding="utf-8")
        periodic = load_problem(folder / "periodic.json")
        assert periodic["input_mode"] == "periodic_density"
        assert np.allclose(periodic["gram"], density)

        np.save(folder / "asymmetric.npy", density + np.triu(np.ones_like(density), 1))
        for bad, expect in (
            ({"density_matrix": "asymmetric.npy"}, "symmetric"),
            ({"density_matrix": "density.npy", "atom_a": {"columns": [0, 0]}}, "duplicate"),
            ({"density_matrix": "density.npy", "density_eigenvalues": "density.npy"}, "input forms"),
            ({"density_matrix": "density.npy", "atom_b": {"columns": [4, 5]}}, "disjoint"),
            ({"density_matrix": "density.npy", "atom_b": {
                "columns": pair_b.tolist(), "lattice_translation": [1, 0, 0]
            }}, "R=(0,0,0)"),
        ):
            (folder / "bad.json").write_text(json_text({**config, **bad}), encoding="utf-8")
            try:
                load_problem(folder / "bad.json")
            except ValueError as exc:
                assert expect in str(exc), f"{expect} not in {exc}"
            else:  # pragma: no cover
                raise AssertionError(f"bad config accepted: {bad}")

        negative = density.copy()
        negative[0, 0] -= float(np.linalg.eigvalsh(density)[0]) + 1.0
        np.save(folder / "negative.npy", negative)
        (folder / "bad.json").write_text(
            json_text({**config, "density_matrix": "negative.npy"}), encoding="utf-8"
        )
        try:
            load_problem(folder / "bad.json")
        except ValueError as exc:
            assert "positive semi-definite" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("indefinite density matrix accepted")

    print("principal density channel v2 self-test: PASS")


def check_expected(problem: dict, result: dict) -> None:
    expected = problem["config"].get("expected")
    if not expected:
        raise ValueError("Reference test requires an expected section in the config")
    tolerance = float(expected.get("absolute_tolerance", 1.0e-12))
    comparisons = [
        ("singular_values", result["singular_values"]),
        ("squared_coherence_fractions", result["fractions"]),
        ("canonical_correlations", result["canonical"]),
    ]
    for key, observed in comparisons:
        if key not in expected:
            continue
        target = np.asarray(expected[key], dtype=float)
        if target.shape != observed.shape or not np.allclose(observed, target, atol=tolerance, rtol=0.0):
            raise AssertionError(f"Reference mismatch for {key}: {observed.tolist()} != {target.tolist()}")
    # v2 additionally pins the subspaces, not just the spectrum
    for key, vectors in (("atom_a_subspace", result["u"]), ("atom_b_subspace", result["v"])):
        if key not in expected:
            continue
        target = np.asarray(expected[key], dtype=float)
        angles = principal_angles(np.linalg.qr(target)[0], np.linalg.qr(vectors)[0])
        if angles.max() > float(expected.get("subspace_angle_tolerance_deg", 1.0e-6)):
            raise AssertionError(f"Reference subspace mismatch for {key}: max angle {angles.max():.3e} deg")


def _load_channel_run(path: Path) -> dict:
    path = path.resolve()
    with np.load(path / "channel_coefficients.npz", allow_pickle=False) as data:
        result = {
            "path": path,
            "columns_a": data["atom_a_columns"].copy(),
            "columns_b": data["atom_b_columns"].copy(),
            "labels_a": (data["atom_a_coordinate_labels"].astype(str).copy()
                         if "atom_a_coordinate_labels" in data.files else None),
            "labels_b": (data["atom_b_coordinate_labels"].astype(str).copy()
                         if "atom_b_coordinate_labels" in data.files else None),
            "u": data["atom_a_svd_coefficients"].copy(),
            "v": data["atom_b_svd_coefficients"].copy(),
            "singular_values": data["singular_values"].copy(),
            "interpretable": (
                data["interpretable"].astype(bool).copy()
                if "interpretable" in data.files
                else np.ones(data["singular_values"].size, dtype=bool)
            ),
        }
    result["relative_gaps"] = _relative_spectral_gaps(result["singular_values"])
    return result


def _best_assignment(score: np.ndarray) -> tuple[int, ...]:
    """Maximum-weight one-to-one channel assignment.

    Hungarian (scipy) when available; otherwise brute force, which is only sound
    for a handful of channels. The pair blocks in this project give three
    channels (O 2p) or six (Ru-Ru), but n! explodes past that -- 12 channels is
    4.8e8 permutations -- so the fallback refuses rather than silently hanging.
    """
    count = score.shape[0]
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        _, order = linear_sum_assignment(score, maximize=True)
        return tuple(int(value) for value in order)
    except ImportError:
        if count > 8:
            raise RuntimeError(
                f"{count} channels need Hungarian assignment but scipy is unavailable; "
                "brute force is refused above 8 channels"
            ) from None
        return max(
            itertools.permutations(range(count)),
            key=lambda candidate: sum(score[index, candidate[index]] for index in range(count)),
        )


def match_channels(first: Path, second: Path, degeneracy_threshold: float = 0.10) -> dict:
    """Match raw channels in their fixed atomic IAO-label coordinates."""
    a, b = _load_channel_run(first), _load_channel_run(second)
    label_metadata = (a["labels_a"], a["labels_b"], b["labels_a"], b["labels_b"])
    if any(labels is not None for labels in label_metadata):
        if any(labels is None for labels in label_metadata) or not np.array_equal(
            a["labels_a"], b["labels_a"]
        ) or not np.array_equal(a["labels_b"], b["labels_b"]):
            raise ValueError("runs do not use identical atom A/B IAO coordinate labels")
    elif not np.array_equal(a["columns_a"], b["columns_a"]) or not np.array_equal(
        a["columns_b"], b["columns_b"]
    ):
        raise ValueError("legacy runs do not use identical atom A/B local columns")
    if a["u"].shape != b["u"].shape or a["v"].shape != b["v"].shape:
        raise ValueError(f"channel coefficient shapes differ: {a['u'].shape}/{a['v'].shape} vs "
                         f"{b['u'].shape}/{b['v'].shape}")
    if not np.isfinite(degeneracy_threshold) or degeneracy_threshold <= 0.0:
        raise ValueError("degeneracy_threshold must be finite and positive")

    # Each channel is the direct-sum vector (u_i, v_i)/sqrt(2). A simultaneous
    # sign flip is its only non-degenerate gauge freedom.
    joint_overlap = 0.5 * (a["u"].T @ b["u"] + a["v"].T @ b["v"])
    count = joint_overlap.shape[0]
    assignment = _best_assignment(np.abs(joint_overlap))
    signs = [1 if joint_overlap[index, assigned] >= 0.0 else -1
             for index, assigned in enumerate(assignment)]

    matches = []
    for index, (assigned, sign) in enumerate(zip(assignment, signs)):
        gap_a = a["relative_gaps"][index]
        gap_b = b["relative_gaps"][assigned]
        trackable = bool(
            a["interpretable"][index]
            and b["interpretable"][assigned]
            and (gap_a is None or gap_a >= degeneracy_threshold)
            and (gap_b is None or gap_b >= degeneracy_threshold)
        )
        matches.append(
            {
                "run_a_channel": index + 1,
                "run_b_channel": assigned + 1,
                "simultaneous_sign_for_run_b": sign,
                "absolute_joint_overlap": float(abs(joint_overlap[index, assigned])),
                "aligned_atom_a_overlap": float(sign * (a["u"][:, index] @ b["u"][:, assigned])),
                "aligned_atom_b_overlap": float(sign * (a["v"][:, index] @ b["v"][:, assigned])),
                "singular_value_run_a": float(a["singular_values"][index]),
                "singular_value_run_b": float(b["singular_values"][assigned]),
                "singular_value_shift": float(
                    b["singular_values"][assigned] - a["singular_values"][index]
                ),
                "relative_spectral_gap_run_a": gap_a,
                "relative_spectral_gap_run_b": gap_b,
                "individually_trackable": trackable,
            }
        )

    groups, start = [], 0
    for index in range(count - 1):
        assigned_i, assigned_j = assignment[index], assignment[index + 1]
        gap_a = abs(a["singular_values"][index] - a["singular_values"][index + 1]) / max(
            a["singular_values"][index], np.finfo(float).tiny
        )
        gap_b = abs(b["singular_values"][assigned_i] - b["singular_values"][assigned_j]) / max(
            b["singular_values"][assigned_i], b["singular_values"][assigned_j], np.finfo(float).tiny
        )
        if min(gap_a, gap_b) >= degeneracy_threshold:
            if index + 1 - start > 1:
                groups.append(list(range(start, index + 1)))
            start = index + 1
    if count - start > 1:
        groups.append(list(range(start, count)))

    procrustes = []
    for group in groups:
        assigned = [assignment[index] for index in group]
        p = np.vstack((a["u"][:, group], a["v"][:, group])) / np.sqrt(2.0)
        q = np.vstack((b["u"][:, assigned], b["v"][:, assigned])) / np.sqrt(2.0)
        left, _, right_t = np.linalg.svd(q.T @ p)
        procrustes.append(
            {
                "run_a_channels": [index + 1 for index in group],
                "run_b_channels": [index + 1 for index in assigned],
                "principal_angles_deg": principal_angles(p, q).tolist(),
                "rotation_run_b_to_run_a": (left @ right_t).tolist(),
            }
        )

    return {
        "status": "pass",
        "run_a": str(a["path"]),
        "run_b": str(b["path"]),
        "coordinate_system": "fixed atomic IAO labels",
        "degeneracy_threshold": degeneracy_threshold,
        "absolute_joint_overlap_matrix": np.abs(joint_overlap).tolist(),
        "matches": matches,
        "near_degenerate_groups_procrustes": procrustes,
        "note": (
            "Individual coefficient comparisons require individually_trackable=true. "
            "Near-degenerate groups are compared only as Procrustes-aligned subspaces."
        ),
    }


def run_channel_matcher(first: Path, second: Path, output: Path | None,
                        degeneracy_threshold: float) -> None:
    text = json_text(match_channels(first, second, degeneracy_threshold))
    if output:
        output = output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing matcher output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run the full solve and all checks, write nothing (v1 called this --preflight)",
    )
    parser.add_argument("--reference-test", action="store_true")
    parser.add_argument("--match-channels", nargs=2, type=Path, metavar=("RUN_A", "RUN_B"))
    parser.add_argument("--degeneracy-threshold", type=float, default=0.10)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.match_channels:
        run_channel_matcher(
            args.match_channels[0], args.match_channels[1], args.output,
            args.degeneracy_threshold,
        )
        return
    if args.config is None:
        parser.error("--config is required")

    problem = load_problem(args.config)
    result = compute(problem)
    validation = validate_result(problem, result)

    if args.reference_test:
        check_expected(problem, result)
        print("principal density channel v2 reference test: PASS")
        return
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "singular_values": result["singular_values"].tolist(),
                    "numerical_rank": result["rank"],
                    "minimum_gauge_margin": float(result["gauge_margins"].min()),
                },
                indent=2,
                allow_nan=False,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required unless --validate-only/--reference-test/--self-test")
    save_outputs(problem, result, validation, args.output)
    print(f"PAW-IAO-PDC v2 output -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
