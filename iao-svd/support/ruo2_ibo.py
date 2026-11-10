"""Support routines retained from the established project implementation."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np

Array = np.ndarray


def hermitian(matrix: Array) -> Array:
    """Return the Hermitian part of a square matrix."""

    return 0.5 * (matrix + matrix.conj().T)


def orthonormalize_columns(
    matrix: Array,
    *,
    rcond: float = 1.0e-12,
    label: str = "matrix",
) -> tuple[Array, dict[str, float | int]]:
    """Symmetrically orthonormalize a full-column-rank matrix.

    The symmetric Löwdin transform preserves the correspondence between input
    and output columns, which is required for atom labels on the IAO basis.
    """

    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
        raise ValueError(f"{label} must be a tall or square 2D matrix")
    gram = hermitian(matrix.conj().T @ matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = float(np.max(eigenvalues)) if len(eigenvalues) else 0.0
    cutoff = rcond * largest
    rank = int(np.count_nonzero(eigenvalues > cutoff))
    if rank != matrix.shape[1]:
        raise np.linalg.LinAlgError(
            f"{label} has numerical rank {rank}/{matrix.shape[1]} "
            f"at rcond={rcond:.1e}"
        )
    inverse_sqrt = (
        eigenvectors
        * (1.0 / np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.conj().T
    orthonormal = matrix @ inverse_sqrt
    error = float(
        np.max(
            np.abs(
                orthonormal.conj().T @ orthonormal
                - np.eye(matrix.shape[1])
            )
        )
    )
    return orthonormal, {
        "rank": rank,
        "columns": int(matrix.shape[1]),
        "smallest_gram_eigenvalue": float(eigenvalues[0]),
        "largest_gram_eigenvalue": largest,
        "condition_number": float(largest / eigenvalues[0]),
        "orthonormality_max_abs_error": error,
    }


def build_intrinsic_atomic_orbitals(
    reference_features: Array,
    occupied_orbitals: Array,
    *,
    rcond: float = 1.0e-12,
) -> tuple[Array, dict[str, Any]]:
    """Construct IAOs in an orthonormal finite working basis.

    ``reference_features`` contains atom-labelled minimal reference functions
    in columns. ``occupied_orbitals`` contains orthonormal occupied orbitals.
    The implementation follows the projector construction

        A_raw = [P P~ + (1-P)(1-P~)] F,

    where F is the Löwdin-orthonormalized reference basis, P is the occupied
    projector, and P~ is the occupied space depolarized into span(F).
    """

    reference_features = np.asarray(reference_features, dtype=np.complex128)
    occupied_orbitals = np.asarray(occupied_orbitals, dtype=np.complex128)
    if reference_features.ndim != 2 or occupied_orbitals.ndim != 2:
        raise ValueError("reference_features and occupied_orbitals must be 2D")
    if reference_features.shape[0] != occupied_orbitals.shape[0]:
        raise ValueError("reference and occupied matrices need equal row counts")
    if reference_features.shape[1] < occupied_orbitals.shape[1]:
        raise ValueError(
            "the minimal reference space must have at least noccupied columns"
        )

    occupied_error = float(
        np.max(
            np.abs(
                occupied_orbitals.conj().T @ occupied_orbitals
                - np.eye(occupied_orbitals.shape[1])
            )
        )
    )
    if occupied_error > 1.0e-9:
        raise ValueError(
            f"occupied orbitals are not orthonormal: {occupied_error:.3e}"
        )

    reference_orth, reference_diag = orthonormalize_columns(
        reference_features, rcond=rcond, label="reference feature matrix"
    )
    projected_occupied = reference_orth @ (
        reference_orth.conj().T @ occupied_orbitals
    )
    depolarized_occupied, depolarized_diag = orthonormalize_columns(
        projected_occupied,
        rcond=rcond,
        label="occupied projection into reference space",
    )

    occupied_projector = occupied_orbitals @ occupied_orbitals.conj().T
    depolarized_projector = (
        depolarized_occupied @ depolarized_occupied.conj().T
    )
    identity = np.eye(reference_features.shape[0], dtype=np.complex128)
    raw_iaos = (
        occupied_projector @ depolarized_projector
        + (identity - occupied_projector)
        @ (identity - depolarized_projector)
    ) @ reference_orth
    iaos, iao_diag = orthonormalize_columns(
        raw_iaos, rcond=rcond, label="raw IAO matrix"
    )

    iao_projector = iaos @ iaos.conj().T
    occupied_closure = (
        occupied_orbitals.conj().T
        @ iao_projector
        @ occupied_orbitals
    )
    closure_error = float(
        np.max(
            np.abs(
                occupied_closure - np.eye(occupied_orbitals.shape[1])
            )
        )
    )
    span_error = float(
        np.linalg.norm(
            (identity - iao_projector) @ occupied_orbitals,
            ord="fro",
        )
    )
    diagnostics = {
        "working_dimension": int(reference_features.shape[0]),
        "reference_count": int(reference_features.shape[1]),
        "occupied_count": int(occupied_orbitals.shape[1]),
        "occupied_input_orthonormality_max_abs_error": occupied_error,
        "reference_lowdin": reference_diag,
        "depolarized_occupied": depolarized_diag,
        "iao_lowdin": iao_diag,
        "occupied_projector_closure_max_abs_error": closure_error,
        "occupied_span_frobenius_residual": span_error,
    }
    return iaos, diagnostics


def make_atomic_population_factors(
    occupied_orbitals: Array,
    iaos: Array,
    atom_columns: Sequence[Sequence[int]],
) -> list[Array]:
    """Return low-rank factors B_A for Q_A = B_A B_A^dagger."""

    occupied_orbitals = np.asarray(occupied_orbitals, dtype=np.complex128)
    iaos = np.asarray(iaos, dtype=np.complex128)
    flattened = [int(index) for group in atom_columns for index in group]
    if sorted(flattened) != list(range(iaos.shape[1])):
        raise ValueError("atom_columns must partition every IAO column once")
    occupied_to_iao = occupied_orbitals.conj().T @ iaos
    return [
        occupied_to_iao[:, np.asarray(indices, dtype=np.int64)]
        for indices in atom_columns
    ]


def factor_closure_error(factors: Sequence[Array]) -> float:
    """Maximum error in sum_A Q_A = I over the selected orbital space."""

    if not factors:
        raise ValueError("at least one atomic population factor is required")
    size = factors[0].shape[0]
    closure = np.zeros((size, size), dtype=np.complex128)
    for factor in factors:
        if factor.ndim != 2 or factor.shape[0] != size:
            raise ValueError("all factors must have shape (norbital, rank_A)")
        closure += factor @ factor.conj().T
    return float(np.max(np.abs(closure - np.eye(size))))
