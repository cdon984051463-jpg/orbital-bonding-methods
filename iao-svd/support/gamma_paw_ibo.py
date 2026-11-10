"""Support routines retained from the established project implementation."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from ase.io import read as ase_read
from paw import nonlq, pawpotcar
from scipy.ndimage import map_coordinates
import paw_aware_region_modes as paw_core
from vasp_readers import Wavecar, parse_poscar

def assert_required_files(directory: Path, names: list[str]) -> None:
    missing = [
        name
        for name in names
        if not (directory / name).exists()
        or (directory / name).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"{directory} is missing required nonempty files: {missing}"
        )


def phase_fix_coefficients(
    coefficients: np.ndarray,
    centered_grid: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fix each atomic-orbital phase by its largest reciprocal coefficient."""

    result = coefficients.copy()
    grid = centered_grid.copy()
    maximum_phase_residual = 0.0
    for orbital in range(len(result)):
        pivot = int(np.argmax(np.abs(result[orbital])))
        value = result[orbital, pivot]
        if abs(value) == 0.0:
            raise ValueError(f"atomic reference orbital {orbital + 1} is zero")
        phase = np.exp(-1j * np.angle(value))
        result[orbital] *= phase
        grid[orbital] *= phase
        maximum_phase_residual = max(
            maximum_phase_residual,
            float(abs(result[orbital, pivot].imag)),
        )
    return grid, maximum_phase_residual


def load_centered_atomic_template(
    template_dir: Path,
    orbital_count: int,
    occupation_tolerance: float = 1.0e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read majority-spin atomic orbitals and center their Fourier phases."""

    assert_required_files(template_dir, ["POSCAR", "POTCAR", "WAVECAR"])
    structure = parse_poscar(template_dir / "POSCAR")
    if len(structure.sites) != 1:
        raise ValueError(f"atomic template is not one atom: {template_dir}")
    center_fractional = np.asarray(
        structure.sites[0].fractional, dtype=np.float64
    )
    with Wavecar(template_dir / "WAVECAR") as wav:
        if wav.nk != 1 or not np.allclose(wav.kinfo[0][1], 0.0, atol=1.0e-12):
            raise ValueError(f"atomic template is not Gamma-only: {template_dir}")
        if wav.nspin != 2:
            raise ValueError(
                f"atomic template must be spin-polarized: {template_dir}"
            )
        occupations = np.asarray(wav.kinfo[0][3], dtype=np.float64)
        maximum_occupation_error = float(
            np.max(np.abs(occupations[:orbital_count] - 1.0))
        )
        if maximum_occupation_error > occupation_tolerance:
            raise ValueError(
                f"first {orbital_count} majority orbitals are not occupied "
                f"in {template_dir}: {maximum_occupation_error:.3e}"
            )
        gvec = wav.gvecs(0)
        shape = tuple(int(value) for value in wav.ng)
        raw = np.empty((orbital_count, len(gvec)), dtype=np.complex128)
        grid = np.zeros((orbital_count, *shape), dtype=np.complex128)
        center_phase = np.exp(
            2j * np.pi * (gvec @ center_fractional)
        )
        for orbital in range(orbital_count):
            raw[orbital] = wav.coeffs(0, orbital)
            grid[
                orbital,
                gvec[:, 0] % shape[0],
                gvec[:, 1] % shape[1],
                gvec[:, 2] % shape[2],
            ] = raw[orbital] * center_phase
        grid, maximum_phase_residual = phase_fix_coefficients(
            raw * center_phase[None, :], grid
        )

        exact_coordinates = (gvec % np.asarray(shape)[None, :]).T
        roundtrip = np.empty_like(raw)
        for orbital in range(orbital_count):
            roundtrip[orbital] = map_coordinates(
                grid[orbital].real,
                exact_coordinates,
                order=3,
                mode="grid-wrap",
                prefilter=True,
            ) + 1j * map_coordinates(
                grid[orbital].imag,
                exact_coordinates,
                order=3,
                mode="grid-wrap",
                prefilter=True,
            )
        centered_raw = np.empty_like(raw)
        for orbital in range(orbital_count):
            values = grid[
                orbital,
                gvec[:, 0] % shape[0],
                gvec[:, 1] % shape[1],
                gvec[:, 2] % shape[2],
            ]
            centered_raw[orbital] = values
        denominator = max(1.0e-30, float(np.max(np.abs(centered_raw))))
        roundtrip_relative_error = float(
            np.max(np.abs(roundtrip - centered_raw)) / denominator
        )
        diagnostics = {
            "directory": str(template_dir),
            "element": structure.sites[0].element,
            "orbital_count": orbital_count,
            "majority_spin_occupations": occupations[:orbital_count],
            "maximum_occupation_error_from_1": maximum_occupation_error,
            "lattice_A": np.asarray(wav.A),
            "volume_A3": abs(float(wav.vol)),
            "encut_eV": float(wav.encut),
            "FFT_grid": shape,
            "plane_wave_count": len(gvec),
            "center_fractional": center_fractional,
            "phase_fix_maximum_pivot_imaginary": maximum_phase_residual,
            "cubic_integer_node_roundtrip_relative_max_error": (
                roundtrip_relative_error
            ),
        }
        if roundtrip_relative_error > 1.0e-10:
            raise RuntimeError(
                f"atomic interpolation roundtrip failed: "
                f"{roundtrip_relative_error:.3e}"
            )
        return grid, diagnostics


def interpolate_template_to_main_g(
    centered_grid: np.ndarray,
    template_lattice: np.ndarray,
    main_gvec: np.ndarray,
    main_reciprocal: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cubic-interpolate centered atomic Fourier amplitudes at main G."""

    reference_reciprocal = np.linalg.inv(template_lattice).T
    physical_cycles_per_A = main_gvec @ main_reciprocal
    reference_indices = (
        physical_cycles_per_A @ np.linalg.inv(reference_reciprocal)
    )
    shape = np.asarray(centered_grid.shape[1:], dtype=np.int64)
    coordinates = (reference_indices % shape[None, :]).T
    result = np.empty(
        (centered_grid.shape[0], len(main_gvec)), dtype=np.complex128
    )
    for orbital in range(centered_grid.shape[0]):
        result[orbital] = map_coordinates(
            centered_grid[orbital].real,
            coordinates,
            order=3,
            mode="grid-wrap",
            prefilter=True,
        ) + 1j * map_coordinates(
            centered_grid[orbital].imag,
            coordinates,
            order=3,
            mode="grid-wrap",
            prefilter=True,
        )
    half_shape = 0.5 * shape
    margin = half_shape[None, :] - np.abs(reference_indices)
    diagnostics = {
        "interpolation": "cubic periodic reciprocal-grid interpolation",
        "main_G_count": len(main_gvec),
        "reference_index_abs_max_by_axis": np.max(
            np.abs(reference_indices), axis=0
        ),
        "reference_half_grid_by_axis": half_shape,
        "minimum_index_margin_to_reference_grid_edge": float(np.min(margin)),
        "finite": bool(np.all(np.isfinite(result))),
    }
    if not diagnostics["finite"]:
        raise RuntimeError("interpolated atomic reference contains non-finite values")
    if diagnostics["minimum_index_margin_to_reference_grid_edge"] < 0.0:
        raise RuntimeError(
            "main plane-wave sphere exceeds the atomic reference FFT grid"
        )
    return result, diagnostics


def atomic_template_paw_identity_test(
    template_dir: Path,
    centered_grid: np.ndarray,
    template_diagnostics: dict[str, Any],
    orbital_count: int,
) -> dict[str, Any]:
    """Re-embed an atomic template into itself and check the PAW metric."""

    structure = parse_poscar(template_dir / "POSCAR")
    center = np.asarray(
        structure.sites[0].fractional, dtype=np.float64
    )
    atoms = ase_read(template_dir / "POSCAR")
    paw_datasets = parse_paw_datasets(template_dir / "POTCAR")
    with Wavecar(template_dir / "WAVECAR") as wav:
        gvec = wav.gvecs(0)
        base, interpolation = interpolate_template_to_main_g(
            centered_grid,
            np.asarray(
                template_diagnostics["lattice_A"], dtype=np.float64
            ),
            gvec,
            np.asarray(wav.B, dtype=np.float64),
        )
        reference = base * np.exp(
            -2j * np.pi * (gvec @ center)
        )[None, :]
        occupied = np.stack(
            [
                np.asarray(wav.coeffs(0, orbital), dtype=np.complex128)
                for orbital in range(orbital_count)
            ]
        )
        qengine = nonlq(
            atoms,
            wav.encut,
            paw_datasets,
            k=np.zeros(3, dtype=np.float64),
        )
        if not np.array_equal(gvec, qengine.Gvec):
            raise RuntimeError(
                "atomic template WAVECAR and PAW G-vector orders differ"
            )
        occupied_overlap = occupied.conj() @ occupied.T
        reference_overlap = reference.conj() @ reference.T
        cross_overlap = occupied.conj() @ reference.T
        projector = paw_core.atom_projector_matrix(qengine, 0)
        correction = np.asarray(
            paw_datasets[0].get_Qij(), dtype=np.complex128
        )
        occupied_beta = occupied @ projector.T
        reference_beta = reference @ projector.T
        occupied_overlap += (
            occupied_beta.conj()
            @ correction
            @ occupied_beta.T
        )
        reference_overlap += (
            reference_beta.conj()
            @ correction
            @ reference_beta.T
        )
        cross_overlap += (
            occupied_beta.conj()
            @ correction
            @ reference_beta.T
        )
    occupied_error = float(
        np.max(np.abs(occupied_overlap - np.eye(orbital_count)))
    )
    reference_error = float(
        np.max(np.abs(reference_overlap - np.eye(orbital_count)))
    )
    singular_values = np.linalg.svd(
        cross_overlap, compute_uv=False
    )
    checks = {
        "occupied_PAW_orthonormality_below_1e-5": occupied_error < 1.0e-5,
        "reembedded_reference_PAW_orthonormality_below_1e-5": (
            reference_error < 1.0e-5
        ),
        "cross_overlap_is_unitary_below_1e-5": float(
            np.max(np.abs(singular_values - 1.0))
        )
        < 1.0e-5,
        "interpolation_inside_grid": (
            interpolation[
                "minimum_index_margin_to_reference_grid_edge"
            ]
            >= 0.0
        ),
    }
    return {
        "occupied_PAW_orthonormality_max_abs_error": occupied_error,
        "reference_PAW_orthonormality_max_abs_error": reference_error,
        "cross_overlap_singular_values": singular_values,
        "cross_overlap_singular_value_max_error_from_1": float(
            np.max(np.abs(singular_values - 1.0))
        ),
        "interpolation": interpolation,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def parse_paw_datasets(path: Path) -> list[pawpotcar]:
    text = path.read_text(errors="replace")
    return [
        pawpotcar(dataset)
        for dataset in text.split("End of Dataset")[:-1]
    ]


def inverse_sqrt_hermitian(
    matrix: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    cutoff = rcond * float(eigenvalues[-1])
    rank = int(np.count_nonzero(eigenvalues > cutoff))
    if rank != len(eigenvalues):
        raise np.linalg.LinAlgError(
            f"matrix rank {rank}/{len(eigenvalues)} at rcond {rcond:g}"
        )
    inverse_sqrt = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.conj().T
    transformed = inverse_sqrt @ matrix @ inverse_sqrt
    diagnostics = {
        "rank": rank,
        "eigenvalue_range": [
            float(eigenvalues[0]),
            float(eigenvalues[-1]),
        ],
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "normalized_max_abs_error": float(
            np.max(np.abs(transformed - np.eye(len(matrix))))
        ),
    }
    return inverse_sqrt, diagnostics


def compact_joint_coordinates(
    occupied_overlap: np.ndarray,
    cross_overlap: np.ndarray,
    reference_overlap: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    occupied_lowdin, occupied_diag = inverse_sqrt_hermitian(
        occupied_overlap, rcond
    )
    normalized_cross = occupied_lowdin @ cross_overlap
    noccupied = len(occupied_overlap)
    joint = np.block(
        [
            [np.eye(noccupied, dtype=np.complex128), normalized_cross],
            [normalized_cross.conj().T, reference_overlap],
        ]
    )
    joint = 0.5 * (joint + joint.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(joint)
    largest = float(eigenvalues[-1])
    cutoff = rcond * largest
    severe_negative = float(eigenvalues[0])
    if severe_negative < -100.0 * cutoff:
        raise np.linalg.LinAlgError(
            f"joint PAW Gram has negative eigenvalue {severe_negative:.3e}"
        )
    keep = eigenvalues > cutoff
    rank = int(np.count_nonzero(keep))
    kept_values = eigenvalues[keep]
    kept_vectors = eigenvectors[:, keep]
    coordinates = (
        np.sqrt(kept_values)[:, None] * kept_vectors.conj().T
    )
    reconstruction = coordinates.conj().T @ coordinates
    occupied_coordinates = coordinates[:, :noccupied]
    reference_coordinates = coordinates[:, noccupied:]
    diagnostics = {
        "occupied_lowdin": occupied_diag,
        "joint_dimension": len(joint),
        "joint_rank": rank,
        "joint_eigenvalue_range": [
            float(eigenvalues[0]),
            largest,
        ],
        "joint_cutoff": cutoff,
        "discarded_eigenvalue_count": int(len(joint) - rank),
        "joint_gram_reconstruction_max_abs_error": float(
            np.max(np.abs(reconstruction - joint))
        ),
        "occupied_coordinate_orthonormality_max_abs_error": float(
            np.max(
                np.abs(
                    occupied_coordinates.conj().T
                    @ occupied_coordinates
                    - np.eye(noccupied)
                )
            )
        ),
    }
    return (
        occupied_coordinates,
        reference_coordinates,
        occupied_lowdin,
        diagnostics,
    )
