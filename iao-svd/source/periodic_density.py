#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""Fourier-transform weighted IAO density matrices between lattice images."""

from __future__ import annotations

import numpy as np


def minimum_image(cell: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the minimum-image displacement and its integer lattice translation."""
    fractional = np.asarray(delta, dtype=float) @ np.linalg.inv(cell)
    translation = -np.rint(fractional).astype(int)
    return (fractional + translation) @ cell, translation


def validate(density_k: np.ndarray, kpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    density_k = np.asarray(density_k)
    kpoints = np.asarray(kpoints, dtype=float)
    if (
        density_k.ndim != 3 or not density_k.shape[0] or not density_k.shape[1]
        or density_k.shape[1] != density_k.shape[2]
    ):
        raise ValueError(f"density_matrices_k must have shape (nk,n,n), got {density_k.shape}")
    if kpoints.shape != (density_k.shape[0], 3):
        raise ValueError(
            f"kpoint_fractional_coordinates must have shape ({density_k.shape[0]},3), "
            f"got {kpoints.shape}"
        )
    if not np.isfinite(density_k).all() or not np.isfinite(kpoints).all():
        raise ValueError("periodic density inputs contain NaN or infinity")
    hermiticity = float(np.max(np.abs(density_k - density_k.conj().transpose(0, 2, 1))))
    scale = max(float(np.max(np.abs(density_k))), 1.0)
    if hermiticity > 1.0e-10 * scale:
        raise ValueError(f"a D(k) block is not Hermitian: {hermiticity:.3e}")
    return density_k, kpoints


def density_for_translations(
    density_k: np.ndarray, kpoints: np.ndarray, column_translations: np.ndarray
) -> np.ndarray:
    """Build D between one chosen lattice image of every IAO column."""
    density_k, kpoints = validate(density_k, kpoints)
    translations = np.asarray(column_translations)
    if translations.shape != (density_k.shape[1], 3):
        raise ValueError(
            f"column_translations must have shape ({density_k.shape[1]},3), "
            f"got {translations.shape}"
        )
    if not np.issubdtype(translations.dtype, np.integer):
        if not np.isfinite(translations).all() or not np.allclose(translations, np.rint(translations)):
            raise ValueError("column_translations must contain integer lattice vectors")
        translations = np.rint(translations).astype(int)
    phase = np.exp(2j * np.pi * (kpoints @ translations.T))
    density = np.einsum("ki,kij,kj->ij", phase, density_k, phase.conj(), optimize=True)
    hermiticity = float(np.max(np.abs(density - density.conj().T)))
    scale = max(float(np.max(np.abs(density))), 1.0)
    if hermiticity > 1.0e-10 * scale:
        raise ValueError(f"translated density is not Hermitian: {hermiticity:.3e}")
    return np.ascontiguousarray(0.5 * (density.real + density.real.T))


def cross_block(
    density_k: np.ndarray,
    kpoints: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Return D[rows(0), columns(R)] with exp(-2 pi i k.R)."""
    density_k, kpoints = validate(density_k, kpoints)
    translation = np.asarray(translation)
    if translation.shape != (3,) or not np.allclose(translation, np.rint(translation)):
        raise ValueError(f"translation must be a three-integer lattice vector, got {translation}")
    phase = np.exp(-2j * np.pi * (kpoints @ np.rint(translation).astype(int)))
    block = np.einsum(
        "k,kij->ij", phase, density_k[:, np.asarray(rows)[:, None], np.asarray(columns)],
        optimize=True,
    )
    return np.ascontiguousarray(block.real)


def self_test() -> None:
    block = np.ones((2, 2), dtype=np.complex128) / 2.0
    density_k = np.stack((block, block))
    kpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert np.allclose(density_for_translations(density_k, kpoints, [[0, 0, 0], [0, 0, 0]]), 1.0)
    shifted = density_for_translations(density_k, kpoints, [[0, 0, 0], [1, 0, 0]])
    assert abs(shifted[0, 1]) < 1.0e-12
    assert abs(cross_block(density_k, kpoints, [0], [1], [1, 0, 0])[0, 0]) < 1.0e-12
    cell = np.eye(3) * 10.0
    displacement, translation = minimum_image(cell, [9.0, 0.0, 0.0])
    assert np.allclose(displacement, [-1.0, 0.0, 0.0])
    assert np.array_equal(translation, [-1, 0, 0])


if __name__ == "__main__":
    self_test()
    print("periodic density self-test: PASS")
