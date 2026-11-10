"""Support routines retained from the established project implementation."""
from __future__ import annotations
from pathlib import Path
import numpy as np

ATOMIC_NUMBERS = {"H": 1, "O": 8, "Ru": 44}


def read_ibzkpt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    nk = int(lines[1].split()[0])
    rows = [line.split() for line in lines[3 : 3 + nk]]
    kvecs = np.array([[float(x) for x in row[:3]] for row in rows], dtype=np.float64)
    multiplicities = np.array([float(row[3]) for row in rows], dtype=np.float64)
    weights = multiplicities / multiplicities.sum()
    return kvecs, multiplicities, weights


def grid_fractional(shape: tuple[int, int, int]) -> np.ndarray:
    axes = [np.arange(n, dtype=np.float64) / n for n in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
