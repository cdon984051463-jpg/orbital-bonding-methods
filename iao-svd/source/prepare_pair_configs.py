#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""Turn full-cell IAO densities into PDC v2 pair configurations.

For each (structure, Ru, O) target this names the IAO columns of the two bonded
atoms plus an enlarged context shell.

Two output forms, both accepted by the core:

``--emit density`` (default)
    Writes only a JSON that points at the structure's shared weighted D(k)
    matrices and records the lattice image of every IAO column. The channels are
    the SVD of the Fourier-transformed D[A(0), B(R)], so minimum-image bonds are
    not silently replaced by their same-cell block.

``--emit basis``
    The original four-array eigen-coordinate form (eigenvalues, atomic basis,
    environment basis, full basis, ~3 MB per pair). Kept because the accepted
    q214 audit trail and every run before 2026-08-05 use it; it is an equivalent
    rewriting of the same problem (see ../README.md for the definition).

The context shell exists only to supply honest denominators and the multicentre
star diagnostic. The channels themselves are always the strict A-B cross block.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from periodic_density import minimum_image

ANGSTROM = 1.0




def read_poscar(path: Path):
    lines = path.read_text().splitlines()
    scale = float(lines[1])
    cell = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * scale
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    index = 7
    if lines[index].strip().lower().startswith("s"):
        index += 1
    direct = lines[index].strip().lower().startswith("d")
    index += 1
    total = sum(counts)
    raw = np.array([[float(x) for x in lines[index + j].split()[:3]] for j in range(total)])
    positions = raw @ cell if direct else raw
    symbols = [element for element, count in zip(elements, counts) for _ in range(count)]
    return cell, positions, symbols


def atom_basis_map(labels_path: Path) -> tuple[dict[int, list[int]], dict[int, list[str]]]:
    mapping: dict[int, list[tuple[int, str]]] = {}
    with labels_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            atom = int(row["atom_index_1based"])
            label = ":".join((row["element"], row["active_shell"], row["atomic_band_index_1based"]))
            mapping.setdefault(atom, []).append((int(row["reference_column"]), label))
    ordered = {atom: sorted(entries) for atom, entries in mapping.items()}
    return (
        {atom: [column for column, _ in entries] for atom, entries in ordered.items()},
        {atom: [label for _, label in entries] for atom, entries in ordered.items()},
    )


def _relative(target: Path, start: Path) -> str:
    """POSIX-style relative path, falling back to absolute across drives."""
    try:
        return Path(os.path.relpath(target, start)).as_posix()
    except ValueError:
        return str(target)


def build(
    density_dir: Path,
    calc: Path,
    atom_a: int,
    atom_b: int,
    context_radius: float,
    output: Path,
    structure_label: str,
    pair_label: str,
    emit: str = "density",
) -> dict:
    columns, coordinate_labels = atom_basis_map(density_dir / "iao_labels.csv")
    cell, positions, symbols = read_poscar(calc / "CONTCAR")
    width = sum(len(item) for item in columns.values())

    # context = every other atom whose distance to EITHER bonded atom is inside the
    # radius, so a multicentre partner of Ru or of O is caught symmetrically
    pair_displacement, pair_translation = minimum_image(
        cell, positions[atom_b - 1] - positions[atom_a - 1]
    )
    distances = {}
    atom_translations = {}
    for atom in columns:
        delta_a, translation_a = minimum_image(
            cell, positions[atom - 1] - positions[atom_a - 1]
        )
        delta_b, translation_from_b = minimum_image(
            cell, positions[atom - 1] - positions[atom_b - 1]
        )
        d_a, d_b = np.linalg.norm(delta_a), np.linalg.norm(delta_b)
        distances[atom] = min(d_a, d_b)
        atom_translations[atom] = (
            translation_a if d_a <= d_b else pair_translation + translation_from_b
        )
    atom_translations[atom_a] = np.zeros(3, dtype=int)
    atom_translations[atom_b] = pair_translation
    context_atoms = sorted(
        atom for atom, distance in distances.items()
        if atom not in (atom_a, atom_b) and distance <= context_radius
    )

    selected = [atom_a, atom_b] + context_atoms
    selected_columns: list[int] = []
    ranges: dict[int, tuple[int, int]] = {}
    for atom in selected:
        start = len(selected_columns)
        selected_columns.extend(columns[atom])
        ranges[atom] = (start, len(selected_columns))
    selected_set = set(selected_columns)
    remaining = [c for c in range(width) if c not in selected_set]
    column_translations = np.zeros((width, 3), dtype=int)
    for atom, atom_columns in columns.items():
        column_translations[atom_columns] = atom_translations[atom]

    output.mkdir(parents=True, exist_ok=True)
    if emit == "density":
        inputs = {
            "density_matrices_k": density_dir / "iao_density_matrices_k.npy",
            "kpoint_fractional_coordinates": density_dir / "kpoint_fractional_coordinates.npy",
        }
        missing = [str(path) for path in inputs.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "periodic pair input is missing " + ", ".join(missing)
                + "; rerun build_iao_density.py"
            )
        selectors = {
            atom: {
                "columns": columns[atom],
                "coordinate_labels": coordinate_labels[atom],
                "lattice_translation": atom_translations[atom].tolist(),
            }
            for atom in selected
        }
        source = {
            **{key: _relative(path, output) for key, path in inputs.items()},
            "column_translations": column_translations.tolist(),
        }
    else:
        shifted = {
            atom: translation.tolist() for atom, translation in atom_translations.items()
            if atom in selected and np.any(translation)
        }
        if shifted:
            raise ValueError(
                "--emit basis cannot represent lattice-image atoms; use --emit density. "
                f"Non-zero translations: {shifted}"
            )
        values = np.load(density_dir / "density_eigenvalues.npy")
        basis = np.load(density_dir / "iao_basis.npy")
        if basis.shape[0] != values.size or basis.shape[1] != values.size:
            raise ValueError(f"basis {basis.shape} inconsistent with {values.size} eigenvalues")
        reordered = basis[:, selected_columns + remaining]
        atomic = reordered[:, : len(selected_columns)]
        environment = reordered[:, len(selected_columns) :]
        np.save(output / "density_eigenvalues.npy", values, allow_pickle=False)
        np.save(output / "atomic_basis.npy", np.ascontiguousarray(atomic), allow_pickle=False)
        np.save(output / "environment_basis.npy", np.ascontiguousarray(environment),
                allow_pickle=False)
        np.save(output / "full_basis.npy", np.ascontiguousarray(reordered), allow_pickle=False)
        input_files = {
            "density_eigenvalues": output / "density_eigenvalues.npy",
            "atomic_basis": output / "atomic_basis.npy",
            "environment_basis": output / "environment_basis.npy",
            "full_basis": output / "full_basis.npy",
        }
        selectors = {
            atom: {"column_range": list(ranges[atom]),
                   "coordinate_labels": coordinate_labels[atom],
                   "lattice_translation": [0, 0, 0]}
            for atom in selected
        }
        source = {
            "density_eigenvalues": "density_eigenvalues.npy",
            "atomic_basis": "atomic_basis.npy",
            "environment_basis": "environment_basis.npy",
            "full_basis": "full_basis.npy",
        }

    config = {
        **source,
        "atom_a": {"label": f"{symbols[atom_a-1]}{atom_a}", **selectors[atom_a]},
        "atom_b": {"label": f"{symbols[atom_b-1]}{atom_b}", **selectors[atom_b]},
        "context_atoms": [
            {"label": f"{symbols[atom-1]}{atom}", **selectors[atom]}
            for atom in context_atoms
        ],
        "input_form": emit,
        "periodic_image_convention": (
            "D[mu(0),nu(R)] = sum_k exp(-2*pi*i*k.R) D_k[mu,nu]"
        ),
        "iao_density_directory": str(density_dir),
        "orthogonality_tolerance": 1.0e-9,
        "rank_relative_tolerance": 1.0e-12,
        "rank_absolute_floor": 1.0e-14,
        "gauge_margin_warning": 0.05,
        "structure_label": structure_label,
        "pair_label": pair_label,
        "bond_length_angstrom": float(
            np.linalg.norm(pair_displacement)
        ),
        "pair_lattice_translation": pair_translation.tolist(),
        "context_radius_angstrom": context_radius,
        "context_distances_angstrom": {
            f"{symbols[atom-1]}{atom}": float(distances[atom]) for atom in context_atoms
        },
        "atom_indices_1based": {"atom_a": atom_a, "atom_b": atom_b, "context": context_atoms},
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-dir", type=Path, required=True)
    parser.add_argument("--calc", type=Path, required=True)
    parser.add_argument("--atom-a", type=int, required=True)
    parser.add_argument("--atom-b", type=int, required=True)
    parser.add_argument("--context-radius", type=float, default=3.6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structure-label", default="")
    parser.add_argument("--pair-label", default="")
    parser.add_argument("--emit", choices=("density", "basis"), default="density",
                        help="density: name columns of the shared IAO density matrix (default); "
                             "basis: the legacy four-array eigen-coordinate form")
    args = parser.parse_args()
    config = build(
        args.density_dir.resolve(), args.calc.resolve(), args.atom_a, args.atom_b,
        args.context_radius, args.output.resolve(), args.structure_label, args.pair_label,
        args.emit,
    )
    print(json.dumps({k: config[k] for k in
                      ("pair_label", "bond_length_angstrom", "atom_a", "atom_b")}, indent=2))
    print(f"context atoms: {[c['label'] for c in config['context_atoms']]}")


if __name__ == "__main__":
    main()
