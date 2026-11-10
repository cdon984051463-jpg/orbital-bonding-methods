#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "scipy", "ase", "spglib", "PyVaspWfc @ git+https://github.com/QijingZheng/VaspBandUnfolding@3f8a6285b538cdec1d896e803e3d0800147fda88"]
# ///
"""Build bond-centred PDC coherence cubes with all k points closed in a BvK cell.

The full Born-von Karman field is reconstructed internally so distinct k points
are orthogonal on the integration grid.  Only a local crop around the named bond
is written: both signed partner amplitudes and their weighted product for each
PDC channel, plus the product sum.

These are density-weighted partner-product proxies, not bond-energy densities and
not a real-space partition of Q_AB.  Quantitative channel weights remain s_i^2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import build_channel_fields as fields
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'support'))
from vasp_readers import Wavecar


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def read_automatic_mesh(path: Path) -> tuple[int, int, int]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    if len(lines) < 4 or lines[1] != "0":
        raise ValueError(f"{path} is not an automatic KPOINTS mesh")
    try:
        mesh = tuple(int(value) for value in lines[3].split()[:3])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot read three mesh dimensions from {path}") from exc
    if len(mesh) != 3 or any(value <= 0 for value in mesh):
        raise ValueError(f"invalid KPOINTS mesh {mesh}")
    return mesh


def cell_translations(mesh: tuple[int, int, int]) -> np.ndarray:
    return np.stack(
        np.meshgrid(*(np.arange(size) for size in mesh), indexing="ij"), axis=-1
    ).reshape(-1, 3)


def validate_time_reversal_mesh(
    kpoints: np.ndarray, state_rows: list[dict[str, str]], mesh: tuple[int, int, int]
) -> dict:
    """Validate the current ISYM=0, time-reversal-reduced Monkhorst-Pack case."""
    kpoints = np.asarray(kpoints, dtype=float)
    full_count = math.prod(mesh)
    if kpoints.shape != (full_count // 2, 3) or full_count % 2:
        raise ValueError(
            f"expected half of an even {mesh} mesh ({full_count // 2} stored k points), "
            f"got {kpoints.shape}"
        )
    if np.any(np.max(np.abs(2.0 * kpoints - np.rint(2.0 * kpoints)), axis=1) < 1.0e-10):
        raise ValueError("this BvK builder currently requires a mesh with no TRIM points")

    canonical = lambda values: (values + 0.5) % 1.0 - 0.5
    full_k = canonical(np.vstack((kpoints, -kpoints)))
    unique = np.unique(np.round(full_k, 10), axis=0)
    if unique.shape[0] != full_count:
        raise ValueError("stored k points and their time reverses do not form the full mesh")

    translations = cell_translations(mesh)
    phase = np.exp(2j * np.pi * (full_k @ translations.T))
    orthogonality = phase @ phase.conj().T / full_count
    phase_error = float(np.max(np.abs(orthogonality - np.eye(full_count))))
    if phase_error > 1.0e-10:
        raise ValueError(f"k/translation phase table is not orthogonal: {phase_error:.3e}")

    weights = {}
    for row in state_rows:
        weights.setdefault(int(row["k_index"]) - 1, float(row["k_weight"]))
    if set(weights) != set(range(len(kpoints))):
        raise ValueError("state_map.csv does not cover every stored k point")
    multiplicities = np.asarray([weights[index] * full_count for index in range(len(kpoints))])
    if not np.allclose(multiplicities, 2.0, atol=1.0e-10, rtol=0.0):
        raise ValueError(f"expected time-reversal multiplicity 2, got {multiplicities.tolist()}")
    return {
        "mesh": list(mesh),
        "full_k_point_count": full_count,
        "stored_time_reversal_reduced_k_point_count": len(kpoints),
        "time_reversal_multiplicity": 2,
        "phase_orthogonality_max_abs_error": phase_error,
    }


def validate_explicit_full_mesh(
    kpoints: np.ndarray, state_rows: list[dict[str, str]], mesh: tuple[int, int, int]
) -> dict:
    """Validate an ISYM=1 table already expanded to every BvK k point."""
    kpoints = np.asarray(kpoints, dtype=float)
    full_count = math.prod(mesh)
    if kpoints.shape != (full_count, 3):
        raise ValueError(
            f"expected the complete {mesh} mesh ({full_count} k points), got {kpoints.shape}"
        )
    canonical = lambda values: (values + 0.5) % 1.0 - 0.5
    unique = np.unique(np.round(canonical(kpoints), 10), axis=0)
    if unique.shape[0] != full_count:
        raise ValueError("explicit ISYM=1 k-point table contains duplicates")
    translations = cell_translations(mesh)
    phase = np.exp(2j * np.pi * (kpoints @ translations.T))
    orthogonality = phase @ phase.conj().T / full_count
    phase_error = float(np.max(np.abs(orthogonality - np.eye(full_count))))
    if phase_error > 1.0e-10:
        raise ValueError(f"k/translation phase table is not orthogonal: {phase_error:.3e}")
    weights = {}
    for row in state_rows:
        weights.setdefault(int(row["k_index"]) - 1, float(row["k_weight"]))
    if set(weights) != set(range(full_count)):
        raise ValueError("state_map.csv does not cover every explicit full-mesh k point")
    expected_weight = 1.0 / full_count
    if not np.allclose(
        [weights[index] for index in range(full_count)], expected_weight,
        atol=1.0e-12, rtol=0.0,
    ):
        raise ValueError("expanded ISYM=1 k weights are not uniform 1/Nk")
    return {
        "mesh": list(mesh),
        "full_k_point_count": full_count,
        "stored_irreducible_k_point_count": len({
            int(row["source_k_index"]) for row in state_rows
        }),
        "explicit_space_group_and_time_reversal_expansion": True,
        "phase_orthogonality_max_abs_error": phase_error,
    }


def expand_amplitudes(
    amplitudes: np.ndarray,
    state_k: np.ndarray,
    kpoints: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    """Return cell-major amplitudes for real fields containing each k/-k pair."""
    phase = np.exp(2j * np.pi * (kpoints[state_k] @ translations.T))
    expanded = math.sqrt(2.0) * phase[:, :, None] * amplitudes[:, None, :]
    return expanded.reshape(amplitudes.shape[0], -1)


def expand_full_mesh_amplitudes(
    amplitudes: np.ndarray,
    state_k: np.ndarray,
    kpoints: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    """Return cell-major amplitudes when every k/-k state is already explicit."""
    phase = np.exp(2j * np.pi * (kpoints[state_k] @ translations.T))
    expanded = phase[:, :, None] * amplitudes[:, None, :]
    return expanded.reshape(amplitudes.shape[0], -1)


def assemble_supercell(
    cell_fields: np.ndarray,
    modes: int,
    shape: tuple[int, int, int],
    mesh: tuple[int, int, int],
    translations: np.ndarray,
) -> np.ndarray:
    blocks = cell_fields.reshape(len(translations), modes, *shape)
    full_shape = tuple(shape[axis] * mesh[axis] for axis in range(3))
    result = np.empty((modes, *full_shape), dtype=np.float64)
    for block, translation in zip(blocks, translations):
        slices = tuple(
            slice(int(translation[axis]) * shape[axis], (int(translation[axis]) + 1) * shape[axis])
            for axis in range(3)
        )
        result[(slice(None), *slices)] = block
    return result


def crop_and_integrate(cell_fields, channels, translations, slices, voxel):
    """Integrate every BvK cell, copying only the requested crop."""
    shape = np.asarray(cell_fields.shape[1:], dtype=int)
    start = np.asarray([part.start for part in slices])
    stop = np.asarray([part.stop for part in slices])
    crop = np.empty((2 * channels, *(stop - start)), dtype=np.float64)
    gram = np.zeros((2 * channels, 2 * channels))
    absolute_product = np.zeros(channels)
    for block, translation in zip(
        cell_fields.reshape(len(translations), 2 * channels, *shape), translations
    ):
        flat = block.reshape(2 * channels, -1)
        gram += (flat @ flat.T) * voxel
        absolute_product += np.sum(np.abs(flat[:channels] * flat[channels:]), axis=1) * voxel
        cell_start = translation * shape
        lower = np.maximum(start, cell_start)
        upper = np.minimum(stop, cell_start + shape)
        if np.any(upper <= lower):
            continue
        source = tuple(slice(int(a), int(b)) for a, b in zip(lower - cell_start, upper - cell_start))
        target = tuple(slice(int(a), int(b)) for a, b in zip(lower - start, upper - start))
        crop[(slice(None), *target)] = block[(slice(None), *source)]
    norms = np.sqrt(np.diag(gram))
    crop /= norms[:, None, None, None]
    return (
        crop, gram / np.outer(norms, norms), norms,
        absolute_product / (norms[:channels] * norms[channels:]),
    )


def crop_slices(
    target_positions: np.ndarray,
    padding: float,
    supercell_lattice: np.ndarray,
    full_shape: tuple[int, int, int],
) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    low = target_positions.min(axis=0) - padding
    high = target_positions.max(axis=0) + padding
    corners = np.array(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1])
         for z in (low[2], high[2])]
    )
    frac = corners @ np.linalg.inv(supercell_lattice)
    start = np.floor(frac.min(axis=0) * full_shape).astype(int)
    stop = np.ceil(frac.max(axis=0) * full_shape).astype(int) + 1
    start = np.maximum(start, 0)
    stop = np.minimum(stop, np.asarray(full_shape))
    if np.any(stop <= start):
        raise ValueError("local crop does not intersect the BvK grid")
    slices = tuple(slice(int(start[axis]), int(stop[axis])) for axis in range(3))
    return slices, start


def replicated_atoms_in_crop(
    structure,
    lattice: np.ndarray,
    mesh: tuple[int, int, int],
    supercell_lattice: np.ndarray,
    full_shape: tuple[int, int, int],
    slices: tuple[slice, slice, slice],
) -> tuple[list[int], np.ndarray]:
    translations = cell_translations(mesh)
    positions, numbers = [], []
    starts = np.array([part.start for part in slices])
    stops = np.array([part.stop for part in slices])
    inverse = np.linalg.inv(supercell_lattice)
    for translation in translations:
        shift = translation @ lattice
        for site in structure.sites:
            position = np.asarray(site.cartesian, dtype=float) + shift
            grid_position = (position @ inverse) * full_shape
            if np.all(grid_position >= starts) and np.all(grid_position < stops):
                positions.append(position)
                numbers.append(fields.base.ATOMIC_NUMBERS[site.element])
    return numbers, np.asarray(positions, dtype=float)


def normalized_expected(amplitudes: np.ndarray) -> np.ndarray:
    gram = (amplitudes.conj().T @ amplitudes).real
    scale = np.sqrt(np.diag(gram))
    return gram / np.outer(scale, scale)


def run(args) -> dict:
    density_dir = args.density_dir.resolve()
    pair_dir = args.pair_dir.resolve()
    calc = args.calc.resolve()
    output = args.output.resolve()
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output exists and is not a directory: {output}")
    if not np.isfinite(args.crop_padding) or args.crop_padding <= 0.0:
        raise ValueError("--crop-padding must be finite and positive")
    density_report = json.loads(
        (density_dir / "iao-density-validation.json").read_text(encoding="utf-8")
    )
    isym = int(density_report.get("time_reversal", {}).get("ISYM", 0))

    with (density_dir / "state_map.csv").open(newline="", encoding="utf-8") as handle:
        state_rows = list(csv.DictReader(handle))
    coefficients = np.load(density_dir / "iao_state_coefficients.npy", allow_pickle=False)
    kpoints = np.load(
        density_dir / "kpoint_fractional_coordinates.npy", allow_pickle=False
    )
    config = json.loads((pair_dir / "input/config.json").read_text(encoding="utf-8"))
    if "density_matrices_k" not in config:
        raise ValueError("BvK reconstruction requires the periodic density_matrices_k input form")
    report = json.loads(
        (pair_dir / "result/density-channel-validation.json").read_text(encoding="utf-8")
    )
    with np.load(pair_dir / "result/channel_coefficients.npz", allow_pickle=False) as data:
        u = data["atom_a_svd_coefficients"].copy()
        v = data["atom_b_svd_coefficients"].copy()
        singular = data["singular_values"].copy()
        columns_a = data["atom_a_columns"].copy()
        columns_b = data["atom_b_columns"].copy()
    channels = singular.size

    mesh = read_automatic_mesh(calc / "KPOINTS")
    mesh_validation = (
        validate_explicit_full_mesh(kpoints, state_rows, mesh)
        if isym == 1
        else validate_time_reversal_mesh(kpoints, state_rows, mesh)
    )
    translations = cell_translations(mesh)
    state_k = np.asarray([int(row["k_index"]) - 1 for row in state_rows])
    weights = np.asarray([float(row["state_weight_W"]) for row in state_rows])
    centre_cell = np.asarray([size // 2 for size in mesh], dtype=int)
    translation_a = np.asarray(config["atom_a"].get("lattice_translation", [0, 0, 0]), dtype=int)
    translation_b = np.asarray(config["atom_b"].get("lattice_translation", [0, 0, 0]), dtype=int)
    phase_a = np.exp(-2j * np.pi * (kpoints[state_k] @ (centre_cell + translation_a)))
    phase_b = np.exp(-2j * np.pi * (kpoints[state_k] @ (centre_cell + translation_b)))
    root_w = np.sqrt(weights)[:, None]
    amp_a = root_w * phase_a[:, None] * (coefficients[:, columns_a] @ u)
    amp_b = root_w * phase_b[:, None] * (coefficients[:, columns_b] @ v)
    amplitudes = np.concatenate((amp_a, amp_b), axis=1)
    if isym == 0:
        amplitudes *= math.sqrt(2.0)

    with Wavecar(calc / "WAVECAR") as wav:
        native = tuple(int(value) for value in wav.ng)
        wavecar_k = np.asarray([item[1] for item in wav.kinfo], dtype=float)
    if isym == 0 and not np.allclose(wavecar_k, kpoints, atol=1.0e-10, rtol=0.0):
        raise ValueError("WAVECAR k-point order disagrees with the saved density k points")
    shape = tuple(value * args.grid_multiplier for value in native)
    print(f"  reconstructing {2 * channels} partners in {math.prod(mesh)} BvK cells", flush=True)
    print(f"  primitive grid {shape}; assembled grid "
          f"{tuple(shape[i] * mesh[i] for i in range(3))}", flush=True)
    cell_fields, structure, volume, wavefunction_expansion = fields.reconstruct(
        calc, state_rows, amplitudes, shape, args.mode_batch,
        build_ae=not args.smooth_only,
        density_report=density_report,
        target_kpoints=kpoints,
        cell_translations=translations,
    )
    full_shape = tuple(int(shape[axis] * mesh[axis]) for axis in range(3))
    voxel = volume / math.prod(shape)
    lattice = np.asarray(structure.lattice, dtype=float)
    supercell_lattice = np.asarray(mesh)[:, None] * lattice
    atom_a = int(config["atom_indices_1based"]["atom_a"])
    atom_b = int(config["atom_indices_1based"]["atom_b"])
    target_positions = np.vstack((
        np.asarray(structure.sites[atom_a - 1].cartesian)
        + (centre_cell + translation_a) @ lattice,
        np.asarray(structure.sites[atom_b - 1].cartesian)
        + (centre_cell + translation_b) @ lattice,
    ))
    slices, start = crop_slices(
        target_positions, args.crop_padding, supercell_lattice, full_shape
    )
    crop_shape = tuple(part.stop - part.start for part in slices)
    crop, gram, norms, full_absolute_products = crop_and_integrate(
        cell_fields, channels, translations, slices, voxel
    )
    del cell_fields
    crop_a, crop_b = crop[:channels], crop[channels:]
    gram_aa, gram_bb, gram_ab = gram[:channels, :channels], gram[channels:, channels:], gram[:channels, channels:]
    expected_aa = normalized_expected(amp_a)
    expected_bb = normalized_expected(amp_b)
    scale_a = np.sqrt(np.diag((amp_a.conj().T @ amp_a).real))
    scale_b = np.sqrt(np.diag((amp_b.conj().T @ amp_b).real))
    expected_ab = (amp_a.conj().T @ amp_b).real / np.outer(scale_a, scale_b)
    numbers, positions = replicated_atoms_in_crop(
        structure, lattice, mesh, supercell_lattice, full_shape, slices
    )
    axes = (supercell_lattice / np.asarray(full_shape)[:, None]) / fields.BOHR
    origin = (start / np.asarray(full_shape)) @ supercell_lattice / fields.BOHR

    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".bvk-fields.", dir=output.parent))
    rows, total_crop = [], np.zeros(crop_shape)
    try:
        for index in range(channels):
            crop_proxy = 2.0 * singular[index] * crop_a[index] * crop_b[index]
            total_crop += crop_proxy
            partner_a_name = (
                f"channel_{index + 1:02d}_partner_{config['atom_a']['label']}_"
                "bvk_local_amplitude.cube"
            )
            partner_b_name = (
                f"channel_{index + 1:02d}_partner_{config['atom_b']['label']}_"
                "bvk_local_amplitude.cube"
            )
            product_name = f"channel_{index + 1:02d}_bvk_local_coherence_proxy.cube"
            entry = report["channels"][index]
            for name, values, side, label in (
                (partner_a_name, crop_a[index], "A", config["atom_a"]["label"]),
                (partner_b_name, crop_b[index], "B", config["atom_b"]["label"]),
            ):
                fields.write_cube(
                    staged / name, numbers, positions / fields.BOHR, origin, axes,
                    crop_shape, values,
                    (
                        f"{config['structure_label']} {config['pair_label']} BvK-local "
                        f"channel {index + 1} partner {side} ({label})",
                        "normalized signed partner amplitude in Angstrom^-3/2; not density",
                    ),
                )
            fields.write_cube(
                staged / product_name, numbers, positions / fields.BOHR, origin, axes,
                crop_shape, crop_proxy,
                (
                    f"{config['structure_label']} {config['pair_label']} BvK-local channel {index + 1}",
                    f"2*s*Re[A*B], s={singular[index]:.10g}; density-coherence proxy, not bond energy",
                ),
            )
            full_abs = float(2.0 * singular[index] * full_absolute_products[index])
            crop_abs = float(np.sum(np.abs(crop_proxy)) * voxel)
            rows.append({
                "channel": index + 1,
                "singular_value_s": float(singular[index]),
                "singular_value_squared": float(singular[index] ** 2),
                "fraction_of_pair_squared_coherence": entry["fraction_of_pair_squared_coherence"],
                "crop_atom_a_norm_fraction": float(np.sum(crop_a[index] ** 2) * voxel),
                "crop_atom_b_norm_fraction": float(np.sum(crop_b[index] ** 2) * voxel),
                "crop_absolute_proxy_fraction": crop_abs / full_abs if full_abs else 0.0,
                "suggested_partner_a_isovalue_Angstrom^-3/2": float(
                    np.quantile(np.abs(crop_a[index]), 0.9995)
                ),
                "suggested_partner_b_isovalue_Angstrom^-3/2": float(
                    np.quantile(np.abs(crop_b[index]), 0.9995)
                ),
                "suggested_visual_isovalue_e_per_A3": float(np.quantile(np.abs(crop_proxy), 0.9995)),
                "cube_partner_a_amplitude": partner_a_name,
                "cube_partner_b_amplitude": partner_b_name,
                "cube": product_name,
            })
        total_name = "channel_sum_bvk_local_coherence_proxy.cube"
        fields.write_cube(
            staged / total_name, numbers, positions / fields.BOHR, origin, axes,
            crop_shape, total_crop,
            (
                f"{config['structure_label']} {config['pair_label']} sum of all BvK-local PDC channels",
                "sum_i 2*s_i*Re[A_i*B_i]; density-coherence proxy, not bond energy",
            ),
        )
        with (staged / "bvk_local_channel_table.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        grid_checks = {
            "primitive_reconstruction_grid": list(shape),
            "bvk_full_grid": list(full_shape),
            "local_crop_grid": list(crop_shape),
            "voxel_angstrom3": voxel,
            "grid_spacing_angstrom": [
                float(np.linalg.norm(lattice[axis]) / shape[axis]) for axis in range(3)
            ],
            "atom_a_gram_max_abs_deviation": float(np.max(np.abs(gram_aa - expected_aa))),
            "atom_b_gram_max_abs_deviation": float(np.max(np.abs(gram_bb - expected_bb))),
            "cross_gram_max_abs_deviation": float(np.max(np.abs(gram_ab - expected_ab))),
            "partner_norms_before_rescale": norms.tolist(),
        }
        manifest = {
            "status": "pass",
            "scientific_status": "BvK-closed signed partners and density-weighted product proxy",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "structure": config["structure_label"],
            "pair": config["pair_label"],
            "scope": report["scope"],
            "bond_length_angstrom": config["bond_length_angstrom"],
            "observable": "rho_i(r) = 2*s_i*Re[A_i*(r) B_i(r)]",
            "stored_fields": {
                "partner_amplitudes": "A_i(r), B_i(r); normalized signed fields in Angstrom^-3/2",
                "product_proxy": "2*s_i*Re[A_i*(r) B_i(r)]; signed density-like proxy in e/Angstrom^3",
            },
            "bvk": mesh_validation,
            "target_bvk_cell": centre_cell.tolist(),
            "atom_a_lattice_translation": translation_a.tolist(),
            "atom_b_lattice_translation": translation_b.tolist(),
            "crop_padding_angstrom": args.crop_padding,
            "crop_origin_angstrom": (origin * fields.BOHR).tolist(),
            "crop_atom_count": len(numbers),
            "all_electron_reconstruction": not args.smooth_only,
            "channels": rows,
            "channel_sum_cube": total_name,
            "grid_checks": grid_checks,
            "wavefunction_space_group_expansion": wavefunction_expansion,
            "interpretation_boundary": (
                "All stored k points and their time reverses are closed on the BvK supercell. "
                "The written partner amplitudes show gauge-fixed orbital signs and are not densities. "
                "Their products are signed density-like proxies, not electron densities. The product "
                "integral is not Q_AB, a bond population, COHP, or bond energy; Q_AB remains sum_i s_i^2."
            ),
        }
        (staged / "bvk-local-field-validation.json").write_text(
            json_text(manifest), encoding="utf-8"
        )
        backup = None
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.previous.", dir=output.parent))
            backup.rmdir()
            os.replace(output, backup)
        try:
            os.replace(staged, output)
        except BaseException:
            if backup is not None and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        raise
    return manifest


def self_test() -> None:
    mesh = (4, 2, 1)
    kpoints = np.array([
        [0.125, 0.25, 0.0], [-0.125, 0.25, 0.0],
        [0.375, 0.25, 0.0], [-0.375, 0.25, 0.0],
    ])
    rows = [
        {"k_index": str(index + 1), "k_weight": "0.25"}
        for index in range(len(kpoints))
    ]
    check = validate_time_reversal_mesh(kpoints, rows, mesh)
    assert check["phase_orthogonality_max_abs_error"] < 1.0e-12
    translations = cell_translations(mesh)
    state_k = np.arange(len(kpoints))
    amplitudes = np.ones((len(kpoints), 1), dtype=complex)
    expanded = expand_amplitudes(amplitudes, state_k, kpoints, translations)
    assert expanded.shape == (len(kpoints), math.prod(mesh))
    full_kpoints = np.vstack((kpoints, -kpoints))
    full_rows = [
        {
            "k_index": str(index + 1),
            "k_weight": str(1.0 / math.prod(mesh)),
            "source_k_index": str(index % len(kpoints) + 1),
        }
        for index in range(math.prod(mesh))
    ]
    full_check = validate_explicit_full_mesh(full_kpoints, full_rows, mesh)
    assert full_check["phase_orthogonality_max_abs_error"] < 1.0e-12
    full_amplitudes = np.ones((math.prod(mesh), 1), dtype=complex)
    full_expanded = expand_full_mesh_amplitudes(
        full_amplitudes, np.arange(math.prod(mesh)), full_kpoints, translations
    )
    assert full_expanded.shape == (math.prod(mesh), math.prod(mesh))
    toy = np.arange(math.prod(mesh) * 2, dtype=float).reshape(math.prod(mesh), 2, 1, 1, 1)
    assembled = assemble_supercell(toy.reshape(-1, 1, 1, 1), 2, (1, 1, 1), mesh, translations)
    assert assembled.shape == (2, 4, 2, 1)

    shape, channels, voxel = (2, 3, 2), 4, 0.2
    cells = np.random.default_rng(8).normal(size=(len(translations) * 2 * channels, *shape))
    full = assemble_supercell(cells, 2 * channels, shape, mesh, translations)
    slices = (slice(1, 7), slice(1, 5), slice(0, 2))
    crop, gram, norms, absolute = crop_and_integrate(cells, channels, translations, slices, voxel)
    dense = full.reshape(2 * channels, -1)
    expected_norms = np.sqrt(np.sum(dense**2, axis=1) * voxel)
    dense /= expected_norms[:, None]
    assert np.allclose(norms, expected_norms, atol=1e-13, rtol=1e-13)
    assert np.allclose(gram, dense @ dense.T * voxel, atol=1e-13, rtol=1e-13)
    assert np.allclose(crop, full[(slice(None), *slices)], atol=1e-13, rtol=1e-13)
    assert np.allclose(absolute, np.sum(np.abs(dense[:channels] * dense[channels:]), axis=1) * voxel)
    print("BvK local channel field self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-dir", type=Path)
    parser.add_argument("--pair-dir", type=Path)
    parser.add_argument("--calc", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--grid-multiplier", type=int, default=2)
    parser.add_argument("--mode-batch", type=int, default=12)
    parser.add_argument("--crop-padding", type=float, default=4.0)
    parser.add_argument("--smooth-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for required in ("density_dir", "pair_dir", "calc", "output"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required")
    if args.grid_multiplier < 1 or args.mode_batch < 1:
        parser.error("--grid-multiplier and --mode-batch must be positive")
    manifest = run(args)
    print(json_text({
        "status": manifest["status"],
        "bvk_mesh": manifest["bvk"]["mesh"],
        "channels": len(manifest["channels"]),
        "grid_checks": manifest["grid_checks"],
        "output": str(args.output.resolve()),
    }), end="")


if __name__ == "__main__":
    main()
