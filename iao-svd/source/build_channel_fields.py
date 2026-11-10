#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "scipy", "ase", "spglib", "PyVaspWfc @ git+https://github.com/QijingZheng/VaspBandUnfolding@3f8a6285b538cdec1d896e803e3d0800147fda88"]
# ///
"""Reconstruct PDC channel partners and coherence densities in real space.

What this builds
----------------
For channel i the two partners are combinations of IAO columns,

    a_i = sum_{mu in A} u_(mu,i) |IAO_mu>,   b_i = sum_{nu in B} v_(nu,i) |IAO_nu>

Only the part of a partner that lies inside the occupied manifold carries any
density, and that part is available directly from the stored state coefficients
<psi_s|IAO_mu>. So each partner is reconstructed as an occupied-space field

    A_i(r) = sum_s sqrt(W_s) <psi_s|a_i> psi_s(r),    W_s = 2 w_k f_s

with the PAW all-electron augmentation added, then normalised to unit L2 on the
grid. The channel field written for figures is

    proxy_i(r) = 2 s_i Ahat_i(r) Bhat_i(r)

The product suppresses a partner's orthogonalisation tail where the other partner
is near zero, so the bond region survives without masking, cutoff or an assumed
bond axis. The 2*s_i prefactor labels the proxy by the exact coefficient-space
weight; it does not make absolute field magnitudes comparable.

Scope limit, stated deliberately
--------------------------------
proxy_i is a shape carrier weighted by an exact coefficient-space intensity. It is NOT a closure-exact
partition of the pair cross density: the exact cross density is an incoherent sum
over states of per-state atomic projections and would require the IAOs themselves
as real-space functions. Every quantitative statement (s_i, s_i^2, shares,
populations) comes from the raw coefficient-space table. Canonical correlations are
a separate pair-level spectrum. Multi-k Bloch fields are coherently accumulated in
one primitive-cell grid without a Born-von-Karman supercell convention, so these
cubes are qualitative proxies only: do not integrate them or compare absolute field
strengths across structures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'support'))

from ase.io import read as ase_read  # noqa: E402
from paw import nonlq, pawpotcar  # noqa: E402
import paw_aware_region_modes as paw_core  # noqa: E402
import regional_mode_utils as base  # noqa: E402
from vasp_readers import Wavecar, parse_poscar  # noqa: E402
from space_group_expansion import (  # noqa: E402
    reorder_plane_waves,
    transform_plane_waves,
)

BOHR = 0.529177210903




def write_cube(path: Path, numbers, positions_bohr, origin, axes, shape, values, comment):
    lines = [comment[0], comment[1],
             f"{len(numbers):5d}{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}"]
    for index in range(3):
        lines.append(f"{shape[index]:5d}{axes[index][0]:12.6f}{axes[index][1]:12.6f}{axes[index][2]:12.6f}")
    for number, position in zip(numbers, positions_bohr):
        lines.append(f"{int(number):5d}{float(number):12.6f}"
                     f"{position[0]:12.6f}{position[1]:12.6f}{position[2]:12.6f}")
    flat = np.asarray(values).reshape(-1)
    body = ["".join(f"{value:13.5E}" for value in flat[start : start + 6])
            for start in range(0, flat.size, 6)]
    path.write_text("\n".join(lines + body) + "\n", encoding="ascii")


def paw_augmentation_patches(atoms, datasets, element_idx, shape, modes):
    """Sample AE-PS partial waves separately for each periodic atom image."""
    lattice = np.asarray(atoms.cell, dtype=float)
    inverse = np.linalg.inv(lattice)
    spline_cache = {}
    patches = []
    for center, type_index in zip(atoms.get_scaled_positions(), element_idx):
        pp = datasets[type_index]
        points, cart, radii = paw_core.periodic_sphere_grid_points(
            shape, center, lattice, float(pp.proj_rmax), inverse
        )
        directions = np.zeros_like(cart)
        nonzero = radii > 1e-14
        directions[nonzero] = cart[nonzero] / radii[nonzero, None]
        directions[~nonzero, 2] = 1.0
        basis = np.asarray([
            paw_core.radial_difference_values(pp, int(n), int(l), radii, spline_cache)
            * paw_core.sph_r(directions, int(l), int(m))
            for n, l, m in pp.ilm
        ])
        fractional = np.column_stack(np.unravel_index(points, shape)) / np.asarray(shape)
        images = np.rint(fractional - center - cart @ inverse).astype(int)
        atom_patches = []
        for image in np.unique(images, axis=0):
            selected = np.all(images == image, axis=1)
            atom_patches.append((
                points[selected], basis[:, selected], center + image,
                np.zeros((modes, pp.lmmax)),
            ))
        patches.append(atom_patches)
    return patches


def reconstruct(
    calc: Path,
    state_rows,
    amplitudes,
    shape,
    batch,
    build_ae=True,
    *,
    density_report=None,
    target_kpoints=None,
    cell_translations=None,
):
    """Build AE real-space fields for column-wise state amplitudes.

    amplitudes[s, m] is the coefficient of state s in field m, already carrying
    sqrt(W_s). No further k weighting is applied here, which is what keeps the
    normalisation of the different fields mutually consistent.
    """
    structure = parse_poscar(calc / "CONTCAR")
    atoms = ase_read(calc / "CONTCAR")
    datasets = [pawpotcar(block) for block in
                (calc / "POTCAR").read_text(errors="replace").split("End of Dataset")[:-1]]
    volume = abs(float(np.linalg.det(np.asarray(structure.lattice, dtype=np.float64))))
    frac_grid = base.grid_fractional(shape)
    modes = amplitudes.shape[1]
    translations = np.asarray([[0, 0, 0]] if cell_translations is None else cell_translations)
    cell_count = len(translations)
    output_modes = cell_count * modes

    row_by_state = {int(row["state_index"]): row for row in state_rows}
    if set(row_by_state) != set(range(len(state_rows))):
        raise ValueError("state_map.csv state_index values must be contiguous and zero based")
    by_k: dict[int, list[int]] = {}
    for row in state_rows:
        by_k.setdefault(int(row["k_index"]) - 1, []).append(int(row["state_index"]))

    density_report = density_report or {}
    isym = int(density_report.get("time_reversal", {}).get("ISYM", 0))
    operations = {
        int(operation["index_1based"]): operation
        for operation in (density_report.get("space_group_expansion") or {}).get("operations", [])
    }
    if isym == 1:
        target_kpoints = np.asarray(target_kpoints, dtype=np.float64)
        if target_kpoints.shape != (len(by_k), 3):
            raise ValueError(
                "ISYM=1 field reconstruction requires one saved target k point for every "
                f"expanded block; got {target_kpoints.shape}, expected {(len(by_k), 3)}"
            )
        if not operations:
            raise ValueError(
                "ISYM=1 field reconstruction requires space_group_expansion.operations "
                "in iao-density-validation.json"
            )
    elif isym != 0:
        raise ValueError(f"unsupported ISYM={isym}; only spinless ISYM=0 and ISYM=1 are supported")

    smooth = np.zeros((cell_count, modes, *shape), dtype=np.float64)
    element_idx = [[pp.element.split("_")[0] for pp in datasets].index(symbol)
                   for symbol in atoms.get_chemical_symbols()]
    patches = paw_augmentation_patches(atoms, datasets, element_idx, shape, output_modes) if build_ae else []

    transform_diagnostics = {
        "ISYM": isym,
        "expanded_k_block_count": len(by_k),
        "symmetry_generated_k_block_count": 0,
        "gvector_integer_max_abs_error": 0.0,
        "target_kpoint_max_abs_error": 0.0,
        "coefficient_norm_relative_error": 0.0,
        "missing_gvector_count": 0,
        "extra_gvector_count": 0,
        "augmentation_image_patch_count": sum(len(atom) for atom in patches),
    }

    with Wavecar(calc / "WAVECAR") as wav:
        for k_index in sorted(by_k):
            indices = np.asarray(by_k[k_index], dtype=int)
            rows = [row_by_state[int(index)] for index in indices]
            bands = np.asarray([int(row["band_index"]) - 1 for row in rows], dtype=int)
            if isym == 1:
                source_indices = {int(row["source_k_index"]) - 1 for row in rows}
                operation_indices = {
                    int(row["symmetry_operation_index_1based"]) for row in rows
                }
                time_reversal_flags = {bool(int(row["time_reversed"])) for row in rows}
                if len(source_indices) != 1 or len(operation_indices) != 1 or len(time_reversal_flags) != 1:
                    raise ValueError(
                        f"expanded k block {k_index + 1} mixes source k points or symmetry images"
                    )
                source_k_index = source_indices.pop()
                operation_index = operation_indices.pop()
                time_reversed = time_reversal_flags.pop()
                if operation_index not in operations:
                    raise ValueError(f"missing reported symmetry operation {operation_index}")
            else:
                source_k_index = k_index
                operation_index = 0
                time_reversed = False

            source_gvec = wav.gvecs(source_k_index)
            combined = np.zeros((modes, len(source_gvec)), dtype=np.complex128)
            for start in range(0, len(bands), 48):
                stop = min(start + 48, len(bands))
                coeff = paw_core.read_raw_coefficients(
                    wav, source_k_index, bands[start:stop]
                )
                # Spatial symmetry is linear, but spinless time reversal is
                # anti-linear.  The ISYM=1 state table already stores the
                # conjugated TARGET amplitudes.  transform_plane_waves() will
                # conjugate the source wavefunction coefficients, so pre-
                # conjugate the mixing amplitudes here to prevent applying
                # time reversal to them a second time:
                #   conj(conj(A_target) @ psi_source)
                #       = A_target @ conj(psi_source).
                mixing = amplitudes[indices[start:stop]]
                if isym == 1 and time_reversed:
                    mixing = mixing.conj()
                combined += mixing.T @ coeff

            if isym == 1:
                operation = operations[operation_index]
                kvec, mapped_gvec, combined, diagnostics = transform_plane_waves(
                    source_gvec,
                    combined,
                    np.asarray(wav.kinfo[source_k_index][1], dtype=np.float64),
                    operation["fractional_rotation"],
                    operation["fractional_translation"],
                    time_reversed=time_reversed,
                    expected_kpoint=target_kpoints[k_index],
                )
                qengine = nonlq(atoms, wav.encut, datasets, k=kvec)
                combined, mapping = reorder_plane_waves(
                    mapped_gvec, combined, qengine.Gvec
                )
                gvec = np.asarray(qengine.Gvec, dtype=np.int64)
                transform_diagnostics["symmetry_generated_k_block_count"] += 1
                for key in (
                    "gvector_integer_max_abs_error",
                    "target_kpoint_max_abs_error",
                    "coefficient_norm_relative_error",
                ):
                    transform_diagnostics[key] = max(
                        transform_diagnostics[key], float(diagnostics[key])
                    )
                for key in ("missing_gvector_count", "extra_gvector_count"):
                    transform_diagnostics[key] = max(
                        transform_diagnostics[key], int(mapping[key])
                    )
            else:
                kvec = np.asarray(wav.kinfo[k_index][1], dtype=np.float64)
                gvec = source_gvec
                qengine = nonlq(atoms, wav.encut, datasets, k=kvec)
                if not np.array_equal(gvec, np.asarray(qengine.Gvec, dtype=np.int64)):
                    raise RuntimeError(
                        f"WAVECAR and PAW projector G-vector order differ at k={k_index + 1}"
                    )
            paw_core.validate_g_mapping(gvec, shape, f"channel field k={k_index + 1}")
            if build_ae:
                for atom_index in range(len(atoms)):
                    projector = paw_core.atom_projector_matrix(qengine, atom_index)
                    beta = combined @ projector.T
                    # nonlq returns VASP's cell-periodic beta, without exp(i k.tau).
                    for _, _, image_center, accumulated in patches[atom_index]:
                        phases = np.exp(2j * np.pi * ((translations + image_center) @ kvec))
                        accumulated.reshape(cell_count, modes, -1)[:] += (
                            phases[:, None, None] * beta[None]
                        ).real
            del qengine

            bloch = np.exp(2j * np.pi * np.einsum("...i,i->...", frac_grid, kvec))
            translation_phases = np.exp(2j * np.pi * (translations @ kvec))
            for start in range(0, modes, batch):
                stop = min(start + batch, modes)
                reciprocal = np.zeros((stop - start, *shape), dtype=np.complex128)
                reciprocal[:, gvec[:, 0] % shape[0], gvec[:, 1] % shape[1],
                           gvec[:, 2] % shape[2]] = combined[start:stop]
                periodic = np.fft.ifftn(reciprocal, axes=(1, 2, 3)) * np.prod(shape)
                periodic *= bloch[None] / math.sqrt(volume)
                for cell_index, phase in enumerate(translation_phases):
                    smooth[cell_index, start:stop] += np.real(periodic * phase)
                del reciprocal, periodic
            source_note = (
                f" <- IBZ {source_k_index + 1}, op {operation_index}, TR={int(time_reversed)}"
                if isym == 1 else ""
            )
            print(f"    k {k_index + 1:2d}{source_note}: fields accumulated", flush=True)

    flat = smooth.reshape(output_modes, -1)
    for atom_patches in patches:
        for points, basis, _, accumulated in atom_patches:
            flat[:, points] += accumulated @ basis
    return smooth.reshape(output_modes, *shape), structure, volume, transform_diagnostics


def run(args) -> dict:
    density_dir = args.density_dir.resolve()
    pair_dir = args.pair_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    density_report = json.loads(
        (density_dir / "iao-density-validation.json").read_text(encoding="utf-8")
    )
    with (density_dir / "state_map.csv").open(newline="", encoding="utf-8") as handle:
        state_rows = list(csv.DictReader(handle))
    k_point_count = len({int(row["k_index"]) for row in state_rows})
    coefficients = np.load(density_dir / "iao_state_coefficients.npy")
    config = json.loads((pair_dir / "input" / "config.json").read_text(encoding="utf-8"))
    report = json.loads((pair_dir / "result" / "density-channel-validation.json").read_text(encoding="utf-8"))
    data = np.load(pair_dir / "result" / "channel_coefficients.npz")

    # map the reordered pair columns back to original IAO column indices
    selected = json.loads((pair_dir / "input" / "config.json").read_text(encoding="utf-8"))
    atom_a = int(selected["atom_indices_1based"]["atom_a"])
    atom_b = int(selected["atom_indices_1based"]["atom_b"])
    label_map: dict[int, list[int]] = {}
    with (density_dir / "iao_labels.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label_map.setdefault(int(row["atom_index_1based"]), []).append(int(row["reference_column"]))
    columns_a = np.asarray(sorted(label_map[atom_a]))
    columns_b = np.asarray(sorted(label_map[atom_b]))

    u = data["atom_a_svd_coefficients"]
    v = data["atom_b_svd_coefficients"]
    singular = data["singular_values"]
    channels = singular.size

    weights = np.asarray([float(row["state_weight_W"]) for row in state_rows])
    root_w = np.sqrt(weights)[:, None]
    translation_a = np.asarray(config["atom_a"].get("lattice_translation", [0, 0, 0]))
    translation_b = np.asarray(config["atom_b"].get("lattice_translation", [0, 0, 0]))
    pair_translation = translation_b - translation_a
    if np.any(pair_translation):
        kpoints_path = density_dir / "kpoint_fractional_coordinates.npy"
        if not kpoints_path.is_file():
            raise FileNotFoundError(
                f"cross-cell channel fields require {kpoints_path}; rerun build_iao_density.py"
            )
        kpoints = np.load(kpoints_path)
        state_k = np.asarray([int(row["k_index"]) - 1 for row in state_rows])
        phase_b = np.exp(-2j * np.pi * (kpoints[state_k] @ pair_translation))[:, None]
    else:
        phase_b = 1.0
    # <psi_s|a_i> = sum_mu u_(mu,i) <psi_s|IAO_mu>
    amp_a = root_w * (coefficients[:, columns_a] @ u)
    amp_b = root_w * phase_b * (coefficients[:, columns_b] @ v)
    amplitudes = np.concatenate((amp_a, amp_b), axis=1)

    with Wavecar(args.calc.resolve() / "WAVECAR") as wav:
        native = tuple(int(value) for value in wav.ng)
    shape = tuple(int(value) * args.grid_multiplier for value in native)
    print(f"  native WAVECAR grid {native} -> reconstruction grid {shape} "
          f"({args.grid_multiplier}x)", flush=True)

    target_kpoints_path = density_dir / "kpoint_fractional_coordinates.npy"
    target_kpoints = (
        np.load(target_kpoints_path, allow_pickle=False)
        if target_kpoints_path.is_file() else None
    )
    fields, structure, volume, wavefunction_expansion = reconstruct(
        args.calc.resolve(), state_rows, amplitudes, shape, args.mode_batch,
        build_ae=not args.smooth_only,
        density_report=density_report,
        target_kpoints=target_kpoints,
    )
    field_a, field_b = fields[:channels], fields[channels:]

    voxel = volume / float(np.prod(shape))
    flat_a = field_a.reshape(channels, -1)
    flat_b = field_b.reshape(channels, -1)
    norm_a = np.sqrt(np.sum(flat_a**2, axis=1) * voxel)
    norm_b = np.sqrt(np.sum(flat_b**2, axis=1) * voxel)
    unit_a = flat_a / norm_a[:, None]
    unit_b = flat_b / norm_b[:, None]

    gram_aa = unit_a @ unit_a.T * voxel
    gram_bb = unit_b @ unit_b.T * voxel
    gram_ab = unit_a @ unit_b.T * voxel

    # analytic expectation in the same density-weighted metric the fields carry
    expected_aa = (amp_a.conj().T @ amp_a).real
    expected_bb = (amp_b.conj().T @ amp_b).real
    expected_ab = (amp_a.conj().T @ amp_b).real
    scale_a = np.sqrt(np.diag(expected_aa))
    scale_b = np.sqrt(np.diag(expected_bb))
    expected_aa = expected_aa / np.outer(scale_a, scale_a)
    expected_bb = expected_bb / np.outer(scale_b, scale_b)
    expected_ab = expected_ab / np.outer(scale_a, scale_b)

    grid_checks = {
        "reconstruction_grid": list(shape),
        "native_grid": list(native),
        "grid_multiplier": args.grid_multiplier,
        "voxel_angstrom3": voxel,
        "grid_spacing_angstrom": [
            float(np.linalg.norm(np.asarray(structure.lattice)[i]) / shape[i]) for i in range(3)
        ],
        "atom_a_gram_max_abs_deviation": float(np.max(np.abs(gram_aa - expected_aa))),
        "atom_b_gram_max_abs_deviation": float(np.max(np.abs(gram_bb - expected_bb))),
        "cross_gram_max_abs_deviation": float(np.max(np.abs(gram_ab - expected_ab))),
        "atom_a_field_norms_before_rescale": norm_a.tolist(),
        "atom_b_field_norms_before_rescale": norm_b.tolist(),
        "note": (
            "Deviations compare the grid-measured overlaps against the same overlaps computed "
            "analytically from the state amplitudes. For multi-k coherent primitive-cell proxy "
            "fields they are representation diagnostics, not a guaranteed grid-convergence series."
        ),
    }

    positions = np.asarray([site.cartesian for site in structure.sites]) / BOHR
    numbers = [base.ATOMIC_NUMBERS[site.element] for site in structure.sites]
    axes = (np.asarray(structure.lattice) / np.asarray(shape)[:, None]) / BOHR
    origin = np.zeros(3)

    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".fields.", dir=output.parent))
    rows = []
    total = np.zeros(int(np.prod(shape)))
    for index in range(channels):
        proxy = 2.0 * singular[index] * unit_a[index] * unit_b[index]
        total += proxy
        entry = report["channels"][index]
        partner_a_name = f"channel_{index+1:02d}_partner_{config['atom_a']['label']}_proxy.cube"
        partner_b_name = f"channel_{index+1:02d}_partner_{config['atom_b']['label']}_proxy.cube"
        product_name = f"channel_{index+1:02d}_weighted_partner_product_proxy.cube"
        for name, values in (
            (partner_a_name, unit_a[index]),
            (partner_b_name, unit_b[index]),
            (product_name, proxy),
        ):
            write_cube(
                staged / name, numbers, positions, origin, axes, shape, values,
                (f"{config['structure_label']} {config['pair_label']} channel {index+1}: {name}",
                 f"s={singular[index]:.10g}  pair share={entry['fraction_of_pair_squared_coherence']:.6%}"
                  "  qualitative proxy; partners 1/Angstrom^1.5, product e/Angstrom^3"),
            )
        rows.append({
            "channel": index + 1,
            "singular_value_s": float(singular[index]),
            "coherence_density_weight_2s": float(2.0 * singular[index]),
            "fraction_of_pair_squared_coherence": entry["fraction_of_pair_squared_coherence"],
            "fraction_of_selected_atom_a_star_squared_coherence": entry.get(
                "fraction_of_selected_atom_a_star_squared_coherence",
                entry.get("fraction_of_context_squared_coherence"),
            ),
            "fraction_of_selected_atom_b_star_squared_coherence": entry.get(
                "fraction_of_selected_atom_b_star_squared_coherence"
            ),
            "n_atom_a": entry["n_atom_a"],
            "n_atom_b": entry["n_atom_b"],
            "max_abs_weighted_partner_product_proxy_e_per_A3": float(np.max(np.abs(proxy))),
            "cube_partner_a_proxy": partner_a_name,
            "cube_partner_b_proxy": partner_b_name,
            "cube_weighted_partner_product_proxy": product_name,
        })
    write_cube(staged / "channel_sum_weighted_partner_product_proxy.cube", numbers, positions,
               origin, axes, shape, total,
               ("sum over channels of 2 s_i Ahat_i Bhat_i; qualitative proxy",
                "units e/Angstrom^3; do not integrate or compare absolute magnitudes"))

    with (staged / "channel_field_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (staged / "channel-field-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(staged, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-dir", type=Path, required=True)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--calc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-multiplier", type=int, default=2)
    parser.add_argument("--mode-batch", type=int, default=2)
    parser.add_argument("--smooth-only", action="store_true",
                        help="plot the smooth pseudo-field instead of the AE reconstruction")
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps(manifest["grid_checks"], indent=2, allow_nan=False))
    print(f"channel fields -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
