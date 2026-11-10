#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""Assemble a standalone HTML viewer for PDC channel fields across structures.

Design notes
------------
The v1/v2 q214 viewers embedded 25 pre-computed marching-cubes surfaces per cube.
That does not scale here: this instrument carries seven bonds x three channels x
three field types. Instead each field is cropped to a box around its own bond,
resampled onto a small uniform grid, quantised to one byte, and the isosurface is
extracted in the browser so the isovalue slider stays continuous.

three.js and OrbitControls are lifted verbatim from the accepted v2 viewer, so the
page stays offline-capable and no CDN is contacted. The surface extractor is
marching *tetrahedra* rather than marching cubes: six tetrahedra per voxel with a
sixteen-case table, which is far easier to get right than a 256-entry table and
cannot produce the ambiguous-face holes that a naive marching-cubes table does.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from prepare_pair_configs import read_poscar

BOHR = 0.529177210903
ELEMENT_OF_Z = {44: "Ru", 8: "O", 1: "H"}
# Explicit element-pair display windows. VESTA uses explicit bond
# specifications, not one covalent-radius multiplier for every pair.
VESTA_BOND_RANGES = {
    frozenset(("O", "H")): ((0.0, 1.2), (1.2, 2.1)),
    frozenset(("O", "O")): ((0.0, 1.7),),
    frozenset(("Ru", "O")): ((0.0, 2.22646),),
    frozenset(("Ru", "H")): ((0.0, 1.65998),),
}


def read_cube(path: Path) -> dict:
    tokens = path.read_text().splitlines()
    header = tokens[2].split()
    count = int(header[0])
    origin = np.array([float(v) for v in header[1:4]])
    shape, axes = [], []
    for row in range(3, 6):
        parts = tokens[row].split()
        shape.append(int(parts[0]))
        axes.append([float(v) for v in parts[1:4]])
    numbers, positions = [], []
    for row in range(6, 6 + abs(count)):
        parts = tokens[row].split()
        numbers.append(int(parts[0]))
        positions.append([float(v) for v in parts[2:5]])
    values = np.fromstring(" ".join(tokens[6 + abs(count):]), sep=" ")
    shape = tuple(shape)
    return {
        "origin": origin, "shape": shape, "axes": np.array(axes),
        "numbers": np.array(numbers), "positions": np.array(positions),
        "values": values.reshape(shape),
    }


def sample_box(volume: np.ndarray, cell: np.ndarray, centre: np.ndarray,
               half: float, samples: int, origin: np.ndarray | None = None,
               periodic: bool = True) -> np.ndarray:
    """Trilinear resample of a volume onto a cartesian box."""
    axis = np.linspace(-half, half, samples)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    points = centre[None, :] + np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    origin = np.zeros(3) if origin is None else np.asarray(origin)
    fractional = (points - origin) @ np.linalg.inv(cell)
    shape = np.asarray(volume.shape)
    grid = fractional * shape[None, :]
    base = np.floor(grid).astype(np.int64)
    if not periodic:
        if np.any(grid < 0.0) or np.any(grid > shape[None, :] - 1.0):
            raise ValueError("requested display box extends beyond the local CUBE")
        base = np.minimum(base, shape - 2)
    frac = grid - base
    result = np.zeros(len(points))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = ((1 - frac[:, 0]) if dx == 0 else frac[:, 0])
                weight = weight * ((1 - frac[:, 1]) if dy == 0 else frac[:, 1])
                weight = weight * ((1 - frac[:, 2]) if dz == 0 else frac[:, 2])
                index = base + np.array([dx, dy, dz])
                if periodic:
                    index %= shape
                result += weight * volume[index[:, 0], index[:, 1], index[:, 2]]
    return result.reshape(samples, samples, samples)


def quantise(values: np.ndarray) -> dict:
    scale = float(np.max(np.abs(values)))
    if scale <= 0.0:
        scale = 1.0
    bytes_ = np.clip(np.rint(values / scale * 127.0), -127, 127).astype(np.int8)
    return {"scale": scale, "data": base64.b64encode(bytes_.tobytes()).decode("ascii")}


def translated_atoms(symbols: list[str], positions: np.ndarray,
                     origin: np.ndarray) -> list[dict]:
    """Translate the intact CONTCAR cell; never wrap atoms independently."""
    return [
        {"el": symbol, "xyz": [float(v) for v in position - origin]}
        for symbol, position in zip(symbols, positions)
    ]


def translated_cell_edges(cell: np.ndarray, origin: np.ndarray) -> list[float]:
    """Translate the original CONTCAR cell by the same vector as its atoms."""
    corners = []
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                corners.append(np.array([i, j, k]) @ cell - origin)
    edges, flat = [], []
    for a in range(8):
        for b in range(a + 1, 8):
            differing = bin(a ^ b).count("1")
            if differing == 1:
                edges.append((a, b))
    for a, b in edges:
        flat.extend(float(v) for v in corners[a])
        flat.extend(float(v) for v in corners[b])
    return flat


def build_pair(field_dir: Path, pair_dir: Path, half: float, samples: int,
               structures_root: Path) -> dict:
    bvk_path = field_dir / "bvk-local-field-validation.json"
    is_bvk = bvk_path.is_file()
    manifest_path = bvk_path if is_bvk else field_dir / "channel-field-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads((pair_dir / "input" / "config.json").read_text(encoding="utf-8"))
    report = json.loads((pair_dir / "result" / "density-channel-validation.json").read_text(encoding="utf-8"))

    first_entry = manifest["channels"][0]
    first = read_cube(
        field_dir
        / (first_entry["cube"] if is_bvk else first_entry.get(
            "cube_weighted_partner_product_proxy", first_entry.get("cube_coherence_density")))
    )
    cube_cell = (first["axes"] * np.asarray(first["shape"])[:, None]) * BOHR
    structure_path = structures_root / manifest["structure"] / "CONTCAR"
    cell, positions, symbols = read_poscar(structure_path)
    cube_positions = first["positions"] * BOHR
    atom_a = int(config["atom_indices_1based"]["atom_a"]) - 1
    atom_b = int(config["atom_indices_1based"]["atom_b"]) - 1
    if is_bvk:
        centre_cell = np.asarray(manifest["target_bvk_cell"])
        pos_a = positions[atom_a] + (
            centre_cell + np.asarray(manifest["atom_a_lattice_translation"])
        ) @ cell
        pos_b = positions[atom_b] + (
            centre_cell + np.asarray(manifest["atom_b_lattice_translation"])
        ) @ cell
        centre = 0.5 * (pos_a + pos_b)
        target_index = [
            int(np.argmin(np.linalg.norm(cube_positions - position, axis=1)))
            for position in (pos_a, pos_b)
        ]
        if target_index[0] == target_index[1] or any(
            np.linalg.norm(cube_positions[index] - position) > 1e-4
            for index, position in zip(target_index, (pos_a, pos_b))
        ):
            raise ValueError(f"{field_dir.name}: target atoms are missing from local CUBE")
    else:
        if not np.allclose(cube_cell, cell, atol=1e-4) or not np.allclose(
            cube_positions, positions, atol=1e-5
        ):
            raise ValueError(f"{field_dir.name}: CUBE structure differs from source CONTCAR")
        pos_a = positions[atom_a]
        pos_b = positions[atom_b]
        delta = pos_b - pos_a
        fractional = delta @ np.linalg.inv(cell)
        fractional -= np.round(fractional)
        centre = pos_a + 0.5 * (fractional @ cell)
        target_index = [atom_a, atom_b]

    legacy_products = not is_bvk and any(
        "cube_weighted_partner_product_proxy" not in entry
        for entry in manifest["channels"]
    )
    volumes: dict[str, dict] = {}
    channels = []
    for entry in manifest["channels"]:
        index = entry["channel"]
        sampled_fields: dict[str, np.ndarray] = {}
        if is_bvk:
            files = [("rho", entry["cube"])]
            files.extend(
                (key, entry[name]) for key, name in (
                    ("pa", "cube_partner_a_amplitude"),
                    ("pb", "cube_partner_b_amplitude"),
                ) if name in entry
            )
        else:
            files = (
                ("rho", entry.get("cube_weighted_partner_product_proxy", entry.get("cube_coherence_density"))),
                ("pa", entry.get("cube_partner_a_proxy", entry.get("cube_partner_a"))),
                ("pb", entry.get("cube_partner_b_proxy", entry.get("cube_partner_b"))),
            )
        for key, filename in files:
            cube = read_cube(field_dir / filename)
            sampled = sample_box(
                cube["values"], cube_cell, centre, half, samples,
                cube["origin"] * BOHR if is_bvk else None, not is_bvk,
            )
            if (not is_bvk and key == "rho"
                    and "cube_weighted_partner_product_proxy" not in entry):
                # Legacy products were multiplied by BOHR**-3 after the fields had
                # already been normalised in Angstrom units. Undo that exact factor.
                sampled *= BOHR**3
            sampled_fields[key] = sampled
            volumes[f"{key}{index}"] = quantise(sampled)
        # The BvK partner fields are normalized orbital amplitudes on the same
        # real-space grid.  Their normalized in-phase/out-of-phase combinations
        # are therefore the chemically readable bonding/antibonding channel
        # orbitals.  Do not form these from the older partner *proxies*.
        if is_bvk and {"pa", "pb"}.issubset(sampled_fields):
            root_two = math.sqrt(2.0)
            volumes[f"bond{index}"] = quantise(
                (sampled_fields["pa"] + sampled_fields["pb"]) / root_two
            )
            volumes[f"anti{index}"] = quantise(
                (sampled_fields["pa"] - sampled_fields["pb"]) / root_two
            )
        source = report["channels"][index - 1]
        gap = source.get("relative_spectral_gap")
        gap = round(gap, 5) if gap is not None and np.isfinite(gap) else None
        channels.append({
            "i": index,
            "s": round(source["singular_value_s"], 6),
            "twoS": round(source["coherence_density_weight_2s"], 6),
            "s2": round(source["singular_value_squared"], 6),
            "share": round(source["fraction_of_pair_squared_coherence"], 6),
            "ctxShareA": round(
                source.get(
                    "fraction_of_selected_atom_a_star_squared_coherence",
                    source.get("fraction_of_context_squared_coherence"),
                ), 6,
            ),
            "ctxShareB": (
                round(source["fraction_of_selected_atom_b_star_squared_coherence"], 6)
                if "fraction_of_selected_atom_b_star_squared_coherence" in source else None
            ),
            "nA": round(source["n_atom_a"], 4),
            "nB": round(source["n_atom_b"], 4),
            "nPlus": round(source["n_plus"], 4),
            "nMinus": round(source["n_minus"], 4),
            "coherence": round(source["normalized_coherence_c"], 5),
            "gap": gap,
            "degenerate": any(d["channel"] == index for d in report["near_degenerate_channels"]),
            "gaugeMargin": round(source["gauge_margin"], 5),
            "gaugeFragile": source["gauge_margin"] < report.get("gauge", {}).get(
                "warning_threshold", 0.05
            ),
            "leakA": round(source["environment_leak_atom_a"], 5),
            "leakB": round(source["environment_leak_atom_b"], 5),
        })

    if is_bvk:
        total = read_cube(field_dir / manifest["channel_sum_cube"])
        volumes["rhoSum"] = quantise(sample_box(
            total["values"], cube_cell, centre, half, samples,
            total["origin"] * BOHR, False,
        ))

    # One display coordinate system: the core Ru is the model origin.  The
    # sampled field remains centred on the bond midpoint and gets one offset.
    if is_bvk:
        local_symbols = [ELEMENT_OF_Z.get(int(number), "X") for number in first["numbers"]]
        atoms = translated_atoms(local_symbols, cube_positions, pos_a)
        cell_edges = translated_cell_edges(cube_cell, pos_a - first["origin"] * BOHR)
    else:
        atoms = translated_atoms(symbols, positions, pos_a)
        cell_edges = translated_cell_edges(cell, pos_a)
    coordinates = np.array([a["xyz"] for a in atoms])
    bonds = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            ranges = VESTA_BOND_RANGES.get(
                frozenset((atoms[i]["el"], atoms[j]["el"])), ()
            )
            distance = np.linalg.norm(coordinates[j] - coordinates[i])
            if any(minimum <= distance <= maximum for minimum, maximum in ranges):
                bonds.append([i, j])

    canonical_analysis = report.get("canonical_analysis", {})
    canonical = canonical_analysis.get("canonical_correlations")
    if canonical is None:
        canonical = [entry["canonical_correlation_kappa"] for entry in report["channels"]]
    selected_a_star = report.get(
        "selected_atom_a_star_squared_coherence_total",
        report.get("context_squared_coherence_total"),
    )
    return {
        "id": f"{config['structure_label']}__{config['pair_label']}",
        "structure": config["structure_label"],
        "bond": config["pair_label"],
        "scope": report["scope"],
        "labelA": config["atom_a"]["label"],
        "labelB": config["atom_b"]["label"],
        "R": round(config["bond_length_angstrom"], 4),
        "W": round(report["pair_squared_coherence_total"], 6),
        "WSelectedA": round(selected_a_star, 6),
        "WSelectedB": (
            round(report["selected_atom_b_star_squared_coherence_total"], 6)
            if "selected_atom_b_star_squared_coherence_total" in report else None
        ),
        "WFullA": (
            round(report["full_atom_a_star_squared_coherence_total"], 6)
            if "full_atom_a_star_squared_coherence_total" in report else None
        ),
        "WFullB": (
            round(report["full_atom_b_star_squared_coherence_total"], 6)
            if "full_atom_b_star_squared_coherence_total" in report else None
        ),
        "canonicalSpectrum": [round(value, 5) for value in canonical],
        "canonicalSupportRanks": [
            canonical_analysis.get("atom_a_support_rank"),
            canonical_analysis.get("atom_b_support_rank"),
        ],
        "nIAO": report["density_dimension"],
        "channels": channels,
        "contextPairs": [
            {"label": c["label"], "W": round(c["squared_coherence_with_atom_a"], 6),
             "frac": round(c.get("fraction_of_selected_atom_a_star", c.get("fraction_of_context_total")), 5)}
            for c in report["context_pair_table"]
        ],
        "atoms": atoms,
        "bonds": bonds,
        "cellEdges": cell_edges,
        "fieldOffset": [float(v) for v in centre - pos_a],
        "targetIndex": target_index,
        "grid": {"n": samples, "half": half},
        "volumes": volumes,
        "availableFields": [
            key for key in ("bond", "anti", "pa", "pb", "rho")
            if f"{key}1" in volumes
        ],
        "hasChannelSum": is_bvk,
        "cellLabel": "局部框" if is_bvk else "晶胞",
        "bvkClosed": is_bvk,
        "bvkMesh": manifest.get("bvk", {}).get("mesh"),
        "gridChecks": manifest["grid_checks"],
        "allElectron": manifest.get("all_electron_reconstruction", True),
        "scientificStatus": manifest.get(
            "scientific_status",
            "qualitative_proxy_only; legacy product unit factor corrected while packing",
        ),
        "quantitativeUse": manifest.get("quantitative_use", False),
        "legacyProductScaleCorrection": BOHR**3 if legacy_products else 1.0,
    }


def self_test() -> None:
    volume = np.arange(27.0).reshape(3, 3, 3)
    sampled = sample_box(
        volume, np.eye(3) * 3.0, np.ones(3), 0.0, 1,
        origin=np.zeros(3), periodic=False,
    )
    assert sampled.shape == (1, 1, 1) and sampled.item() == volume[1, 1, 1]
    try:
        sample_box(volume, np.eye(3) * 3.0, np.full(3, 4.0), 0.0, 1,
                   origin=np.zeros(3), periodic=False)
    except ValueError:
        pass
    else:
        raise AssertionError("non-periodic bounds check did not fire")
    print("build_channel_viewer self-test: PASS")




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields-root", type=Path, required=True)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument(
        "--structures-root", type=Path, required=True,
        help="<root>/<structure>/CONTCAR tree",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--half-width", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--order", nargs="*", default=None)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Pack only these field-directory names")
    args = parser.parse_args()

    directories = sorted(p for p in args.fields_root.iterdir() if p.is_dir())
    if args.only:
        selected = set(args.only)
        directories = [path for path in directories if path.name in selected]
    if args.order:
        rank = {name: index for index, name in enumerate(args.order)}
        directories = sorted(directories, key=lambda p: rank.get(p.name, len(rank)))

    pairs = []
    for directory in directories:
        pair_dir = args.pairs_root / directory.name
        if not (pair_dir / "result").is_dir():
            print(f"  skip {directory.name}: no pair result")
            continue
        print(f"  packing {directory.name}", flush=True)
        pairs.append(build_pair(
            directory, pair_dir, args.half_width, args.samples, args.structures_root
        ))

    bundle = {
        "generator": "build_channel_viewer.py",
        "pairs": pairs,
        "note": (
            "For BvK-closed fields, phi_bond=(A+B)/sqrt(2) and "
            "phi_antibond=(A-B)/sqrt(2) are normalized density-derived channel orbitals. "
            "Weighted partner products 2 s_i Re[A_i*B_i] remain qualitative shape proxies."
        ),
    }
    assets = Path(__file__).resolve().parent.parent / "viewer"
    three = base64.b64encode(gzip.compress((assets / "three.module.js").read_bytes())).decode("ascii")
    orbit = base64.b64encode(gzip.compress((assets / "OrbitControls.js").read_bytes())).decode("ascii")
    payload = base64.b64encode(
        gzip.compress(
            json.dumps(bundle, separators=(",", ":"), allow_nan=False).encode("utf-8"), 9
        )
    ).decode("ascii")
    app = Path(__file__).with_name("channel_viewer_app.js").read_text(encoding="utf-8")

    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PAW-IAO-PDC channel viewer</title>
</head><body>
<script id="three-data" type="application/octet-stream">{three}</script>
<script id="orbit-data" type="application/octet-stream">{orbit}</script>
<script id="bundle-data" type="application/octet-stream">{payload}</script>
<script type="text/plain" id="app-module">{app}</script>
<script>
"use strict";
(async () => {{
  if (!("DecompressionStream" in self)) {{
    document.body.textContent = "This page needs a browser with DecompressionStream (Chrome/Edge/Firefox 113+).";
    return;
  }}
  async function unpack(id) {{
    const encoded = document.getElementById(id).textContent.trim();
    const raw = atob(encoded);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
  }}
  const [threeSource, orbitSource, bundleText] = await Promise.all(
    [unpack("three-data"), unpack("orbit-data"), unpack("bundle-data")]);
  const threeUrl = URL.createObjectURL(new Blob([threeSource], {{type: "text/javascript"}}));
  // keep the `from` keyword: replacing the whole match with just the quoted URL
  // produces `import {{...}} "blob:..."` and a SyntaxError
  const orbitUrl = URL.createObjectURL(new Blob(
    [orbitSource.replace(/from\\s*["'][^"']*three[^"']*["']/g, "from " + JSON.stringify(threeUrl))],
    {{type: "text/javascript"}}));
  const appSource = document.getElementById("app-module").textContent
    .replace("__THREE_URL__", threeUrl).replace("__ORBIT_URL__", orbitUrl);
  const appUrl = URL.createObjectURL(new Blob([appSource], {{type: "text/javascript"}}));
  window.__BUNDLE__ = JSON.parse(bundleText);
  await import(appUrl);
}})();
</script>
</body></html>
"""
    args.output.write_text(html, encoding="utf-8")
    size = args.output.stat().st_size
    print(f"viewer -> {args.output.resolve()}  ({size/1e6:.2f} MB, {len(pairs)} pairs)")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        main()
