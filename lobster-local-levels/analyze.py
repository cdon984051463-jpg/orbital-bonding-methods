# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""uv run analyze.py input.json result.json — LOBSTER 5.1.1, ISPIN=1."""
import json
import re
import sys
from pathlib import Path

import numpy as np


def matrices(path):
    """Read one complex k block at a time; Real/Imag repeat the same header."""
    pending = None
    current = None
    mode = None
    rows, labels = [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = re.search(r"matrix for (?:spin (\d+) )?kpoint (\d+) at\s+(.*)", line, re.I)
            if match:
                if int(match[1] or 1) != 1:
                    raise ValueError("Only non-spin-polarized LOBSTER matrices are supported")
                pending = (int(match[2]), np.array(list(map(float, match[3].split()[:3]))))
                continue
            low = line.strip().lower()
            if low.startswith("real parts"):
                if current is not None:
                    if mode != "imag" or labels != real_labels:
                        raise ValueError("Incomplete or mismatched complex matrix")
                    yield current, real_labels, real + 1j*np.array(rows)
                if pending is None:
                    raise ValueError("Missing k-point header")
                current = pending
                mode, rows, labels = "real", [], []
                continue
            if low.startswith(("imag parts", "imaginary parts")):
                if mode != "real" or pending[0] != current[0] or not rows:
                    raise ValueError("Imaginary block does not match the real block")
                real, real_labels = np.array(rows), list(labels)
                mode, rows, labels = "imag", [], []
                continue
            if mode and low and "basisfunction" not in low:
                label, numbers = line.split(maxsplit=1)
                labels.append(label)
                rows.append(list(map(float, numbers.split())))
    if current is None or mode != "imag" or labels != real_labels:
        raise ValueError("Incomplete matrix file")
    yield current, real_labels, real + 1j*np.array(rows)


def eigenval(path, occupation_factor):
    lines = path.read_text().splitlines()
    nelect, nk, nb = map(int, lines[5].split()[:3])
    kpoints, weights, energies, occupations = [], [], [], []
    i = 6
    for _ in range(nk):
        while not lines[i].strip():
            i += 1
        row = list(map(float, lines[i].split()))
        kpoints.append(row[:3]); weights.append(row[3]); i += 1
        states = np.array([list(map(float, x.split())) for x in lines[i:i+nb]])
        if states.shape != (nb, 3):
            raise ValueError("Only ISPIN=1 EIGENVAL is supported")
        energies.append(states[:, 1]); occupations.append(states[:, 2]*occupation_factor); i += nb
    w, e, f = np.array(weights), np.array(energies), np.array(occupations)
    if abs(w.sum()-1) > 1e-5 or abs(w @ f.sum(axis=1)-nelect) > 1e-3:
        raise ValueError("k weights or explicit occupation_factor do not reproduce NELECT")
    if f.min() < -1e-8 or f.max() > 2+1e-8:
        raise ValueError("Electron occupations must be between 0 and 2")
    return nelect, np.array(kpoints), w, e, f


def analyze(config, base):
    if set(config["a"]["orbitals"]) & set(config["b"]["orbitals"]):
        raise ValueError("The two orbitals must belong to disjoint atomic AO subspaces")
    for side in ("a", "b"):
        if not np.isclose(sum(float(x)**2 for x in config[side]["orbitals"].values()), 1, atol=1e-10, rtol=0):
            raise ValueError("Use normalized real AO direction coefficients")
    paths = {key: (base / config[key]).resolve() for key in
             ("hamilton", "overlap", "coefficients_lso2", "eigenval", "lobsterout")}
    nelect, ks, w, energies, occupations = eigenval(paths["eigenval"], config["occupation_factor"])
    spilling = [float(x) / 100 for x in re.findall(r"abs\. charge spilling:\s*([\d.]+)%", paths["lobsterout"].read_text())]
    if not spilling:
        raise ValueError("No absolute charge spilling value in lobsterout")
    checks = dict(H_hermiticity_eV=0., S_hermiticity=0., occupied_gram_error=0.,
                  H_orth_residual_rms_eV=0., local_H_spectral_error_eV=0.,
                  pair_orthogonality_error=0., population_bound_violation=0.,
                  electron_trace=0., charge_spilling_fraction=max(spilling))
    records = []
    hblocks = {block[0][0]: block for block in matrices(paths["hamilton"])}
    sblocks = {block[0][0]: block for block in matrices(paths["overlap"])}
    cblocks = {block[0][0]: block for block in matrices(paths["coefficients_lso2"])}
    if not set(hblocks) == set(sblocks) == set(cblocks) == set(range(1, len(w)+1)):
        raise ValueError("Matrix files must contain the same EIGENVAL k-point IDs")
    blocks = ((hblocks[k], sblocks[k], cblocks[k]) for k in sorted(hblocks))
    for i, triple in enumerate(blocks):
        (mh, labels, ho), (ms, sl, s), (mc, cl, co) = triple
        if not labels == sl == cl or i >= len(w):
            raise ValueError("Matrix basis labels or k-point counts disagree")
        for meta in (mh, ms, mc):
            difference = meta[1]-ks[i]
            if meta[0] != i+1 or np.max(abs(difference-np.rint(difference))) > 2e-6:
                raise ValueError("Matrix and EIGENVAL k-points disagree")
        if not all(np.isfinite(x).all() for x in (ho, s, co)):
            raise ValueError("Nonfinite matrix entry")
        if ho.shape != s.shape or s.shape != (len(labels), len(labels)) or co.shape[0] != len(labels):
            raise ValueError("Matrix dimensions disagree")
        se, su = np.linalg.eigh(s)
        if se.min() <= 1e-8:
            raise ValueError("Nonpositive or numerically singular overlap matrix")
        sq = (su*np.sqrt(se)) @ su.conj().T
        h = sq @ ho @ sq
        # S C_AO = S^(1/2) C_LSO2; no explicit full inverse is needed.
        sc = sq @ co
        nb = co.shape[1]
        if nb > energies.shape[1] or abs(occupations[i, nb:]).sum() > 1e-8:
            raise ValueError("Coefficient bands do not cover the occupied state set")
        e, f = energies[i, :nb], occupations[i, :nb]
        occupied = f > 1e-6
        cg = co[:, occupied].conj().T @ co[:, occupied]
        residual = ho @ co - co*e
        checks["H_hermiticity_eV"] = max(checks["H_hermiticity_eV"], float(np.max(abs(ho-ho.conj().T))))
        checks["S_hermiticity"] = max(checks["S_hermiticity"], float(np.max(abs(s-s.conj().T))))
        checks["occupied_gram_error"] = max(checks["occupied_gram_error"], float(np.max(abs(cg-np.eye(len(cg))))))
        checks["H_orth_residual_rms_eV"] = max(checks["H_orth_residual_rms_eV"], float(np.linalg.norm(residual)/np.sqrt(residual.size)))
        checks["electron_trace"] += float(w[i]*np.sum(f*np.sum(abs(co)**2, axis=0)))
        index = {label: n for n, label in enumerate(labels)}
        b = np.zeros((len(labels), 2), dtype=complex)
        for j, side in enumerate(("a", "b")):
            for label, value in config[side]["orbitals"].items():
                b[index[label], j] = float(value)
        metric = b.conj().T @ s @ b
        ev, u = np.linalg.eigh(metric)
        if ev.min() <= 1e-8:
            raise ValueError("The two selected orbitals are not independent")
        q = b @ ((u/np.sqrt(ev)) @ u.conj().T)
        h2 = q.conj().T @ h @ q
        amp = q.conj().T @ sc
        d2 = (amp*f) @ amp.conj().T
        levels, v = np.linalg.eigh(h2)
        if levels[1]-levels[0] < 1e-10:
            raise ValueError("Degenerate pair: separate lower/upper eigenvector directions are undefined")
        n = np.real(np.diag(v.conj().T @ d2 @ v))
        de = np.linalg.eigvalsh(d2)
        offdiag = 2*np.real(v[0].conj()*h2[0, 1]*v[1])
        checks["local_H_spectral_error_eV"] = max(checks["local_H_spectral_error_eV"], float(np.max(abs((amp*e) @ amp.conj().T-h2))))
        checks["pair_orthogonality_error"] = max(checks["pair_orthogonality_error"], float(np.max(abs(q.conj().T @ s @ q-np.eye(2)))))
        checks["population_bound_violation"] = max(checks["population_bound_violation"], float(max(0, -de.min(), de.max()-2)))
        records.append(dict(k=i+1, weight=float(w[i]),
            epsilon_a_relative_EF_eV=float(h2[0, 0].real-config["fermi_eV"]),
            epsilon_b_relative_EF_eV=float(h2[1, 1].real-config["fermi_eV"]),
            delta_eV=float((h2[0, 0]-h2[1, 1]).real), hopping_abs_eV=float(abs(h2[0, 1])),
            E_lower_relative_EF_eV=float(levels[0]-config["fermi_eV"]),
            E_upper_relative_EF_eV=float(levels[1]-config["fermi_eV"]),
            a_fraction_upper=float(abs(v[0, 1])**2), n_lower_e=float(n[0]), n_upper_e=float(n[1]),
            pair_population_index=float((n[0]-n[1])/2),
            offdiag_lower_eV_per_electron=float(offdiag[0]), offdiag_upper_eV_per_electron=float(offdiag[1])))
    if len(records) != len(w):
        raise ValueError("Not all EIGENVAL k-points have matrices")
    checks["electron_error"] = abs(checks["electron_trace"]-nelect)
    limits = dict(H_hermiticity_eV=1e-5, S_hermiticity=1e-7, occupied_gram_error=1e-3,
                  H_orth_residual_rms_eV=1e-4, local_H_spectral_error_eV=1e-3,
                  pair_orthogonality_error=1e-7, population_bound_violation=1e-3,
                  electron_error=1e-3, charge_spilling_fraction=0.02)
    gates = {name: checks[name] <= limit for name, limit in limits.items()}
    summary = {key: sum(row["weight"]*row[key] for row in records)
               for key in records[0] if key not in ("k", "weight")}
    return dict(method="LOBSTER pair-local Lowdin two-orbital Hamiltonian",
                definition_revision="2026-11-04",
                input=config, resolved_sources={k: str(v) for k, v in paths.items()},
                numerical_pass=all(gates.values()), gates=gates, checks=checks, limits=limits,
                expected_electrons=nelect, summary=summary, k_resolved=records,
                model_scope="LOBSTER 5.1.1 nonmagnetic matrices, no SOC; pair-local Lowdin projection of two fixed Bloch-AO directions at each k, followed by weighted means; no image-resolved real-space hopping or environment self-energy",
                interpretation=[
                    "Lower/upper levels are local Ritz values, not isolated crystal bands; populations are projections of the occupied KS states, not a Fermi function of the local levels",
                    "Orbital fractions refer to pair-Lowdin partners; pair_population_index is a two-orbital population-difference index, not total bond order, ICOHP or dissociation energy",
                    "Separate self-consistent calculations generally have different local eigenvectors; changes in their populations do not establish fixed-orbital electron removal or a causal force decomposition",
                    "Each calculation uses its own Fermi reference; absolute cross-state level shifts require an independent common energy reference",
                    "numerical_pass checks algebra and the stated projection diagnostics, not basis/band-window convergence or acceptance of a physical mechanism",
                ])


if __name__ == "__main__":
    config_path, output_path = map(Path, sys.argv[1:])
    if output_path.exists():
        raise FileExistsError(output_path)
    result = analyze(json.loads(config_path.read_text(encoding="utf-8-sig")), config_path.resolve().parent)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"numerical_pass": result["numerical_pass"], "output": str(output_path)}, ensure_ascii=False))
    sys.exit(0 if result["numerical_pass"] else 1)
