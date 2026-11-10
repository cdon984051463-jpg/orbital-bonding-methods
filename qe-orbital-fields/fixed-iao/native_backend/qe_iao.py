#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "scipy"]
# ///
"""QE native-PAW -> full-cell IAO -> spin-resolved Ru/O density SVD.

No wavefunction format conversion to VASP, no SCF, no approximate H operator.
Full UPF reference shells are retained (including Ru semicore). The resulting
reference convention is QE_UPF_full, not the old VASP Ru(ZVAL=8) template.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
from qe_wfc import inventory

CORE = Path(__file__).resolve().parents[3] / 'iao-svd/support/ruo2_ibo.py'
RY_EV = 13.605693122994


def check(condition, message):
    if not condition:
        raise ValueError(message)


def hermitian(a, label='matrix', tol=1e-7):
    a = np.asarray(a, complex)
    check(a.ndim == 2 and a.shape[0] == a.shape[1] and np.isfinite(a).all(), label + ': invalid matrix')
    check(np.max(abs(a-a.conj().T)) < tol, label + ': non-Hermitian')
    return (a+a.conj().T)*.5


def invsqrt(g, rcond=1e-11):
    e, u = np.linalg.eigh(hermitian(g))
    check(e[0] > rcond*e[-1] and e[-1] > 0, 'Rank deficient reference/occupied projection')
    return (u / np.sqrt(e)) @ u.conj().T


def iao_function(core_path: Path = CORE):
    """Use the validated original IAO core; do not silently substitute a method."""
    if not core_path.is_file():
        raise FileNotFoundError(core_path)
    spec = importlib.util.spec_from_file_location('original_iao_matrix_core', core_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_intrinsic_atomic_orbitals, str(core_path)


def read_gram(path: Path):
    with path.open('rb') as f:
        check(f.read(8) == b'QEIAO001', 'Wrong Gram magic')
        sentinel = f.read(4)
        endian = '<' if sentinel == b'\x04\x03\x02\x01' else '>'
        check(struct.unpack(endian+'i', sentinel)[0] == 16909060, 'Wrong endian sentinel')
        nb, nr, ik, spin, nspin, nks, ng = struct.unpack(endian+'7i', f.read(28))
        check(0 < nb < 100000 and 0 < nr < 100000 and nspin in (1, 2), 'Bad Gram dimensions')
        expected = 80+16*nb+16*(nb*nb+nb*nr+nr*nr)
        check(path.stat().st_size == expected, f'Truncated Gram file {path}')
        def array(n, dtype):
            result = np.fromfile(f, dtype=endian+dtype, count=n).astype(np.dtype(dtype))
            check(len(result) == n and np.isfinite(result).all(), 'Invalid Gram data')
            return result
        kweight = array(5, 'f8')
        et, wg = array(nb, 'f8'), array(nb, 'f8')
        gcc = array(nb*nb, 'c16').reshape((nb, nb), order='F')
        gcf = array(nb*nr, 'c16').reshape((nb, nr), order='F')
        gff = array(nr*nr, 'c16').reshape((nr, nr), order='F')
    return dict(nb=nb, nr=nr, ik=ik, spin=spin, nspin=nspin, nks=nks, ng=ng,
                k=kweight[:3], wk=kweight[3], nelec=kweight[4], et=et, wg=wg,
                gcc=gcc, gcf=gcf, gff=gff)


def build_block(block, fn, floor=1e-6):
    wk = block['wk']
    check(wk > 0, 'Zero-weight band-path k points are not density samples')
    occ = block['wg']/wk
    check(occ.min() >= -1e-8 and occ.max() <= 1+1e-8, 'Occupation convention is not supported')
    active = np.flatnonzero(occ > floor)
    check(len(active) > 0, 'Empty occupied window')
    check(len(active) <= block['nr'], 'Insufficient full-cell references; never drop occupied bands silently')
    a = hermitian(block['gcc'][np.ix_(active, active)], 'occupied PAW Gram')
    gram_error = float(np.max(abs(a-np.eye(len(a)))))
    check(gram_error < 1e-6, f'QE PAW orthonormality failed: {gram_error}')
    b = block['gcf'][active]
    d = hermitian(block['gff'], 'reference PAW Gram')
    g = np.block([[a, b], [b.conj().T, d]])
    e, u = np.linalg.eigh(hermitian(g, 'joint Gram'))
    check(e.min() > -1e-7, f'Indefinite joint Gram: {e.min()}')
    keep = e > e.max()*1e-11
    x = np.sqrt(e[keep])[:, None]*u[:, keep].conj().T
    no = len(active)
    raw_c, f = x[:, :no], x[:, no:]
    c = raw_c@invsqrt(a)
    iao, diag = fn(f, c, rcond=1e-11)
    q = raw_c.conj().T@iao
    closure = float(np.max(abs(c.conj().T@iao@iao.conj().T@c-np.eye(no))))
    orth = float(np.max(abs(iao.conj().T@iao-np.eye(iao.shape[1]))))
    check(closure < 1e-6 and orth < 1e-7, 'IAO occupied closure or orthogonality failed')
    weights = block['wg'][active]
    density = q.conj().T@(weights[:, None]*q)
    trace_error = abs(np.trace(density).real-weights.sum())
    check(trace_error < max(1e-6, 1e-7*weights.sum()), 'Density electron trace failed')
    tail = float(block['wg'].sum()-weights.sum())
    transform = (u[:, keep]/np.sqrt(e[keep]))@iao
    return density, dict(Q=q, weights=weights, active_bands=active, transform=transform,
                         energies_eV=block['et'][active]*RY_EV), dict(
        PAW_occupied_gram_error=gram_error, IAO_orthogonality_error=orth,
        occupied_closure_error=closure, density_trace_error=trace_error,
        excluded_occupation_e=tail, electrons=float(weights.sum()), n_occupied=no,
        n_reference=block['nr'], joint_rank=int(keep.sum()))


def k_contract(blocks, mesh):
    mesh = np.asarray(mesh, int)
    check(np.all(mesh > 0), 'Invalid unshifted k mesh')
    nmesh = int(np.prod(mesh))
    points, folded = set(), {}
    for b in blocks:
        deg = 2 if b['nspin'] == 1 else 1
        mult = b['wk']*nmesh/deg
        m = int(round(mult))
        check(m in (1, 2) and abs(mult-m) < 1e-7, 'Unexpected k weights; spatial symmetry/shifted meshes unsupported')
        scaled = b['k']*mesh
        check(np.max(abs(scaled-np.rint(scaled))) < 1e-6, 'k not on declared unshifted mesh')
        idx = tuple((np.rint(scaled).astype(int) % mesh).tolist())
        rev = tuple((-np.rint(scaled).astype(int) % mesh).tolist())
        check(idx not in points, 'Duplicate represented k point')
        points.add(idx)
        if m == 2:
            check(rev != idx and rev not in points, 'Invalid time-reversal weight')
            points.add(rev)
        folded[b['ik']] = m == 2
    check(len(points) == nmesh, f'Incomplete k mesh: {len(points)} / {nmesh}')
    return folded


def labels_read(path):
    rows = []
    for line in path.read_text().splitlines()[1:]:
        col, atom, species, element, shell, l, m, radial = line.split()
        rows.append(dict(column=int(col)-1, atom=int(atom), species=species, element=element,
                         shell=shell.upper(), l=int(l), m=int(m), radial=int(radial)))
    check([x['column'] for x in rows] == list(range(len(rows))), 'Reference labels lost/order changed')
    return rows


def analyze(export: Path, output: Path, pairs, mesh, core=CORE):
    check((export/'EXPORT_COMPLETE').is_file(), 'Native export is incomplete')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='STARTED', method='QE_UPF_full_shell_IAO_density_SVD',
                  original_VASP_template_equivalent=False, all_electron_realspace_export=False,
                  note='W=sum(s^2) is a density-channel measure, not bond order or bond energy.',
                  per_k=[], spins={})
    def save():
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        fn, core_name = iao_function(core)
        report['IAO_matrix_core'] = core_name
        labels = labels_read(export/'labels.tsv')
        (output/'labels.json').write_text(json.dumps(labels, indent=2)+'\n')
        paths = sorted(export.glob('k*.gram'))
        headers = [read_gram(p) for p in paths]
        check(len(headers) == int((export/'EXPORT_COMPLETE').read_text()) == headers[0]['nks'], 'Missing k blocks')
        check(all(b['nr'] == len(labels) and b['nspin'] == headers[0]['nspin'] and
                  abs(b['nelec']-headers[0]['nelec']) < 1e-8 for b in headers), 'Inconsistent k blocks')
        check(len(set(b['ik'] for b in headers)) == len(headers), 'Duplicated k index')
        nspin = headers[0]['nspin']
        source_electrons = float(sum(b['wg'].sum() for b in headers))
        check(abs(source_electrons-headers[0]['nelec']) < 1e-5, 'QE occupations do not reproduce NELECT')
        total_trace, total_tail = 0., 0.
        for spin in range(1, nspin+1):
            blocks = [b for b in headers if b['spin'] == spin]
            check(blocks, 'Missing spin channel')
            folded = k_contract(blocks, mesh)
            sp = output/f'spin{spin}'
            sp.mkdir()
            density = np.zeros((len(labels), len(labels)), complex)
            for b in blocks:
                dk, arrays, diagnostics = build_block(b, fn)
                if folded[b['ik']]:
                    dk = (dk+dk.conj())*.5
                density += dk
                total_tail += diagnostics['excluded_occupation_e']
                np.savez(sp/f'k{b["ik"]:05d}.npz', **arrays, k_crystal=b['k'],
                         QE_wk=b['wk'], time_reversal_pair_averaged=folded[b['ik']])
                report['per_k'].append(dict(ik=b['ik'], spin=spin, **diagnostics))
                save()
            check(np.max(abs(density.imag)) < 1e-7, 'Time-reversal reality check failed; refusing to discard imaginary density')
            density = hermitian(density).real
            eig, vectors = np.linalg.eigh(density)
            check(eig.min() > -1e-7, 'Density is not positive semidefinite')
            np.save(sp/'density.npy', density)
            np.save(sp/'eigenvalues.npy', eig)
            np.save(sp/'basis.npy', vectors.T)
            channels = []
            for a, b in pairs:
                acols = [x['column'] for x in labels if x['atom'] == a and x['shell'] in ('4D', '5S')]
                bcols = [x['column'] for x in labels if x['atom'] == b and x['shell'] == '2P']
                check(len(acols) == 6 and len(bcols) == 3, f'Pair {a}:{b}: require verified Ru5s4d/O2p shell labels')
                ab = density[np.ix_(acols, bcols)]
                u, s, vh = np.linalg.svd(ab, full_matrices=False)
                check(np.max(abs((u*s)@vh-ab)) < 1e-10, 'SVD reconstruction failed')
                np.savez(sp/f'pair_{a}_{b}.npz', singular_values=s, U=u, Vh=vh, D_AB=ab,
                         a_columns=acols, b_columns=bcols)
                channels.append(dict(Ru=a, O=b, singular_values=s.tolist(), W=float(s@s)))
            trace = float(np.trace(density))
            total_trace += trace
            report['spins'][str(spin)] = dict(electrons=trace, pairs=channels)
        check(total_tail < 5e-4, 'Too much occupation excluded by floor')
        check(abs(total_trace+total_tail-source_electrons) < 1e-4, 'Total density trace failed')
        report.update(status='NUMERICAL_GATES_PASS', total_electrons=total_trace,
                      excluded_occupation_e=total_tail, source_electrons=source_electrons,
                      kmesh=list(mesh), spin_matrices_must_not_be_added=True,
                      production_cross_code_reference_validation=False)
        save()
    except Exception as exc:
        report.update(status='FAILED', error=repr(exc))
        save()
        raise
    return report


def run_native(save: Path, source_log: Path, work: Path, exe: Path, ranks: int, pseudo_dir=None):
    """Read a finished result through a separate stage; symlink large WFC only."""
    log = source_log.read_text(errors='replace')
    check('JOB DONE.' in log and 'convergence has been achieved' in log, 'Source calculation is not finished/converged')
    check(not re.search(r'convergence\s+NOT\s+achieved|Error in routine', log, re.I), 'Source calculation contains errors')
    info = inventory(save)
    check(info['ready_for_native_export'], 'Incomplete/unsupported QE wavefunctions')
    work.mkdir(parents=True, exist_ok=False)
    prefix = save.name[:-5] if save.name.endswith('.save') else 'source'
    checkpoint = work/'checkpoint'
    checkpoint.mkdir()
    staged = checkpoint/(prefix+'.save')
    staged.mkdir()
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in save.iterdir() if p.is_file()}
    for p in save.iterdir():
        if not p.is_file():
            continue
        if re.fullmatch(r'wfc(?:up|dw)?\d+\.(?:dat|hdf5|h5)', p.name):
            (staged/p.name).symlink_to(p.resolve())
        elif p.name in ('data-file-schema.xml', 'charge-density.dat', 'charge-density.hdf5', 'paw.txt') or p.suffix.upper() == '.UPF':
            shutil.copyfile(p, staged/p.name)
    if pseudo_dir:
        for p in Path(pseudo_dir).glob('*.UPF'):
            if not (staged/p.name).exists():
                shutil.copyfile(p, staged/p.name)
    check((staged/'paw.txt').exists(), 'PAW restart data missing')
    export = work/'export'
    export.mkdir()
    (work/'source_inventory.json').write_text(json.dumps(info, indent=2)+'\n')
    for parent in (save.parent, save.parent.parent):
        field = parent/'ru_d_field.nml'
        if field.exists():
            shutil.copyfile(field, work/'source_ru_d_field.nml')
            break
    text = f"&iao_export\n prefix='{prefix}', outdir='{checkpoint}/', export_dir='{export}'\n/\n"
    (work/'export.in').write_text(text)
    cmd = [str(exe), '-in', str(work/'export.in')]
    if ranks > 1:
        cmd = ['mpirun', '-np', str(ranks)] + cmd
    with (work/'export.log').open('w') as so, (work/'export.err').open('w') as se:
        subprocess.run(cmd, cwd=work, stdout=so, stderr=se, check=True)
    for name, old in before.items():
        p = Path(name)
        check(p.exists() and (p.stat().st_size, p.stat().st_mtime_ns) == old, 'Source changed during export')
    check((export/'EXPORT_COMPLETE').is_file(), 'Native exporter did not finish')
    return export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inspect')
    p.add_argument('save', type=Path)
    p = sub.add_parser('export')
    p.add_argument('save', type=Path)
    p.add_argument('--source-log', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--ranks', type=int, default=1)
    p.add_argument('--pseudo-dir', type=Path)
    p = sub.add_parser('analyze')
    p.add_argument('export', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kmesh', type=int, nargs=3, required=True)
    p.add_argument('--pair', action='append', default=[])
    p.add_argument('--core', type=Path, default=CORE)
    args = parser.parse_args()
    if args.command == 'inspect':
        print(json.dumps(inventory(args.save), indent=2))
    elif args.command == 'export':
        check(args.ranks > 0, 'Invalid MPI ranks')
        print(run_native(args.save.resolve(), args.source_log.resolve(), args.work.resolve(), args.exe.resolve(), args.ranks, args.pseudo_dir))
    else:
        pairs = [tuple(map(int, p.split(':'))) for p in (args.pair or ['1:113', '1:93'])]
        check(all(len(p) == 2 for p in pairs), 'Pairs use Ru_index:O_index')
        r = analyze(args.export.resolve(), args.output.resolve(), pairs, args.kmesh, args.core)
        print(json.dumps({k: r[k] for k in ('status', 'total_electrons', 'spins')}, indent=2))


if __name__ == '__main__':
    main()
