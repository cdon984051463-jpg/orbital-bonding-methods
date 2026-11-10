#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""Read QE 7.x collected WFC .dat/.hdf5 without loading every band.

This is a coefficient reader, NOT a PAW overlap evaluator. Use the companion
native exporter for physical PAW overlaps. Gamma half storage and spinors are
reported correctly but are deliberately excluded from IAO backend version 1.
"""
from __future__ import annotations
import argparse
import json
import re
import struct
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


class Wfc:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.before = self._stat()
        self.h5 = None
        if self.path.suffix in ('.hdf5', '.h5'):
            self._hdf5()
        else:
            self._binary()
        if self.npol not in (1, 2) or not 0 < self.nbnd <= 100000:
            raise ValueError('Invalid WFC dimensions')
        if self.gvec.shape != (self.ng, 3) or len(np.unique(self.gvec, axis=0)) != self.ng:
            raise ValueError('Invalid/duplicate Miller indices')
        self.check_stable()

    def _stat(self):
        s = self.path.stat()
        return s.st_size, s.st_mtime_ns

    def check_stable(self):
        if self._stat() != self.before:
            raise RuntimeError('Source WFC changed during reading; use a finished snapshot')

    def _binary(self):
        # QE Modules/io_base.f90: four metadata records, then one complex128
        # record per band. Fortran record markers may be 4 or 8 bytes.
        self.offsets = []
        with self.path.open('rb') as f:
            lead = f.read(8)
            candidates = []
            for endian in ('<', '>'):
                for marker, code in ((4, 'i'), (8, 'q')):
                    if len(lead) >= marker and struct.unpack(endian + code, lead[:marker])[0] == 44:
                        f.seek(marker + 44)
                        tail = f.read(marker)
                        if len(tail) == marker and struct.unpack(endian + code, tail)[0] == 44:
                            candidates.append((endian, marker, code))
            if len(candidates) != 1:
                raise ValueError('Not a supported QE collected WFC; ambiguous or corrupt header')
            self.endian, self.marker, self.marker_code = candidates[0]
            f.seek(0)

            def record(expected=None, skip=False):
                raw = f.read(self.marker)
                if len(raw) != self.marker:
                    raise ValueError('Truncated Fortran record')
                n = struct.unpack(self.endian + self.marker_code, raw)[0]
                if n < 0 or (expected is not None and n != expected):
                    raise ValueError(f'Unexpected record size {n}, expected {expected}')
                off = f.tell()
                if skip:
                    f.seek(n, 1)
                    value = b''
                else:
                    value = f.read(n)
                    if len(value) != n:
                        raise ValueError('Truncated record payload')
                tail = f.read(self.marker)
                if len(tail) != self.marker or struct.unpack(self.endian + self.marker_code, tail)[0] != n:
                    raise ValueError('Record trailer mismatch')
                return value, off

            h = struct.unpack(self.endian + 'i3diid', record(44)[0])
            self.ik, self.k_cart, self.spin, self.gamma_only, self.scale = h[0], np.array(h[1:4]), h[4], bool(h[5]), h[6]
            self.ngw, self.ng, self.npol, self.nbnd = struct.unpack(self.endian + '4i', record(16)[0])
            if not 0 < self.ng <= 100000000 or self.npol not in (1, 2) or not 0 < self.nbnd <= 100000:
                raise ValueError('Invalid dimensions')
            self.reciprocal = np.frombuffer(record(72)[0], self.endian + 'f8').reshape(3, 3).copy()
            self.gvec = np.frombuffer(record(12 * self.ng)[0], self.endian + 'i4').reshape(self.ng, 3).copy()
            for _ in range(self.nbnd):
                self.offsets.append(record(16 * self.ng * self.npol, skip=True)[1])
            if f.tell() != self.before[0]:
                raise ValueError('Unexpected trailing WFC bytes')

    def _hdf5(self):
        import h5py
        self.h5 = h5py.File(self.path, 'r')
        h = self.h5.attrs
        self.ik, self.spin = int(h['ik']), int(h['ispin'])
        self.ngw, self.ng = int(h['ngw']), int(h['igwx'])
        self.npol, self.nbnd = int(h['npol']), int(h['nbnd'])
        self.k_cart = np.asarray(h['xk'], float)
        flag = h['gamma_only']
        if isinstance(flag, bytes):
            flag = flag.decode()
        self.gamma_only = str(flag).strip().upper() in ('.TRUE.', 'TRUE', '1')
        self.scale = float(h['scale_factor'])
        g = self.h5['MillerIndices']
        self.gvec = np.asarray(g, dtype=int)
        if self.gvec.shape != (self.ng, 3):
            raise ValueError('Unsupported HDF5 MillerIndices layout')
        self.reciprocal = np.array([g.attrs['bg' + str(i)] for i in (1, 2, 3)])
        shape = self.h5['evc'].shape
        if shape != (self.nbnd, 2 * self.npol * self.ng):
            raise ValueError(f'Unsupported HDF5 evc layout {shape}')

    def band(self, index: int) -> np.ndarray:
        """Zero-based band, shape (npol, nG); no implicit PAW normalization."""
        if not 0 <= index < self.nbnd:
            raise IndexError(index)
        self.check_stable()
        if self.h5 is not None:
            a = np.asarray(self.h5['evc'][index], dtype=np.float64)
            c = a[0::2] + 1j * a[1::2]
        else:
            with self.path.open('rb') as f:
                f.seek(self.offsets[index])
                raw = f.read(16 * self.ng * self.npol)
            c = np.frombuffer(raw, self.endian + 'c16').astype(np.complex128)
        if c.size != self.ng * self.npol or not np.isfinite(c).all():
            raise ValueError('Invalid coefficients')
        self.check_stable()
        return c.reshape(self.npol, self.ng)

    def info(self):
        return dict(path=str(self.path), bytes=self.before[0], GiB=self.before[0] / 2**30,
                    ik=self.ik, spin=self.spin, nG=self.ng, npol=self.npol, nbnd=self.nbnd,
                    gamma_only=self.gamma_only, scale_factor=self.scale,
                    k_cart=self.k_cart.tolist(), physical_PAW_normalization_tested=False)

    def close(self):
        if self.h5 is not None:
            self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def xml_info(save: Path):
    root = ET.parse(save / 'data-file-schema.xml').getroot()
    for e in root.iter():
        e.tag = e.tag.split('}')[-1]
    out = root.find('output')
    if out is None:
        raise ValueError('XML has no output section')
    b = out.find('band_structure')
    if b is None:
        raise ValueError('XML has no band structure')
    lsda = (b.findtext('lsda') or out.findtext('magnetization/lsda') or 'false').strip().lower() == 'true'
    spinor = (b.findtext('noncolin') or 'false').strip().lower() == 'true'
    spinorbit = (b.findtext('spinorbit') or 'false').strip().lower() == 'true'
    conv = out.findtext('convergence_info/scf_conv/convergence_achieved')
    return dict(nspin=2 if lsda else 1, noncollinear=spinor, spinorbit=spinorbit,
                nk=int(b.findtext('nks')), nelec=float(b.findtext('nelec')),
                nbnd_up=int(b.findtext('nbnd_up') or b.findtext('nbnd')),
                nbnd_down=int(b.findtext('nbnd_dw') or b.findtext('nbnd_up') or b.findtext('nbnd')),
                scf_converged=(conv or '').strip().lower() == 'true')


def inventory(save: Path):
    save = Path(save).resolve()
    meta = xml_info(save)
    files = sorted(p for p in save.iterdir() if re.fullmatch(r'wfc(?:up|dw)?\d+\.(?:dat|hdf5|h5)', p.name))
    infos = []
    for path in files:
        with Wfc(path) as f:
            infos.append(f.info())
    counts = {s: sum(i['spin'] == s for i in infos) for s in range(1, meta['nspin'] + 1)}
    complete = bool(files) and all(v == meta['nk'] for v in counts.values())
    unsupported = meta['noncollinear'] or meta['spinorbit'] or any(i['npol'] != 1 or i['gamma_only'] for i in infos)
    errors = []
    for s, n in counts.items():
        if n != meta['nk']:
            errors.append(f'spin {s}: {n} files, expected {meta["nk"]}')
    for i in infos:
        expected_bands = meta['nbnd_up'] if i['spin'] == 1 else meta['nbnd_down']
        if i['nbnd'] != expected_bands or abs(i['scale_factor'] - 1.) > 1e-12:
            errors.append('Band-count or PW scale-factor mismatch: ' + i['path'])
    return dict(source=str(save), metadata=meta, files=infos,
                total_bytes=sum(i['bytes'] for i in infos),
                total_GiB=sum(i['GiB'] for i in infos), errors=errors,
                ready_for_native_export=complete and not unsupported and not errors and meta['scf_converged'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    if args.path.is_dir():
        print(json.dumps(inventory(args.path), indent=2))
    else:
        with Wfc(args.path) as f:
            print(json.dumps(f.info(), indent=2))
