#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""OFE_V1 history file (ofe_history.dat) parser and writer, plus a minimal pw.x log parser (V1.2, 2026-10-26).

V1.2: parses OFE_SITE_ATOM / OFE_SITE_ANCHORS (header) and the H_ch records of the shared V1.2 interface
(OFE_HCH, OFE_HCH_FIELD[_IM], OFE_HCH_IM, OFE_HCH_IMAG_RAW_MAX, OFE_HCH_K, header OFE_HCH_UNITS); label words
after a record keyword are optional for the H_ch records.  Helpers: group layout, spin-summed N_AO, native
H_ch block, run loader (XML etot > qe.log), unit constants.

The Fortran engine appends one OFE_HEADER segment per run and one OFE_BEGIN_STEP .. OFE_END_STEP block per
force call (raw / native / final records are written by the later forces.f90 hooks inside the same block).
parse_history() keeps the LAST segment and returns its header and every step block (last = final).
write_history() mirrors the Fortran record formats (used by the validation self-test).
Units: energy Ry, force Ry/bohr (F = -dE/dR), positions bohr; D in eV; N in electrons.
"""
import os
import re
import numpy as np

RY_BOHR_TO_EV_A = 25.71104309541616     # 13.605693122994 / 0.529177210903
RYTOEV = 13.605693122994

PREFIX = {   # label words written after the keyword by orbital_field_engine.f90
    'OFE_N': 'site spin row', 'OFE_N_IM': 'site spin row', 'OFE_N_AO': 'site spin row',
    'OFE_FRAME': 'site col', 'OFE_N_EIG_MIN_MAX_TRACE': 'site spin', 'OFE_EEXT': 'site spin Ry',
    'OFE_DEDV': 'site name spin value_eV dE/dV(e)', 'OFE_FORCE_SPLIT': 'site atom phi aug frame cder',
    'OFE_FORCE_PHI_FROZEN_DIAG': 'site atom (not applied)', 'OFE_SUMRULE': 'site sum_proj commutator sum_frame',
    'OFE_FORCE_RAW': 'atom pos_bohr raw', 'OFE_SITE_STEP': 'site tag kind cond_max',
    'OFE_DLOC_EV': 'site spin row re im', 'OFE_PARAM': 'site name spin value_eV',
    'OFE_RUN': 'nspin nkstot nsym nat nsite naotot nchtot 0 0 trev restart',
}
HCH_KEYS = ('OFE_HCH', 'OFE_HCH_FIELD', 'OFE_HCH_FIELD_IM', 'OFE_HCH_IM')
BOHR_A = 0.529177210903


def _is_num(x):
    try:
        _num(x)
        return True
    except ValueError:
        return False


def _drop_labels(tok):
    """Drop leading non-numeric label words (e.g. 'site spin row') so both 'OFE_HCH 1 1 1 ...' and
    'OFE_HCH site spin row 1 1 1 ...' parse."""
    k = 0
    while k < len(tok) and not _is_num(tok[k]):
        k += 1
    return tok[k:]


def _num(x):
    x = x.replace('D', 'E')
    if 'E' not in x and 'e' not in x:
        x = re.sub(r'(\d)([+-]\d{3})$', r'\1E\2', x)    # Fortran Ew.d with a 3-digit exponent drops the E
    return float(x)


def _nums(tok):
    return [_num(x) for x in tok]


def parse_history(path_or_text):
    text = path_or_text if '\n' in path_or_text else open(path_or_text).read()
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith('OFE_HEADER version')]
    if not starts:
        raise ValueError('no OFE_HEADER segment')
    seg = lines[starts[-1]:]
    hdr = {'version': seg[0].split()[2], 'sites': {}, 'params': [], 'dloc': {}, 'header_lines': []}
    steps, cur = [], None
    for ln in seg:
        if not ln.strip():
            continue
        key = ln.split()[0]
        rest = ln[len(key):].strip()
        pre = PREFIX.get(key)
        if pre and rest.startswith(pre):
            rest = rest[len(pre):].strip()
        tok = rest.split()
        if key == 'OFE_HEADER' and not rest.startswith('OFE_HCH_UNITS'):
            hdr['header_lines'].append(rest)
        elif key == 'OFE_RUN':
            v = tok
            hdr['run'] = dict(nspin=int(v[0]), nkstot=int(v[1]), nsym=int(v[2]), nat=int(v[3]), nsite=int(v[4]),
                              naotot=int(v[5]), nchtot=int(v[6]), time_reversal=v[9] == 'T', restart=v[10] == 'T')
        elif key == 'OFE_SITE':
            hdr['sites'][int(tok[0])] = dict(kind=tok[1], tag=tok[2], frame=tok[3], center=tok[4], spin=tok[5],
                                            natg=int(tok[6]), nao=int(tok[7]), nch=int(tok[8]))
        elif key == 'OFE_SITE_ATOM':
            # 'OFE_SITE_ATOM ' is g atom img(3) element nsh labels...
            s_, g = int(tok[0]), int(tok[1])
            nsh = int(tok[7])
            hdr.setdefault('site_atoms', {}).setdefault(s_, {})[g] = dict(
                atom=int(tok[2]), img=(int(tok[3]), int(tok[4]), int(tok[5])), element=tok[6], shells=tok[8:8 + nsh])
        elif key == 'OFE_SITE_ANCHORS':
            v = tok
            s_ = int(v[0])
            anc = tuple((int(v[1 + 4 * i]), (int(v[2 + 4 * i]), int(v[3 + 4 * i]), int(v[4 + 4 * i]))) for i in range(3))
            hdr.setdefault('site_anchors', {})[s_] = dict(anchors=anc, use_x=v[13].upper() in ('T', '.TRUE.'))
        elif key == 'OFE_HCH_UNITS' or (key == 'OFE_HEADER' and rest.startswith('OFE_HCH_UNITS')):
            _u = rest.split()
            _u = _u[1:] if _u and _u[0] == 'OFE_HCH_UNITS' else _u
            hdr['hch_units'] = _u[0].rstrip(':') if _u else ''
            if key == 'OFE_HEADER':
                hdr['header_lines'].append(rest)
        elif key == 'OFE_PARAM':
            hdr['params'].append(dict(site=int(tok[0]), name=tok[1], spin=int(tok[2]), value=float(tok[3])))
        elif key == 'OFE_DLOC_EV':
            s, sp, i = int(tok[0]), int(tok[1]), int(tok[2]); v = _nums(tok[3:]); n = len(v) // 2
            hdr['dloc'].setdefault((s, sp), {})[i] = np.array(v[:n]) + 1j * np.array(v[n:])
        elif key == 'OFE_BEGIN_STEP':
            cur = dict(istep=int(tok[0]), frame={}, N={}, N_im={}, N_ao={}, eig={}, eext={}, dedv={}, split={},
                       phid={}, sumrule={}, expl={}, raw={}, pos={}, native={}, final={}, ended=False,
                       HCH={}, HCH_FIELD={}, HCH_FIELD_IM={}, HCH_IM={}, hch_imag_raw_max={}, HCH_K={},
                       diag_hch={}, hch_rayleigh=None)
            steps.append(cur)
        elif cur is None:
            continue
        elif key == 'OFE_H_APPLY_COUNT':
            cur['h_apply_count'] = int(tok[0])
        elif key == 'OFE_SITE_STEP':
            cur.setdefault('cond', {})[int(tok[0])] = float(tok[3])
        elif key == 'OFE_FRAME':
            cur['frame'].setdefault(int(tok[0]), np.zeros((3, 3)))[:, int(tok[1]) - 1] = _nums(tok[2:5])
        elif key in ('OFE_N', 'OFE_N_IM', 'OFE_N_AO'):
            d = {'OFE_N': cur['N'], 'OFE_N_IM': cur['N_im'], 'OFE_N_AO': cur['N_ao']}[key]
            d.setdefault((int(tok[0]), int(tok[1])), {})[int(tok[2])] = _nums(tok[3:])
        elif key in HCH_KEYS:
            t2 = _drop_labels(tok)
            cur[key[4:]].setdefault((int(t2[0]), int(t2[1])), {})[int(t2[2])] = _nums(t2[3:])
        elif key == 'OFE_HCH_IMAG_RAW_MAX':
            t2 = _drop_labels(tok)
            cur['hch_imag_raw_max'][(int(t2[0]), int(t2[1]))] = (_num(t2[2]), t2[3] if len(t2) > 3 else '')
        elif key == 'OFE_DIAG_HCH':
            # engine V1.2 diagnostic: site spin herm_Ry xsite_Ry lowdin (xsite = other sites' field left in H_ch - D)
            t2 = _drop_labels(tok)
            cur['diag_hch'][(int(t2[0]), int(t2[1]))] = dict(herm_Ry=_num(t2[2]), xsite_Ry=_num(t2[3]), lowdin=_num(t2[4]))
        elif key == 'OFE_DIAG_HCH_RAYLEIGH':
            t2 = _drop_labels(tok)
            cur['hch_rayleigh'] = dict(nbands=int(t2[0]), maxerr_Ry=_num(t2[1]))
        elif key == 'OFE_HCH_K':
            t2 = _drop_labels(tok)
            v = _nums(t2[4:]); n = len(v) // 2
            cur['HCH_K'].setdefault((int(t2[0]), int(t2[1]), int(t2[2])), {})[int(t2[3])] = \
                np.array(v[:n]) + 1j * np.array(v[n:])
        elif key == 'OFE_N_EIG_MIN_MAX_TRACE':
            cur['eig'][(int(tok[0]), int(tok[1]))] = _nums(tok[2:5])
        elif key == 'OFE_EEXT':
            cur['eext'][(int(tok[0]), int(tok[1]))] = float(tok[2])
        elif key == 'OFE_DEDV':
            cur['dedv'][(int(tok[0]), tok[1], int(tok[2]))] = (float(tok[3]), float(tok[4]))
        elif key == 'OFE_FORCE_SPLIT':
            cur['split'][(int(tok[0]), int(tok[1]))] = np.array(_nums(tok[2:14])).reshape(4, 3)
        elif key == 'OFE_FORCE_PHI_FROZEN_DIAG':
            cur['phid'][(int(tok[0]), int(tok[1]))] = np.array(_nums(tok[2:5]))
        elif key == 'OFE_SUMRULE':
            cur['sumrule'][int(tok[0])] = np.array(_nums(tok[1:10])).reshape(3, 3)
        elif key == 'OFE_N_IMAG_RAW_MAX':
            cur['n_imag_raw_max'] = (float(tok[0]), tok[1])
        elif key == 'OFE_EEXT_TOTAL_RY_ALREADY_IN_EBAND':
            cur['eext_total'] = float(tok[0])
        elif key == 'OFE_FORCE_EXPLICIT_TOTAL':
            cur['expl'][int(tok[0])] = np.array(_nums(tok[1:4]))
        elif key == 'OFE_FORCE_EXPLICIT_SUM':
            cur['expl_sum'] = np.array(_nums(tok[0:3]))
        elif key == 'OFE_RAW_FORCE_SUM':
            cur['raw_sum'] = np.array(_nums(tok[0:3]))
        elif key == 'OFE_FORCE_RAW':
            cur['pos'][int(tok[0])] = np.array(_nums(tok[1:4])); cur['raw'][int(tok[0])] = np.array(_nums(tok[4:7]))
        elif key == 'OFE_FORCE_NATIVE':
            cur['native'][int(tok[0])] = np.array(_nums(tok[1:4]))
        elif key == 'OFE_FORCE_FINAL_QE':
            cur['final'][int(tok[1])] = np.array(_nums(tok[2:5]))
        elif key == 'OFE_END_STEP':
            cur['ended'] = True
    for st in steps:
        for name in ('N', 'N_im', 'N_ao', 'HCH', 'HCH_FIELD', 'HCH_FIELD_IM', 'HCH_IM', 'HCH_K'):
            st[name] = {k: np.array([v[i] for i in sorted(v)]) for k, v in st[name].items()}
        for name in ('HCH', 'HCH_FIELD', 'HCH_IM', 'HCH_FIELD_IM', 'HCH_K'):
            for k, m in st[name].items():
                if m.shape[0] != m.shape[1]:
                    raise ValueError('%s %s: %d rows but %d columns' % (name, k, m.shape[0], m.shape[1]))
        nat = len(st['expl'])
        for name in ('expl', 'raw', 'native', 'final', 'pos'):
            if st[name]:
                st[name + '_arr'] = np.array([st[name][i] for i in sorted(st[name])])
        st['nat'] = nat
    return {'header': hdr, 'steps': steps, 'last': steps[-1] if steps else None}


def _e(x):
    return '%25.16E' % x


def write_history(fh, run, sites, step):
    """Mirror of the Fortran writes (for tests/self-test).  sites: [dict(kind, tag, frame, center, spin, natg, nao,
    nch, dloc=[nch x nch per spin] eV, params=[(name, spin, value)])]; step: dict with the per-step arrays."""
    w = fh.write
    w('OFE_HEADER version OFE_V1_20261025\n')
    w('OFE_HEADER units energy Ry; force Ry/bohr (F = -dE/dR); position bohr; D printed in eV\n')
    w('OFE_RUN nspin nkstot nsym nat nsite naotot nchtot 0 0 trev restart %8d%8d%8d%8d%8d%8d%8d%8d%8d  %s  %s\n' %
      (run['nspin'], run['nkstot'], 1, run['nat'], len(sites), sum(s['nao'] for s in sites),
       sum(s['nch'] for s in sites), 0, 0, 'T' if run.get('trev', True) else 'F', 'T' if run.get('restart') else 'F'))
    for i, s in enumerate(sites, 1):
        w('OFE_SITE %4d %s %s %s %s %s%5d%5d%5d\n' % (i, s['kind'], s['tag'], s['frame'].upper(), s['center'].upper(),
                                                     s['spin'].upper(), s['natg'], s['nao'], s['nch']))
        for sp in range(2):
            for r in range(s['nch']):
                row = np.asarray(s['dloc'][sp][r])
                w('OFE_DLOC_EV site spin row re im %4d%4d%4d' % (i, sp + 1, r + 1) +
                  ''.join('%20.11E' % v for v in row.real) + ''.join('%20.11E' % v for v in row.imag) + '\n')
        for g, at in enumerate(s.get('atoms', []), 1):        # V1.2 (optional): (atom, img, element, shells)
            a_, img, el, sh = at
            w('OFE_SITE_ATOM %4d%4d%7d%4d%4d%4d %s %3d %s\n' % (i, g, a_, img[0], img[1], img[2], el, len(sh), ' '.join(sh)))
        if s.get('anchors') is not None:
            w('OFE_SITE_ANCHORS %4d' % i + ''.join('%7d%4d%4d%4d' % (a_, im[0], im[1], im[2]) for a_, im in s['anchors'])
              + '  %s\n' % ('T' if s.get('use_x', True) else 'F'))
        for name, sp, val in s.get('params', []):
            w('OFE_PARAM site name spin value_eV %4d %s%3d%20.11E\n' % (i, name, sp, val))
    if run.get('hch_units'):
        w('OFE_HEADER OFE_HCH_UNITS %s\n' % run['hch_units'])
    w('OFE_BEGIN_STEP %8d\n' % step.get('istep', 0))
    w('OFE_H_APPLY_COUNT %12d\n' % step.get('h_apply_count', 0))
    for i, s in enumerate(sites, 1):
        w('OFE_SITE_STEP site tag kind cond_max %4d %s %s%12.4E\n' % (i, s['tag'], s['kind'], 1.5))
        q = step['frame'][i]
        for j in range(3):
            w('OFE_FRAME site col %4d%4d' % (i, j + 1) + ''.join(_e(v) for v in q[:, j]) + '\n')
        for sp in range(run['nspin']):
            n = step['N'][(i, sp + 1)]
            for r in range(s['nch']):
                w('OFE_N site spin row %4d%4d%4d' % (i, sp + 1, r + 1) + ''.join(_e(v) for v in n[r].real) + '\n')
            if (i, sp + 1) in step.get('N_ao', {}):                 # V1.2 writer option (engine: print_nao)
                nao_ = np.asarray(step['N_ao'][(i, sp + 1)])
                for r in range(nao_.shape[0]):
                    w('OFE_N_AO site spin row %4d%4d%4d' % (i, sp + 1, r + 1) + ''.join(_e(v) for v in nao_[r].real) + '\n')
            if (i, sp + 1) in step.get('HCH', {}):
                _write_hch(w, i, sp + 1, step, run.get('hch_labels', False))
            ev = np.linalg.eigvalsh(0.5 * (n.real + n.real.T))
            w('OFE_N_EIG_MIN_MAX_TRACE site spin %4d%4d%20.11E%20.11E%20.11E\n' % (i, sp + 1, ev.min(), ev.max(),
                                                                                  np.trace(n).real))
            w('OFE_EEXT site spin Ry %4d%4d%s\n' % (i, sp + 1, _e(step['eext'][(i, sp + 1)])))
            for (si, name, spn), (val, dv) in sorted(step.get('dedv', {}).items()):
                if si == i and spn == sp + 1:
                    w('OFE_DEDV site name spin value_eV dE/dV(e) %4d %s%3d%s%s\n' % (i, name, spn, _e(val), _e(dv)))
        for a in range(run['nat']):
            sp4 = step['split'][(i, a + 1)]
            w('OFE_FORCE_SPLIT site atom phi aug frame cder %6d%6d' % (i, a + 1) +
              ''.join('%20.11E' % v for v in sp4.ravel()) + '\n')
        sr = step['sumrule'][i]
        w('OFE_SUMRULE site sum_proj commutator sum_frame %4d' % i + ''.join(_e(v) for v in sr.ravel()) + '\n')
    w('OFE_N_IMAG_RAW_MAX %14.6E TR_REDUCED_K_ARTEFACT\n' % step.get('n_imag_raw_max', 0.0))
    w('OFE_EEXT_TOTAL_RY_ALREADY_IN_EBAND %s\n' % _e(sum(step['eext'].values())))
    ex = step['expl']
    for a in range(run['nat']):
        w('OFE_FORCE_EXPLICIT_TOTAL %6d' % (a + 1) + ''.join(_e(v) for v in ex[a]) + '\n')
    w('OFE_FORCE_EXPLICIT_SUM ' + ''.join(_e(v) for v in ex.sum(0)) + '\n')
    raw = step['raw']
    w('OFE_RAW_FORCE_SUM ' + ''.join(_e(v) for v in raw.sum(0)) + '\n')
    for a in range(run['nat']):
        w('OFE_FORCE_RAW atom pos_bohr raw %6d' % (a + 1) + ''.join(_e(v) for v in step['pos'][a]) +
          ''.join(_e(v) for v in raw[a]) + '\n')
    for a in range(run['nat']):
        w('OFE_FORCE_NATIVE %6d' % (a + 1) + ''.join(_e(v) for v in raw[a] - ex[a]) + '\n')
    fin = raw - raw.mean(0)
    for a in range(run['nat']):
        w('OFE_FORCE_FINAL_QE %6d%6d' % (step.get('istep', 0), a + 1) + ''.join(_e(v) for v in fin[a]) + '\n')
    w('OFE_END_STEP %8d\n' % step.get('istep', 0))


def _write_hch(w, i, sp, step, labels=False):
    """V1.2 H_ch records (shared interface item 4; Ry, es25.16).  step['HCH'][(i, sp)] complex k-average
    (imag written only if step['full_mesh']), step['HCH_FIELD'][(i, sp)] (D in Ry; _IM only if complex),
    optional step['HCH_K'][(i, sp)] = {ik_global: complex matrix}."""
    lab = (lambda s: s + ' ') if labels else (lambda s: '')
    hc = np.asarray(step['HCH'][(i, sp)], complex)
    hf = np.asarray(step['HCH_FIELD'][(i, sp)], complex)
    full = step.get('full_mesh', False)
    for r in range(hc.shape[0]):
        w('OFE_HCH %s%4d%4d%4d' % (lab('site spin row'), i, sp, r + 1) + ''.join(_e(v) for v in hc[r].real) + '\n')
    for r in range(hf.shape[0]):
        w('OFE_HCH_FIELD %s%4d%4d%4d' % (lab('site spin row'), i, sp, r + 1) + ''.join(_e(v) for v in hf[r].real) + '\n')
    if np.any(hf.imag != 0):
        for r in range(hf.shape[0]):
            w('OFE_HCH_FIELD_IM %s%4d%4d%4d' % (lab('site spin row'), i, sp, r + 1) + ''.join(_e(v) for v in hf[r].imag) + '\n')
    if full:
        for r in range(hc.shape[0]):
            w('OFE_HCH_IM %s%4d%4d%4d' % (lab('site spin row'), i, sp, r + 1) + ''.join(_e(v) for v in hc[r].imag) + '\n')
    w('OFE_HCH_IMAG_RAW_MAX %s%4d%4d%14.6E %s\n' % (lab('site spin'), i, sp, step.get('hch_imag_raw', np.max(np.abs(hc.imag))),
                                                   'FULL_MESH_PHYSICAL' if full else 'TR_REDUCED_K_ARTEFACT'))
    for ik, hk in sorted(step.get('HCH_K', {}).get((i, sp), {}).items()):
        hk = np.asarray(hk, complex)
        for r in range(hk.shape[0]):
            w('OFE_HCH_K %s%4d%4d%6d%4d' % (lab('site spin ik row'), i, sp, ik, r + 1) + ''.join(_e(v) for v in hk[r].real)
              + ''.join(_e(v) for v in hk[r].imag) + '\n')


# --------------------------------------------------------------------- V1.2 helpers
def site_group(hist, site):
    """[(atom, img, element, [shells])] of a site from the OFE_SITE_ATOM header records."""
    sa = hist['header'].get('site_atoms', {}).get(site)
    if not sa:
        raise ValueError('history has no OFE_SITE_ATOM records for site %d' % site)
    return [(v['atom'], v['img'], v['element'], list(v['shells'])) for g, v in sorted(sa.items())]


def n_ao_spin_summed(step, site, nspin):
    """Spin-summed AO density (lab frame): nspin 1 records are already spin-summed; nspin 2 sums spins 1, 2."""
    keys = [(site, 1)] if nspin == 1 else [(site, 1), (site, 2)]
    for k in keys:
        if k not in step['N_ao']:
            raise ValueError('OFE_N_AO missing for site %d spin %d (print_nao = .true. needed)' % k)
    return sum(step['N_ao'][k] for k in keys)


def hch_blocks(step, site, spin):
    """-> (H_ch, D_field, H_native) in Ry for one site/spin.  H_ch real part (+ i Im when the full-mesh OFE_HCH_IM
    records exist); D_field from OFE_HCH_FIELD[_IM]; H_native = H_ch - D_field (shared interface item 2)."""
    k = (site, spin)
    if k not in step['HCH']:
        raise ValueError('no OFE_HCH records for site %d spin %d (print_hch = .true. and a V1.2 engine needed)' % k)
    hc = step['HCH'][k] + (1j * step['HCH_IM'][k] if k in step['HCH_IM'] else 0)
    if k not in step['HCH_FIELD']:
        raise ValueError('OFE_HCH present but OFE_HCH_FIELD missing for site %d spin %d' % k)
    hf = step['HCH_FIELD'][k] + (1j * step['HCH_FIELD_IM'][k] if k in step['HCH_FIELD_IM'] else 0)
    return np.asarray(hc, complex), np.asarray(hf, complex), np.asarray(hc - hf, complex)


def xml_energy(path):
    """etot (Ha) of data-file-schema.xml in Ry, or None."""
    if not os.path.exists(path):
        return None
    m = re.search(r'<etot>\s*([-+0-9.Ee]+)\s*</etot>', open(path).read())
    return None if m is None else 2.0 * float(m.group(1))


def load_run(d):
    """Run directory -> dict(dir, hist, log, E (Ry; XML etot preferred, full precision), E_precise, job)."""
    out = dict(dir=d, hist=None, log=None, E=None, E_precise=False, job=None)
    if os.path.exists(os.path.join(d, 'job.json')):
        import json
        out['job'] = json.load(open(os.path.join(d, 'job.json')))
    if os.path.exists(os.path.join(d, 'qe.log')):
        out['log'] = parse_qe_log(open(os.path.join(d, 'qe.log')).read())
        out['E'] = out['log']['energy']
    ex = xml_energy(os.path.join(d, 'data-file-schema.xml'))
    if ex is not None:
        out['E'], out['E_precise'] = ex, True
    if os.path.exists(os.path.join(d, 'ofe_history.dat')):
        out['hist'] = parse_history(open(os.path.join(d, 'ofe_history.dat')).read())
    return out


def run_ok(r):
    """converged, JOB DONE, no error block, energy present."""
    lg = r.get('log')
    return bool(lg and lg['energy'] is not None and lg['converged'] and lg['job_done'] and not lg['errors'])


def param_values(hist):
    """{(site, name, spin): value_eV} from the OFE_PARAM header records."""
    return {(p['site'], p['name'], p['spin']): p['value'] for p in hist['header']['params']}


def dedv_total(step, site, name):
    """sum over spins of OFE_DEDV for (site, name): the dE/dV conjugate (electrons)."""
    return sum(v[1] for (s, n, sp), v in step['dedv'].items() if s == site and n == name)


def force_reading(step, reading):
    """(nat, 3) Ry/bohr: 'raw' (QE total before net-force removal), 'native' (= raw - explicit), 'explicit'."""
    key = {'raw': 'raw_arr', 'native': 'native_arr', 'explicit': 'expl_arr', 'final': 'final_arr'}[reading]
    if key not in step:
        raise ValueError('force reading %s not in history step' % reading)
    return step[key]


# --------------------------------------------------------------------- pw.x log
def parse_qe_log(path_or_text):
    text = path_or_text if '\n' in path_or_text else open(path_or_text).read()
    out = {'energy': None, 'forces': None, 'converged': 'convergence has been achieved' in text,
           'job_done': 'JOB DONE' in text, 'errors': [], 'clocks': {}, 'ofe': {}}
    e = re.findall(r'^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry', text, re.M)
    if e:
        out['energy'] = float(e[-1])
    m = [mm.end() for mm in re.finditer(r'Forces acting on atoms \(cartesian axes, Ry/au\):', text)]
    if m:
        blk = text[m[-1]:].split('The ')[0]
        f = re.findall(r'atom\s+(\d+)\s+type\s+\d+\s+force =\s+(\S+)\s+(\S+)\s+(\S+)', blk.split('Total force')[0])
        out['forces'] = np.array([[float(x) for x in r[1:]] for r in f])
    for mm in re.finditer(r'%%%%+\s*\n(.*?)\n\s*%%%%+', text, re.S):
        out['errors'].append(' '.join(mm.group(1).split()))
    for name in ('PWSCF', 'electrons', 'c_bands', 'h_psi', 'forces', 'ofe_hpsi', 'ofe_force'):
        cm = re.findall(r'^\s*%s\s*:.*? CPU\s+(.+?) WALL' % re.escape(name), text, re.M)
        if cm:
            out['clocks'][name] = sum(float(value) * {'d': 86400, 'h': 3600, 'm': 60, 's': 1}[unit]
                                      for value, unit in re.findall(r'([\d.]+)\s*([dhms])', cm[-1]))
    it = re.findall(r'convergence has been achieved in\s+(\d+) iterations', text)
    if it:
        out['iterations'] = int(it[-1])
    sr = re.findall(r'OFE_SUMRULE_MAXERR\s+(\S+)\s+OFE_FRAME_SUM_MAX\s+(\S+)', text)
    if sr:
        out['ofe']['sumrule_maxerr'] = float(sr[-1][0]); out['ofe']['frame_sum_max'] = float(sr[-1][1])
    pe = re.findall(r'OFE_ANGULAR_PHASE_ERROR\s+(\S+)', text)
    if pe:
        out['ofe']['phase_error'] = float(pe[-1])
    return out
