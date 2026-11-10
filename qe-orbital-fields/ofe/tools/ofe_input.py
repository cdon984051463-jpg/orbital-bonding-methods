#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""OFE_V1 input tools (V1.2 front end, 2026-10-26): site builders for the four designs, canonical ofe.in
writer/reader, JSON compiler.

V1.2 additions (Python only; engine KINDs unchanged): RU_O_PI subspace 'sigma2' (d_z2, p_z) and full8 block
named parameters; JSON pass-through of d_full8 / d_pi4_dn / bond-path frame & centre keys; IAO_CHANNEL JSON
pair_params, user PARAM blocks and complex D; the sigma+5s constructor (SIGMA_5S -> IAO_CHANNEL);
&ofe_control print_hch / print_hch_k (written only when given); bottom-bond sites default to
anchor_max_A = group_radius_A = 3.6 A (noted in ofe.in); every unknown or inapplicable JSON key is an error.

The Fortran engine (orbital_field_engine.f90) reads ONE canonical file, ofe.in:
  &ofe_control model='OFE_V1', nsite=N, ... /
  SITE i ... END_SITE        (generic group + frame + C + D + named parameters)
This module turns design-specific parameters into that generic form (all conversions documented here and
in qe-orbital-fields/README.md), and parses ofe.in back (for the validation driver/tests).

Conventions (identical to the engine):
  * AO phase (-i)^l, positive Cartesian real harmonics; d order z2, xz, yz, x2-y2, xy; p order x, y, z.
    => real-space orbitals are +Cartesian lobes for every l (QE atomic_wfc i^l gives p = -Cartesian).
  * image vectors: position = tau + at . img (QE crystal units of the lattice vectors).
  * frame: z = unit(pos(z_from) - pos(origin)), x = orthonormalised pos(x_from) - pos(origin)
    (x_from = 0: constant helper axis, field must be axially symmetric), y = z cross x;
    FRAME_REF lists the columns x, y, z.  D_ao = C_lab D C_lab^T, C_lab = Rblk(Q) C_loc.
  * D in eV in ofe.in (engine converts with 1 Ry = 13.605693122994 eV); E_ext = Tr(D N).
  * PARAM name spin value nnz + entries (i j re im): basis matrix B of a named parameter V, D = sum V B
    + rest; the engine prints dE/dV = Re Tr(B N_ch) (the Hellmann-Feynman conjugate).
Python >= 3.5, numpy >= 1.13.
"""
import json
import sys
import numpy as np

RYTOEV = 13.605693122994
KINDS = ('RU_D', 'O_P_BOND', 'RU_O_PI', 'IAO_CHANNEL')
D_LABELS = ('z2', 'xz', 'yz', 'x2-y2', 'xy')
P_LABELS = ('x', 'y', 'z')
PI4_INDEX = (1, 2, 5, 6)          # 0-based local AO indices of (d_xz, d_yz, p_x, p_y) in the 8-AO RU_O_PI group


def shell_l(label):
    s = label.strip().upper()
    if s[0] == 'L':
        return int(s[1])
    return {'S': 0, 'P': 1, 'D': 2}[s[1]]


class Site(object):
    """Canonical OFE site (the content of one SITE ... END_SITE block)."""

    def __init__(self, kind, tag, atoms, frame='moving', anchors=None, anchor_elements=None, frame_ref=None,
                 xhelper=None, center='moving', center_ref=None, cloc=None, dmat=None, spin='shared',
                 params=None, relax_validated=False, bottom=None):
        if kind not in KINDS:
            raise ValueError('kind must be one of %s' % (KINDS,))
        self.kind, self.tag = kind, tag
        # atoms: list of (atom_index_1based, img(3), element, [shell labels])
        self.atoms = [(int(a), tuple(int(v) for v in img), str(el), list(sh)) for a, img, el, sh in atoms]
        self.frame = frame.lower()
        # anchors: ((origin, img), (z_from, img), (x_from or 0, img))
        self.anchors = None if anchors is None else tuple((int(a), tuple(int(v) for v in img)) for a, img in anchors)
        self.anchor_elements = anchor_elements
        self.frame_ref = None if frame_ref is None else np.asarray(frame_ref, float)
        self.xhelper = None if xhelper is None else np.asarray(xhelper, float)
        self.center = center.lower()
        self.center_ref = {} if center_ref is None else {int(k): np.asarray(v, float) for k, v in center_ref.items()}
        nao = self.nao
        self.cloc = np.eye(nao) if cloc is None else np.asarray(cloc, float)
        self.cidentity = cloc is None
        nch = self.cloc.shape[1]
        if dmat is None:
            dmat = [np.zeros((nch, nch), complex)]
        self.dmat = [np.asarray(d, complex) for d in dmat]
        self.spin = spin.lower()
        if self.spin == 'shared':
            self.dmat = [self.dmat[0], self.dmat[0].copy()]
        elif len(self.dmat) == 1:
            self.dmat = [self.dmat[0], np.zeros_like(self.dmat[0])]
        self.params = [] if params is None else list(params)   # (name, spin, value_eV, B)
        self.relax_validated = bool(relax_validated)
        self.bottom = bottom          # V1.2: True/False = user flag, None = auto (bottom O list)
        self.validate()

    @property
    def nao(self):
        return sum(2 * shell_l(s) + 1 for _, _, _, sh in self.atoms for s in sh)

    @property
    def nch(self):
        return self.cloc.shape[1]

    def shell_list(self):
        """[(group_index, atom, img, l, start)] in AO order."""
        out, start = [], 0
        for g, (a, img, el, sh) in enumerate(self.atoms):
            for s in sh:
                l = shell_l(s)
                out.append((g, a, img, l, start, s))
                start += 2 * l + 1
        return out

    def validate(self):
        nao, nch = self.nao, self.nch
        if self.cloc.shape[0] != nao:
            raise ValueError('C_LOCAL rows %d != nao %d' % (self.cloc.shape[0], nao))
        if np.max(np.abs(self.cloc.T @ self.cloc - np.eye(nch))) > 1e-8:
            raise ValueError('C_LOCAL columns not orthonormal')
        for d in self.dmat:
            if d.shape != (nch, nch) or np.max(np.abs(d - d.conj().T)) > 1e-12:
                raise ValueError('D must be Hermitian nch x nch')
        if self.frame not in ('moving', 'frozen', 'global'):
            raise ValueError('frame')
        if self.center not in ('moving', 'frozen'):
            raise ValueError('center')
        if self.frame == 'moving' and self.anchors is None:
            raise ValueError('moving frame needs anchors')
        if self.frame == 'frozen' and self.anchors is None and self.frame_ref is None:
            raise ValueError('frozen frame needs FRAME_REF or anchors')
        for name, sp, val, b in self.params:
            b = np.asarray(b, complex)
            if b.shape != (nch, nch) or np.max(np.abs(b - b.conj().T)) > 1e-12:
                raise ValueError('PARAM %s basis must be Hermitian nch x nch' % name)

    def d_from_params(self, spin_index=0):
        """sum_V V B over the named parameters acting on this spin (eV)."""
        out = np.zeros((self.nch, self.nch), complex)
        for name, sp, val, b in self.params:
            if sp in (0, spin_index + 1):
                out += val * np.asarray(b, complex)
        return out


# --------------------------------------------------------------------- helpers
def _e(n, i, j, val=1.0):
    m = np.zeros((n, n), complex)
    m[i, j] += val
    return m


def frame_from_dirs(zdir, xdir=None):
    """Columns (x, y, z) from a z direction and an optional x direction (orthonormalised)."""
    z = np.asarray(zdir, float); z = z / np.linalg.norm(z)
    if xdir is None:
        h = np.zeros(3); h[int(np.argmin(np.abs(z)))] = 1.0
        xdir = h
    x = np.asarray(xdir, float); x = x - z * (z @ x); x = x / np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


# --------------------------------------------------------------------- builders (four designs)
def build_ru_d(atom, top, side, top_img=(0, 0, 0), side_img=(0, 0, 0), v=None, dloc=None, v_dn=None,
               dloc_dn=None, frame='moving', frame_ref=None, element='Ru', tag=None, center='moving',
               center_ref=None, shell='L2', relax_validated=False, bottom=None):
    """RU_D: one metal atom, the unique occupied l=2 PP_CHI ('L2') or an explicit label ('4D').
    Images are OFE images (position = tau + at.img).  v: 5 diagonal values (eV, local frame, order
    z2 xz yz x2-y2 xy); dloc: full 5x5 real symmetric; *_dn: spin-down (spin resolved)."""
    if dloc is None:
        dloc = np.diag(np.zeros(5) if v is None else np.asarray(v, float))
    resolved = (v_dn is not None) or (dloc_dn is not None)
    dl = [np.asarray(dloc, complex)]
    if resolved:
        dd = np.diag(np.asarray(v_dn, float)) if dloc_dn is None else np.asarray(dloc_dn, float)
        dl.append(np.asarray(dd, complex))
    params = []
    for i, lab in enumerate(D_LABELS):
        if resolved:
            params.append(('v_%s_up' % lab, 1, float(np.real(dl[0][i, i])), _e(5, i, i)))
            params.append(('v_%s_dn' % lab, 2, float(np.real(dl[1][i, i])), _e(5, i, i)))
        else:
            params.append(('v_%s' % lab, 0, float(np.real(dl[0][i, i])), _e(5, i, i)))
    return Site('RU_D', tag or ('Ru%d_d' % atom), [(atom, (0, 0, 0), element, [shell])], frame=frame,
                anchors=((atom, (0, 0, 0)), (top, top_img), (side, side_img)),
                anchor_elements=(element, 'ANY', 'ANY'), frame_ref=frame_ref, center=center,
                center_ref=center_ref, dmat=dl, spin='resolved' if resolved else 'shared', params=params,
                relax_validated=relax_validated, bottom=bottom)


def build_o_p_bond(o, partner, partner_img=(0, 0, 0), x_anchor=None, x_img=(0, 0, 0), vsigma=0.0, vpi=None,
                   vpi_in=None, vpi_out=None, lloc=None, frame='moving', zdir=None, xdir=None,
                   partner_element='ANY', tag=None, center='moving', center_ref=None, shell='L1',
                   relax_validated=False, frame_ref=None, xhelper=None, bottom=None):
    """O_P_BOND: one O (unique occupied l=1 PP_CHI 'L1' or '2P'); frame origin = axis partner P,
    z_from = the O itself, so z = unit(tau_O - tau_P) (bond_frame_field convention); x_from = x anchor
    (pi_in direction) or none (helper; only isotropic pi allowed).  L = diag(pi_in, pi_out, sigma)
    (+ optional full lloc 3x3), D_ao = R L R^T."""
    if lloc is None:
        if vpi is not None:
            if vpi_in is not None or vpi_out is not None:
                raise ValueError('give vpi (isotropic) or vpi_in/vpi_out')
            vpi_in = vpi_out = vpi
        vpi_in = 0.0 if vpi_in is None else vpi_in
        vpi_out = 0.0 if vpi_out is None else vpi_out
        lloc = np.diag([vpi_in, vpi_out, vsigma])
    lloc = np.asarray(lloc, float)
    if vpi is not None:
        params = [('vsigma', 0, float(lloc[2, 2]), _e(3, 2, 2)), ('vpi', 0, float(vpi), _e(3, 0, 0) + _e(3, 1, 1))]
    else:
        params = [('vsigma', 0, float(lloc[2, 2]), _e(3, 2, 2)), ('vpi_in', 0, float(lloc[0, 0]), _e(3, 0, 0)),
                  ('vpi_out', 0, float(lloc[1, 1]), _e(3, 1, 1))]
    fref = None
    if frame == 'frozen':
        if frame_ref is not None:
            if zdir is not None or xdir is not None:
                raise ValueError('frozen O_P_BOND: give frame_ref or zdir/xdir, not both')
            fref = np.asarray(frame_ref, float)
        elif zdir is None:
            raise ValueError('frozen O_P_BOND needs zdir')
        else:
            fref = frame_from_dirs(zdir, xdir)
    elif frame_ref is not None or zdir is not None or xdir is not None:
        raise ValueError('O_P_BOND: frame_ref / zdir / xdir only apply to frame frozen')
    anchors = ((partner, partner_img), (o, (0, 0, 0)), (x_anchor if x_anchor else 0, x_img))
    return Site('O_P_BOND', tag or ('O%d_p' % o), [(o, (0, 0, 0), 'O', [shell])], frame=frame, anchors=anchors,
                anchor_elements=(partner_element, 'O', 'ANY'), frame_ref=fref, center=center,
                center_ref=center_ref, dmat=[lloc.astype(complex)], params=params, relax_validated=relax_validated,
                xhelper=xhelper, bottom=bottom)


def bond_to_channel(theta, sigma, vb, va, h=0.0):
    """Bonding/antibonding (|b> = s|d> + sigma c|p>, |a> = c|d> - sigma s|p>, s = sin theta, OFE gauge:
    positive Cartesian orbitals) -> (eps_d, eps_p, t).  t is the D(d,p) element of ONE pi channel."""
    s, c = np.sin(theta), np.cos(theta)
    return vb * s * s + va * c * c, vb * c * c + va * s * s, sigma * s * c * (vb - va) + h


def channel_to_bond(theta, sigma, eps_d, eps_p, t):
    c2 = np.cos(2 * theta)
    if abs(c2) < 1e-6:
        raise ValueError('theta = 45 deg: V_b - V_a and h are not separable')
    dv = (eps_p - eps_d) / c2
    vsum = eps_d + eps_p
    return 0.5 * (vsum + dv), 0.5 * (vsum - dv), t - sigma * 0.5 * np.sin(2 * theta) * dv


# 0-based local AO indices of the 8-AO RU_O_PI group (d: z2 xz yz x2-y2 xy = 0..4; p: x y z = 5..7)
SIGMA2_INDEX = (0, 7)             # (d_z2, p_z)
FULL8_PAIRS = (('sigma', 0, 7), ('1', 1, 5), ('2', 2, 6))     # sigma pair, pi pair 1 (xz, p_x), pi pair 2 (yz, p_y)
FULL8_DELTA = (('x2y2', 3), ('xy', 4))


def _pi4_params_from_d(dl):
    """pi4 element parameters read from explicit D (extract_params=True): eps_d c, eps_p c, t c_re, t c_im
    (c = 1: xz/p_x, 2: yz/p_y), suffix _up/_dn when spin resolved."""
    out, res = [], len(dl) == 2
    for s, d in enumerate(dl):
        sp = (s + 1) if res else 0
        sfx = _spin_suffix(res, s)
        d = np.asarray(d, complex)
        for c in range(2):
            i, j = c, c + 2
            out += [('eps_d%d%s' % (c + 1, sfx), sp, float(d[i, i].real), _e(4, i, i)),
                    ('eps_p%d%s' % (c + 1, sfx), sp, float(d[j, j].real), _e(4, j, j)),
                    ('t%d_re%s' % (c + 1, sfx), sp, float(d[i, j].real), _e(4, i, j) + _e(4, j, i)),
                    ('t%d_im%s' % (c + 1, sfx), sp, float(d[i, j].imag), _e(4, i, j, 1j) + _e(4, j, i, -1j))]
    return out


def _spin_suffix(resolved, s):
    return '' if not resolved else ('_up' if s == 0 else '_dn')


def params_sigma2(dl):
    """sigma2 (nch = 2: d_z2, p_z): eps_d_sigma, eps_p_sigma, t_sigma_re, t_sigma_im (per spin if resolved)."""
    out, resolved = [], len(dl) == 2
    for s, d in enumerate(dl):
        sp = (s + 1) if resolved else 0
        sfx = _spin_suffix(resolved, s)
        d = np.asarray(d, complex)
        out += [('eps_d_sigma' + sfx, sp, float(d[0, 0].real), _e(2, 0, 0)),
                ('eps_p_sigma' + sfx, sp, float(d[1, 1].real), _e(2, 1, 1)),
                ('t_sigma_re' + sfx, sp, float(d[0, 1].real), _e(2, 0, 1) + _e(2, 1, 0)),
                ('t_sigma_im' + sfx, sp, float(d[0, 1].imag), _e(2, 0, 1, 1j) + _e(2, 1, 0, -1j))]
    return out


def params_full8(dl):
    """full8 block parameters (C = I8): sigma pair (z2, p_z): eps_d_sigma, eps_p_sigma, t_sigma_re/_im;
    pi pairs c = 1 (xz, p_x), 2 (yz, p_y): eps_d c, eps_p c, t c_re/_im (same names as pi4);
    delta diagonals eps_d_x2y2, eps_d_xy.  Values are read from D; the remaining elements of D (cross-symmetry
    couplings, p-p and d-d off-diagonals) stay in the unnamed rest."""
    out, resolved = [], len(dl) == 2
    for s, d in enumerate(dl):
        sp = (s + 1) if resolved else 0
        sfx = _spin_suffix(resolved, s)
        d = np.asarray(d, complex)
        for name, i, j in FULL8_PAIRS:
            if name == 'sigma':
                nd, np_, nt = 'eps_d_sigma', 'eps_p_sigma', 't_sigma'
            else:
                nd, np_, nt = 'eps_d' + name, 'eps_p' + name, 't' + name
            out += [(nd + sfx, sp, float(d[i, i].real), _e(8, i, i)),
                    (np_ + sfx, sp, float(d[j, j].real), _e(8, j, j)),
                    (nt + '_re' + sfx, sp, float(d[i, j].real), _e(8, i, j) + _e(8, j, i)),
                    (nt + '_im' + sfx, sp, float(d[i, j].imag), _e(8, i, j, 1j) + _e(8, j, i, -1j))]
        for name, i in FULL8_DELTA:
            out.append(('eps_d_' + name + sfx, sp, float(d[i, i].real), _e(8, i, i)))
    return out


def full8_from_blocks(eps_d_sigma=0.0, eps_p_sigma=0.0, t_sigma=0.0, eps_d=(0.0, 0.0), eps_p=(0.0, 0.0),
                      t=(0.0, 0.0), eps_delta=(0.0, 0.0)):
    """8x8 D (eV) from the block values (sigma pair, two pi pairs, delta diagonals)."""
    d = np.zeros((8, 8), complex)
    for (name, i, j), (ed, ep, tt) in zip(FULL8_PAIRS, ((eps_d_sigma, eps_p_sigma, t_sigma),
                                                       (eps_d[0], eps_p[0], t[0]), (eps_d[1], eps_p[1], t[1]))):
        d[i, i] += ed; d[j, j] += ep; d[i, j] += tt; d[j, i] += np.conj(tt)
    for (name, i), v in zip(FULL8_DELTA, eps_delta):
        d[i, i] += v
    return d


def build_ru_o_pi(ru, o, o_img=(0, 0, 0), side=0, side_img=(0, 0, 0), eps_d=(0.0, 0.0), eps_p=(0.0, 0.0),
                  t=(0.0, 0.0), d_pi4=None, d_pi4_dn=None, subspace='pi4', d_full8=None, frame='moving',
                  frame_ref=None, tag=None, center='moving', center_ref=None, shells=('L2', 'L1'),
                  relax_validated=False, eps_d_sigma=0.0, eps_p_sigma=0.0, t_sigma=0.0, d_sigma2=None,
                  d_sigma2_dn=None, d_full8_dn=None, eps_delta=(0.0, 0.0), extract_params=False, bottom=None):
    """RU_O_PI: joint group {Ru: l=2, O: l=1} (8 AOs, joint S-Lowdin), frame origin Ru, z_from the bonded O
    image, x_from a side O (always required by the engine).
      subspace 'pi4'   : channels (d_xz, d_yz, p_x, p_y); per pi channel c = 1 (xz, x), 2 (yz, y):
                         D = [[eps_d, t], [t*, eps_p]]; d_pi4 (4x4 Hermitian) overrides; V1.1 behaviour: an
                         explicit d_pi4 / d_pi4_dn clears the named parameters unless extract_params=True.
      subspace 'sigma2': (V1.2) channels (d_z2, p_z); D = [[eps_d_sigma, t_sigma], [t_sigma*, eps_p_sigma]] or
                         explicit d_sigma2 (2x2) [+ d_sigma2_dn]; named parameters eps_d_sigma, eps_p_sigma,
                         t_sigma_re, t_sigma_im (per spin _up/_dn when resolved), values read from D.
      subspace 'full8' : C = I8; d_full8 (8x8) [+ d_full8_dn] or the block values (eps_d_sigma, eps_p_sigma,
                         t_sigma, eps_d, eps_p, t, eps_delta); (V1.2) block named parameters (params_full8).
    Conjugates: dE/d eps_d = N_dd, dE/d eps_p = N_pp, dE/dRe t = 2 Re N_dp, dE/dIm t = 2 Im N_dp."""
    def _nz(x):
        return np.any(np.abs(np.asarray(x, complex)) > 0)
    sig_given = _nz([eps_d_sigma, eps_p_sigma, t_sigma])
    pi_given = _nz(list(eps_d) + list(eps_p) + list(t))
    if subspace == 'pi4':
        if sig_given or d_sigma2 is not None or d_sigma2_dn is not None or d_full8 is not None or \
                d_full8_dn is not None or _nz(eps_delta):
            raise ValueError("subspace 'pi4': sigma / full8 / delta inputs do not apply")
        cl = np.zeros((8, 4))
        for j, i in enumerate(PI4_INDEX):
            cl[i, j] = 1.0
        params = []
        dd = np.zeros((4, 4), complex)
        for c in range(2):
            dd[c, c] += eps_d[c]; dd[c + 2, c + 2] += eps_p[c]
            dd[c, c + 2] += t[c]; dd[c + 2, c] += np.conj(t[c])
            params += [('eps_d%d' % (c + 1), 0, float(np.real(eps_d[c])), _e(4, c, c)),
                       ('eps_p%d' % (c + 1), 0, float(np.real(eps_p[c])), _e(4, c + 2, c + 2)),
                       ('t%d_re' % (c + 1), 0, float(np.real(t[c])), _e(4, c, c + 2) + _e(4, c + 2, c)),
                       ('t%d_im' % (c + 1), 0, float(np.imag(t[c])), _e(4, c, c + 2, 1j) + _e(4, c + 2, c, -1j))]
        if d_pi4 is not None:
            if pi_given:
                raise ValueError("give d_pi4 or eps_d/eps_p/t, not both")
            dd = np.asarray(d_pi4, complex); params = []
        dl = [dd]
        if d_pi4_dn is not None:
            if d_pi4 is None:
                raise ValueError('d_pi4_dn needs d_pi4 (spin up)')
            dl.append(np.asarray(d_pi4_dn, complex)); params = []
        if (d_pi4 is not None) and extract_params:
            params = _pi4_params_from_d(dl)
    elif subspace == 'sigma2':
        if pi_given or d_pi4 is not None or d_pi4_dn is not None or d_full8 is not None or d_full8_dn is not None \
                or _nz(eps_delta):
            raise ValueError("subspace 'sigma2': pi4 / full8 / delta inputs do not apply")
        cl = np.zeros((8, 2))
        for j, i in enumerate(SIGMA2_INDEX):
            cl[i, j] = 1.0
        if d_sigma2 is not None:
            if sig_given:
                raise ValueError('give d_sigma2 or eps_d_sigma/eps_p_sigma/t_sigma, not both')
            dl = [np.asarray(d_sigma2, complex)]
        else:
            if d_sigma2_dn is not None:
                raise ValueError('d_sigma2_dn needs d_sigma2 (spin up)')
            dl = [np.array([[eps_d_sigma, t_sigma], [np.conj(t_sigma), eps_p_sigma]], complex)]
        if d_sigma2_dn is not None:
            dl.append(np.asarray(d_sigma2_dn, complex))
        params = params_sigma2(dl)
    elif subspace == 'full8':
        if d_pi4 is not None or d_pi4_dn is not None or d_sigma2 is not None or d_sigma2_dn is not None:
            raise ValueError("subspace 'full8': d_pi4 / d_sigma2 do not apply (use d_full8 or block values)")
        cl = np.eye(8)
        blocks = sig_given or pi_given or _nz(eps_delta)
        if d_full8 is not None:
            if blocks:
                raise ValueError('give d_full8 or the full8 block values, not both')
            dl = [np.asarray(d_full8, complex)]
        else:
            if d_full8_dn is not None:
                raise ValueError('d_full8_dn needs d_full8 (spin up)')
            dl = [full8_from_blocks(eps_d_sigma, eps_p_sigma, t_sigma, eps_d, eps_p, t, eps_delta)]
        if d_full8_dn is not None:
            dl.append(np.asarray(d_full8_dn, complex))
        params = params_full8(dl)
    else:
        raise ValueError('subspace pi4 | sigma2 | full8')
    if side == 0:
        raise ValueError('RU_O_PI needs a side O (x anchor)')
    return Site('RU_O_PI', tag or ('Ru%d_O%d_pi' % (ru, o)), [(ru, (0, 0, 0), 'Ru', [shells[0]]),
                                                              (o, o_img, 'O', [shells[1]])],
                frame=frame, anchors=((ru, (0, 0, 0)), (o, o_img), (side, side_img)),
                anchor_elements=('Ru', 'O', 'ANY'), frame_ref=frame_ref, center=center, center_ref=center_ref,
                cloc=cl, dmat=dl, spin='resolved' if len(dl) == 2 else 'shared', params=params,
                relax_validated=relax_validated, bottom=bottom)


def pair_dmat(r, v_pop=None, v_pol=None, v_coh=None):
    """Paired channels [u_1..u_r, v_1..v_r]: D = sum V_pop(E_uu+E_vv)/2 + V_pol(E_uu-E_vv)/2 + V_coh(E_uv+E_vu)/2."""
    z = np.zeros(r)
    v_pop = z if v_pop is None else np.asarray(v_pop, float)
    v_pol = z if v_pol is None else np.asarray(v_pol, float)
    v_coh = z if v_coh is None else np.asarray(v_coh, float)
    d = np.zeros((2 * r, 2 * r))
    for i in range(r):
        d[i, i] += 0.5 * (v_pop[i] + v_pol[i]); d[r + i, r + i] += 0.5 * (v_pop[i] - v_pol[i])
        d[i, r + i] += 0.5 * v_coh[i]; d[r + i, i] += 0.5 * v_coh[i]
    return d


def pair_params(r, v_pop=None, v_pol=None, v_coh=None):
    out = []
    n = 2 * r
    for i in range(r):
        for name, b, vals in (('v_pop', _e(n, i, i, .5) + _e(n, r + i, r + i, .5), v_pop),
                              ('v_pol', _e(n, i, i, .5) - _e(n, r + i, r + i, .5), v_pol),
                              ('v_coh', _e(n, i, r + i, .5) + _e(n, r + i, i, .5), v_coh)):
            out.append(('%s%d' % (name, i + 1), 0, 0.0 if vals is None else float(vals[i]), b))
    return out


def build_iao_channel(atoms, cloc, frame='moving', anchors=None, dmat=None, tag='IAO_channel', params=None,
                      center='moving', center_ref=None, frame_ref=None, spin=None, relax_validated=False,
                      xhelper=None, anchor_elements=None, bottom=None):
    """IAO_CHANNEL: general group (atoms = [(atom, img, element, [labels])]), frozen channel C (local
    frame for 'moving'/'frozen', lab frame for 'global'; None = C_IDENTITY, V1.2), Hermitian D per spin."""
    cl = None if cloc is None else np.asarray(cloc, float)
    if cl is None:
        nch = sum(2 * shell_l(s) + 1 for _, _, _, sh in atoms for s in sh)
    else:
        nch = cl.shape[1]
    if dmat is None:
        dmat = [np.zeros((nch, nch))]
    if spin is None:
        spin = 'resolved' if len(dmat) == 2 else 'shared'
    return Site('IAO_CHANNEL', tag, atoms, frame=frame, anchors=anchors, anchor_elements=anchor_elements,
                frame_ref=frame_ref, center=center, center_ref=center_ref, cloc=cl, dmat=dmat, spin=spin,
                params=params, relax_validated=relax_validated, xhelper=xhelper, bottom=bottom)


def params_hermitian_basis(n, labels, dl, prefix_diag='eps_', prefix_off='t_'):
    """Complete real basis of n x n Hermitian matrices as named parameters (values read from D):
    eps_<a> (E_aa), t_<a><b>_re (E_ab + E_ba), t_<a><b>_im (i E_ab - i E_ba).  Per spin _up/_dn if resolved."""
    out, resolved = [], len(dl) == 2
    for s, d in enumerate(dl):
        sp = (s + 1) if resolved else 0
        sfx = _spin_suffix(resolved, s)
        d = np.asarray(d, complex)
        for i in range(n):
            out.append((prefix_diag + labels[i] + sfx, sp, float(d[i, i].real), _e(n, i, i)))
        for i in range(n):
            for j in range(i + 1, n):
                nm = prefix_off + labels[i] + labels[j]
                out.append((nm + '_re' + sfx, sp, float(d[i, j].real), _e(n, i, j) + _e(n, j, i)))
                out.append((nm + '_im' + sfx, sp, float(d[i, j].imag), _e(n, i, j, 1j) + _e(n, j, i, -1j)))
    return out


SIGMA5S_LABELS = ('s', 'z', 'p')     # channel 1 = Ru 5s, 2 = Ru d_z2, 3 = O p_z (local z = Ru -> O)


def build_sigma5s(ru, o, o_img=(0, 0, 0), side=0, side_img=(0, 0, 0), d=None, d_dn=None, eps=None, t=None,
                  with_o2s=False, frame='moving', frame_ref=None, center='moving', center_ref=None, tag=None,
                  ru_shells=('5S', '4D'), o_shell='2P', o2s_shell='2S', relax_validated=False, xhelper=None,
                  bottom=None):
    """(V1.2) sigma + 5s channel on IAO_CHANNEL: group {Ru: 5S 4D; O: [2S (Lowdin only)] 2P}, local frame
    origin Ru, z_from the bonded O image, x_from = side O or 0 (helper axis: allowed, the three channels are
    axially symmetric about z, so every D on them passes the engine's axial check).
    C columns (local frame): Ru 5s, Ru d_z2, O p_z  (gerade sigma of the ledger: e_g(z2) + 5s, with O p_z).
    D: 3x3 Hermitian (eV) d [+ d_dn], or eps = (eps_s, eps_z, eps_p) and t = dict(sz=, sp=, zp=) (complex ok).
    Named parameters (values read from D): eps_s, eps_z, eps_p, t_sz_re/_im, t_sp_re/_im, t_zp_re/_im
    (z = d_z2, p = O p_z; t_zp is the d_z2-p_z sigma hopping).  Conjugates as for RU_O_PI."""
    o_sh = ([o2s_shell] if with_o2s else []) + [o_shell]
    atoms = [(ru, (0, 0, 0), 'Ru', list(ru_shells)), (o, o_img, 'O', o_sh)]
    nao = sum(2 * shell_l(s) + 1 for _, _, _, sh in atoms for s in sh)
    if [shell_l(s) for s in ru_shells] != [0, 2] or shell_l(o_shell) != 1:
        raise ValueError('sigma5s: Ru shells must be (s, d) and the O shell p')
    cl = np.zeros((nao, 3))
    cl[0, 0] = 1.0                    # Ru 5s
    cl[1, 1] = 1.0                    # Ru d_z2 (first d component)
    cl[nao - 1, 2] = 1.0              # O p_z (last p component)
    if d is not None:
        if eps is not None or t is not None:
            raise ValueError('sigma5s: give d or eps/t, not both')
        dl = [np.asarray(d, complex)]
    else:
        if d_dn is not None:
            raise ValueError('sigma5s: d_dn needs d')
        e3 = (0.0, 0.0, 0.0) if eps is None else eps
        tt = {} if t is None else dict(t)
        bad = set(tt) - {'sz', 'sp', 'zp'}
        if bad:
            raise ValueError('sigma5s: t keys must be sz, sp, zp (got %s)' % sorted(bad))
        m = np.diag(np.asarray(e3, complex))
        for key, (i, j) in (('sz', (0, 1)), ('sp', (0, 2)), ('zp', (1, 2))):
            v = complex(*tt[key]) if isinstance(tt.get(key), (list, tuple)) else complex(tt.get(key, 0.0))
            m[i, j] += v; m[j, i] += np.conj(v)
        dl = [m]
    if d_dn is not None:
        dl.append(np.asarray(d_dn, complex))
    params = params_hermitian_basis(3, SIGMA5S_LABELS, dl)
    if frame == 'global':
        raise ValueError('sigma5s needs a local frame (moving or frozen): z must be the Ru -> O axis')
    return Site('IAO_CHANNEL', tag or ('Ru%d_O%d_sigma5s' % (ru, o)), atoms, frame=frame,
                anchors=((ru, (0, 0, 0)), (o, o_img), (side, side_img)), anchor_elements=('Ru', 'O', 'ANY'),
                frame_ref=frame_ref, xhelper=xhelper, center=center, center_ref=center_ref, cloc=cl, dmat=dl,
                spin='resolved' if len(dl) == 2 else 'shared', params=params, relax_validated=relax_validated,
                bottom=bottom)


def build_observer(atoms, tag='observer', frame='global', anchors=None, frame_ref=None, bottom=None):
    """(V1.2) zero-field observation site: IAO_CHANNEL, C = I (C_IDENTITY), D = 0.  The engine computes and
    prints N / N_AO (and H_ch when print_hch) for it while h_psi returns immediately (no explicit force:
    force splits are only evaluated for a nonzero D).  frame 'global' (lab frame) needs no anchors."""
    return build_iao_channel(atoms, None, frame=frame, anchors=anchors, dmat=None, tag=tag, frame_ref=frame_ref,
                             bottom=bottom)


# --------------------------------------------------------------------- canonical ofe.in
CONTROL_DEFAULTS = dict(model='OFE_V1', control='potential', allow_relax=False, print_nao=True, dmax_eV=3.0,
                        v_hard_eV=10.0, anchor_max_A=3.2, group_radius_A=3.2, fprint_min=1e-6,
                        history='ofe_history.dat')
# V1.2 engine switches: written to &ofe_control ONLY when given (a V1.1 engine stops on an unknown namelist
# variable, so an old binary can never silently ignore a requested H_ch print).
OPTIONAL_CONTROL = dict(print_hch=False, print_hch_k=False)
CONTROL_KEYS = set(CONTROL_DEFAULTS) | set(OPTIONAL_CONTROL) | {'nsite'}
BOTTOM_O_DEFAULT = (93, 52)       # Hess RuO2(100) Ru36O78H2 bottom O of Ru1 / Ru6 (project numbering)
BOTTOM_GATE_A = 3.6               # anchor_max_A / group_radius_A default for bottom-bond sites (review P1-4)


def _fmt(x):
    return '%.16e' % float(x)


def _fval(v):
    if isinstance(v, bool):
        return '.true.' if v else '.false.'
    if isinstance(v, str):
        return "'%s'" % v
    if isinstance(v, int):
        return str(v)
    return repr(float(v))


def site_atoms_all(s):
    """Every atom index a site touches (group atoms and frame anchors)."""
    out = [a for a, _, _, _ in s.atoms]
    if s.anchors is not None:
        out += [a for a, _ in s.anchors if a != 0]
    return out


def is_bottom_site(s, bottom_o_atoms=BOTTOM_O_DEFAULT):
    if getattr(s, 'bottom', None) is not None:
        return bool(s.bottom)
    return any(a in tuple(bottom_o_atoms) for a in site_atoms_all(s))


def resolve_control(sites, control=None, bottom_o_atoms=BOTTOM_O_DEFAULT):
    """-> (ctl dict for the namelist, list of note strings).  Applies the V1.2 rules:
    unknown keys -> error; nsite must match; print_hch_k needs print_hch; bottom-bond sites (site.bottom True, or
    any group/anchor atom in bottom_o_atoms) raise anchor_max_A and group_radius_A to 3.6 A unless the user set
    them (a user value < 3.6 is kept and flagged)."""
    user = dict(control or {})
    bad = sorted(set(user) - CONTROL_KEYS)
    if bad:
        raise ValueError('unknown &ofe_control key(s): %s (allowed: %s)' % (bad, sorted(CONTROL_KEYS)))
    if 'nsite' in user and int(user['nsite']) != len(sites):
        raise ValueError('control nsite = %s but %d site(s) given' % (user['nsite'], len(sites)))
    for k in ('print_hch', 'print_hch_k', 'allow_relax', 'print_nao'):
        if k in user and not isinstance(user[k], bool):
            raise ValueError('control %s must be a boolean' % k)
    if user.get('print_hch_k') and not user.get('print_hch'):
        raise ValueError('print_hch_k = .true. needs print_hch = .true. (per-k records are written in addition to '
                         'the k-averaged OFE_HCH block)')
    ctl = dict(CONTROL_DEFAULTS)
    ctl.update({k: v for k, v in user.items() if k != 'nsite'})
    notes = []
    bsites = [(i, s.tag) for i, s in enumerate(sites, 1) if is_bottom_site(s, bottom_o_atoms)]
    if bsites:
        lab = ', '.join('%d:%s' % b for b in bsites)
        for k in ('anchor_max_A', 'group_radius_A'):
            if k not in user:
                ctl[k] = max(float(CONTROL_DEFAULTS[k]), BOTTOM_GATE_A)
                notes.append('V1.2 bottom-bond default: %s = %.2f A (V1.1 default %.2f) because of site(s) %s'
                             % (k, ctl[k], CONTROL_DEFAULTS[k], lab))
            elif float(user[k]) < BOTTOM_GATE_A:
                notes.append('V1.2 WARNING: %s = %s A set by the user is below the bottom-bond value %.2f A '
                             '(site(s) %s); exit-state b ~ 3.15-3.19 A may hit the engine gate' %
                             (k, user[k], BOTTOM_GATE_A, lab))
    return ctl, notes


def write_ofe_in(sites, control=None, comment=None, bottom_o_atoms=BOTTOM_O_DEFAULT, notes_out=None):
    ctl, notes = resolve_control(sites, control, bottom_o_atoms)
    ctl['nsite'] = len(sites)
    user = dict(control or {})
    lines = []
    keys = ['model', 'control', 'nsite', 'allow_relax', 'print_nao', 'dmax_eV', 'v_hard_eV', 'anchor_max_A',
            'group_radius_A', 'fprint_min', 'history']
    keys += [k for k in ('print_hch', 'print_hch_k') if k in user]
    lines.append('&ofe_control')
    for k in keys:
        lines.append('  %s = %s,' % (k, _fval(ctl[k])))
    lines.append('/')
    if comment:
        for c in comment.splitlines():
            lines.append('! ' + c)
    for n in notes:
        lines.append('! ' + n)
    if notes_out is not None:
        notes_out.extend(notes)
    for i, s in enumerate(sites, 1):
        lines.append('SITE %d' % i)
        lines.append('  KIND %s' % s.kind)
        lines.append('  TAG %s' % s.tag)
        for a, img, el, sh in s.atoms:
            lines.append('  ATOM %d %d %d %d %s %d %s' % (a, img[0], img[1], img[2], el, len(sh), ' '.join(sh)))
        lines.append('  FRAME %s' % s.frame)
        if s.anchors is not None:
            lines.append('  ANCHORS ' + '  '.join('%d %d %d %d' % (a, i1[0], i1[1], i1[2]) for a, i1 in s.anchors))
            if s.anchor_elements is not None:
                lines.append('  ANCHOR_ELEMENTS %s %s %s' % tuple(s.anchor_elements))
        if s.frame_ref is not None:
            lines.append('  FRAME_REF ' + ' '.join(_fmt(v) for v in s.frame_ref.T.ravel()))
        if s.xhelper is not None:
            lines.append('  XHELPER ' + ' '.join(_fmt(v) for v in s.xhelper))
        lines.append('  CENTER %s' % s.center)
        for g, p in sorted(s.center_ref.items()):
            lines.append('  CENTER_REF %d %s' % (g, ' '.join(_fmt(v) for v in p)))
        lines.append('  NCH %d' % s.nch)
        if s.cidentity:
            lines.append('  C_IDENTITY')
        else:
            lines.append('  C_LOCAL')
            for row in s.cloc:
                lines.append('   ' + ' '.join(_fmt(v) for v in row))
        lines.append('  SPIN %s' % s.spin)
        nsp = 1 if s.spin == 'shared' else 2
        for sp in range(nsp):
            lines.append('  D_EV %d' % (sp + 1))
            for row in s.dmat[sp]:
                lines.append('   ' + ' '.join(_fmt(v) for v in np.real(row)) + '  ' +
                             ' '.join(_fmt(v) for v in np.imag(row)))
        for name, sp, val, b in s.params:
            b = np.asarray(b, complex)
            nz = [(i, j) for i in range(s.nch) for j in range(s.nch) if abs(b[i, j]) > 0]
            lines.append('  PARAM %s %d %s %d' % (name, sp, _fmt(val), len(nz)))
            for i, j in nz:
                lines.append('   %d %d %s %s' % (i + 1, j + 1, _fmt(b[i, j].real), _fmt(b[i, j].imag)))
        if s.relax_validated:
            lines.append('  RELAX_VALIDATED T')
        lines.append('END_SITE')
    return '\n'.join(lines) + '\n'


def _strip(line):
    for ch in '!#':
        k = line.find(ch)
        if k >= 0:
            line = line[:k]
    return line.replace('\t', ' ').strip()


def parse_namelist_simple(text):
    """Minimal parser for the &ofe_control namelist (scalars only)."""
    out = {}
    body = text.split('&', 1)[1]
    body = body[body.index('\n'):] if '\n' in body else body
    body = body.split('/', 1)[0]
    for part in body.replace('\n', ',').split(','):
        part = _strip(part)
        if '=' not in part:
            continue
        k, v = [x.strip() for x in part.split('=', 1)]
        vl = v.lower()
        if vl in ('.true.', 't', '.t.'):
            out[k] = True
        elif vl in ('.false.', 'f', '.f.'):
            out[k] = False
        elif v.startswith("'") or v.startswith('"'):
            out[k] = v.strip("'\"")
        else:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = float(v.replace('d', 'e').replace('D', 'e'))
    return out


def read_ofe_in(text):
    """Parse canonical ofe.in text -> (control dict, [Site]).  Mirrors the Fortran parser."""
    i0 = text.index('&')
    j0 = text.index('/', i0)
    ctl = dict(CONTROL_DEFAULTS)
    ctl.update(parse_namelist_simple(text[i0:j0 + 1]))
    lines = text[j0 + 1:].splitlines()
    sites, cur, k = [], None, 0
    while k < len(lines):
        ln = _strip(lines[k]); k += 1
        if not ln:
            continue
        key, _, rest = ln.partition(' ')
        key = key.upper(); rest = rest.strip(); tok = rest.split()
        if key == 'SITE':
            cur = dict(atoms=[], frame='moving', anchors=None, anchor_elements=None, frame_ref=None, xhelper=None,
                       center='moving', center_ref={}, nch=None, cloc=None, spin='shared', dmat={}, params=[],
                       relax=False, kind=None, tag='')
        elif key == 'END_SITE':
            nsp = 1 if cur['spin'] == 'shared' else 2
            nch = cur['nch']
            dm = [cur['dmat'].get(sp, np.zeros((nch, nch), complex)) for sp in range(1, nsp + 1)]
            sites.append(Site(cur['kind'], cur['tag'], cur['atoms'], frame=cur['frame'], anchors=cur['anchors'],
                              anchor_elements=cur['anchor_elements'], frame_ref=cur['frame_ref'],
                              xhelper=cur['xhelper'], center=cur['center'], center_ref=cur['center_ref'],
                              cloc=cur['cloc'], dmat=dm, spin=cur['spin'], params=cur['params'],
                              relax_validated=cur['relax']))
            cur = None
        elif key == 'KIND':
            cur['kind'] = tok[0].upper()
        elif key == 'TAG':
            cur['tag'] = tok[0]
        elif key == 'ATOM':
            ns = int(tok[5])
            cur['atoms'].append((int(tok[0]), tuple(int(v) for v in tok[1:4]), tok[4], tok[6:6 + ns]))
        elif key == 'FRAME':
            cur['frame'] = tok[0].lower()
        elif key == 'ANCHORS':
            v = [int(x) for x in tok]
            cur['anchors'] = tuple((v[4 * i], tuple(v[4 * i + 1:4 * i + 4])) for i in range(3))
        elif key == 'ANCHOR_ELEMENTS':
            cur['anchor_elements'] = tuple(tok[:3])
        elif key == 'FRAME_REF':
            cur['frame_ref'] = np.array([float(x) for x in tok]).reshape(3, 3).T
        elif key == 'XHELPER':
            cur['xhelper'] = np.array([float(x) for x in tok])
        elif key == 'CENTER':
            cur['center'] = tok[0].lower()
        elif key == 'CENTER_REF':
            cur['center_ref'][int(tok[0])] = np.array([float(x) for x in tok[1:4]])
        elif key == 'NCH':
            cur['nch'] = int(tok[0])
        elif key == 'C_IDENTITY':
            cur['cloc'] = None
        elif key == 'C_LOCAL':
            nao = sum(2 * shell_l(s) + 1 for _, _, _, sh in cur['atoms'] for s in sh)
            rows = []
            for _ in range(nao):
                rows.append([float(x) for x in lines[k].split()]); k += 1
            cur['cloc'] = np.array(rows)
        elif key == 'SPIN':
            cur['spin'] = tok[0].lower()
        elif key == 'D_EV':
            sp = int(tok[0]); nch = cur['nch']; rows = []
            for _ in range(nch):
                v = [float(x) for x in lines[k].split()]; k += 1
                rows.append(np.array(v[:nch]) + 1j * np.array(v[nch:2 * nch]))
            cur['dmat'][sp] = np.array(rows)
        elif key == 'PARAM':
            name, sp, val, nnz = tok[0], int(tok[1]), float(tok[2]), int(tok[3])
            b = np.zeros((cur['nch'], cur['nch']), complex)
            for _ in range(nnz):
                v = lines[k].split(); k += 1
                b[int(v[0]) - 1, int(v[1]) - 1] = float(v[2]) + 1j * float(v[3])
            cur['params'].append((name, sp, val, b))
        elif key == 'RELAX_VALIDATED':
            cur['relax'] = tok[0].upper() in ('T', '.TRUE.', 'TRUE')
        else:
            raise ValueError('unknown keyword %s' % key)
    if len(sites) != ctl['nsite']:
        raise ValueError('nsite mismatch')
    return ctl, sites


def set_param(site, name, value):
    """Change a named parameter and rebuild D accordingly (keeps all other D content)."""
    for idx, (n, sp, val, b) in enumerate(site.params):
        if n == name:
            b = np.asarray(b, complex)
            for s in range(2):
                if sp in (0, s + 1):
                    site.dmat[s] = site.dmat[s] + (value - val) * b
            site.params[idx] = (n, sp, value, b)
            return site
    raise KeyError(name)




# --------------------------------------------------------------------- JSON front end (V1.2: strict keys)
COMMON_KEYS = {'kind', 'tag', 'bottom', 'relax_validated'}
KIND_KEYS = {
    'RU_D': {'atom', 'top', 'side', 'top_img', 'side_img', 'v_eV', 'dloc_eV', 'v_dn_eV', 'dloc_dn_eV', 'frame',
             'frame_ref', 'element', 'center', 'center_ref', 'shell'},
    'O_P_BOND': {'o', 'partner', 'partner_img', 'x_anchor', 'x_img', 'vsigma_eV', 'vpi_eV', 'vpi_in_eV', 'vpi_out_eV',
                 'lloc_eV', 'frame', 'zdir', 'xdir', 'frame_ref', 'partner_element', 'center', 'center_ref', 'shell',
                 'xhelper'},
    'RU_O_PI': {'ru', 'o', 'o_img', 'side', 'side_img', 'subspace', 'eps_d_eV', 'eps_p_eV', 't_eV',
                'd_pi4_eV', 'd_pi4_im_eV', 'd_pi4_dn_eV', 'd_pi4_dn_im_eV',
                'eps_d_sigma_eV', 'eps_p_sigma_eV', 't_sigma_eV', 'd_sigma2_eV', 'd_sigma2_im_eV', 'd_sigma2_dn_eV',
                'd_sigma2_dn_im_eV', 'd_full8_eV', 'd_full8_im_eV', 'd_full8_dn_eV', 'd_full8_dn_im_eV',
                'eps_delta_eV', 'bond', 'frame', 'frame_ref', 'center', 'center_ref', 'shells', 'extract_params'},
    'IAO_CHANNEL': {'atoms', 'cloc', 'frame', 'anchors', 'anchor_elements', 'frame_ref', 'center', 'center_ref',
                    'spin', 'd_eV', 'd_im_eV', 'pair', 'params', 'xhelper'},
    'SIGMA_5S': {'ru', 'o', 'o_img', 'side', 'side_img', 'd_eV', 'd_im_eV', 'd_dn_eV', 'd_dn_im_eV', 'eps_eV', 't_eV',
                 'with_o2s', 'frame', 'frame_ref', 'center', 'center_ref', 'ru_shells', 'o_shell', 'o2s_shell',
                 'xhelper'},
    'OBSERVER': {'atoms', 'frame', 'anchors', 'frame_ref'},
}
KEY_ALIASES = {'d_full8': 'd_full8_eV', 'd_full8_dn': 'd_full8_dn_eV', 'd_pi4': 'd_pi4_eV', 'd_pi4_dn': 'd_pi4_dn_eV'}
TOP_KEYS = {'control', 'sites', 'comment', 'bottom_o_atoms'}
ATOM_KEYS = {'atom', 'img', 'element', 'shells'}
PAIR_KEYS = {'rank', 'v_pop_eV', 'v_pol_eV', 'v_coh_eV'}
PARAM_KEYS = {'name', 'spin', 'value_eV', 'basis'}
BOND_KEYS = {'theta_deg', 'sigma', 'vb_eV', 'va_eV', 'h_eV'}


def _check_keys(d, allowed, where):
    """Every key must be known; keys starting with '_' are comments.  Raises instead of silently ignoring."""
    bad = sorted(k for k in d if not str(k).startswith('_') and k not in allowed)
    if bad:
        raise ValueError('%s: unknown or inapplicable key(s) %s; allowed: %s' % (where, bad, sorted(allowed)))


def _normalise_aliases(d, where):
    d = dict(d)
    for a, k in KEY_ALIASES.items():
        if a in d:
            if k in d:
                raise ValueError('%s: give %s or %s, not both' % (where, a, k))
            d[k] = d.pop(a)
    return d


def _cmat(d, key):
    """Complex matrix from key (real part) and key without '_eV' + '_im_eV' (imaginary part)."""
    if key not in d:
        if key.replace('_eV', '_im_eV') in d:
            raise ValueError('%s given without %s' % (key.replace('_eV', '_im_eV'), key))
        return None
    re_ = np.asarray(d[key], float)
    ik = key.replace('_eV', '_im_eV')
    im_ = np.zeros_like(re_) if ik not in d else np.asarray(d[ik], float)
    if im_.shape != re_.shape:
        raise ValueError('%s and %s shapes differ' % (key, ik))
    return re_ + 1j * im_


def _cnum(x):
    return complex(*x) if isinstance(x, (list, tuple)) else complex(x)


def _img(x):
    return tuple(int(v) for v in x)


def _json_bond(d, n_expected, where):
    bl = d['bond']
    if len(bl) != n_expected:
        raise ValueError('%s: bond needs %d entr%s for this subspace' % (where, n_expected, 'y' if n_expected == 1 else 'ies'))
    ed, ep, tt = [], [], []
    for c in bl:
        _check_keys(c, BOND_KEYS, where + ' bond entry')
        for k in ('theta_deg', 'sigma', 'vb_eV', 'va_eV'):
            if k not in c:
                raise ValueError('%s bond entry needs %s' % (where, k))
        if c['sigma'] not in (1, -1):
            raise ValueError('%s bond sigma must be +1 or -1' % where)
        a, b, h = bond_to_channel(np.radians(c['theta_deg']), c['sigma'], c['vb_eV'], c['va_eV'], c.get('h_eV', 0))
        ed.append(a); ep.append(b); tt.append(h)
    return ed, ep, tt


def site_from_json(d, index=None):
    where = 'site %s' % (index if index is not None else d.get('tag', '?'))
    if 'kind' not in d:
        raise ValueError('%s: missing kind' % where)
    kind = d['kind'].upper()
    if kind not in KIND_KEYS:
        raise ValueError('%s: kind must be one of %s' % (where, sorted(KIND_KEYS)))
    d = _normalise_aliases(d, where)
    _check_keys(d, KIND_KEYS[kind] | COMMON_KEYS, '%s (%s)' % (where, kind))
    common = dict(relax_validated=bool(d.get('relax_validated', False)), bottom=d.get('bottom'))
    if kind == 'RU_D':
        return build_ru_d(d['atom'], d['top'], d['side'], d.get('top_img', (0, 0, 0)), d.get('side_img', (0, 0, 0)),
                          v=d.get('v_eV'), dloc=d.get('dloc_eV'), v_dn=d.get('v_dn_eV'), dloc_dn=d.get('dloc_dn_eV'),
                          frame=d.get('frame', 'moving'), frame_ref=d.get('frame_ref'), element=d.get('element', 'Ru'),
                          tag=d.get('tag'), center=d.get('center', 'moving'), center_ref=d.get('center_ref'),
                          shell=d.get('shell', 'L2'), **common)
    if kind == 'O_P_BOND':
        return build_o_p_bond(d['o'], d['partner'], d.get('partner_img', (0, 0, 0)), d.get('x_anchor'),
                              d.get('x_img', (0, 0, 0)), d.get('vsigma_eV', 0.0), d.get('vpi_eV'), d.get('vpi_in_eV'),
                              d.get('vpi_out_eV'), d.get('lloc_eV'), d.get('frame', 'moving'), d.get('zdir'),
                              d.get('xdir'), d.get('partner_element', 'ANY'), d.get('tag'), d.get('center', 'moving'),
                              d.get('center_ref'), d.get('shell', 'L1'), frame_ref=d.get('frame_ref'),
                              xhelper=d.get('xhelper'), **common)
    if kind == 'RU_O_PI':
        sub = d.get('subspace', 'pi4')
        geo = dict(frame=d.get('frame', 'moving'), frame_ref=d.get('frame_ref'), tag=d.get('tag'),
                   center=d.get('center', 'moving'), center_ref=d.get('center_ref'),
                   shells=tuple(d.get('shells', ('L2', 'L1'))), **common)
        pos = (d['ru'], d['o'], _img(d.get('o_img', (0, 0, 0))), d['side'], _img(d.get('side_img', (0, 0, 0))))
        if 'bond' in d:      # bonding/antibonding convenience input (theta, sigma, V_b, V_a, h) per channel
            others = sorted(k for k in d if k.startswith(('eps_', 't_', 'd_')) or k == 'extract_params')
            if others:
                raise ValueError('%s: bond input cannot be combined with %s' % (where, others))
            if sub == 'pi4':
                ed, ep, tt = _json_bond(d, 2, where)
                return build_ru_o_pi(*pos, eps_d=ed, eps_p=ep, t=tt, subspace='pi4', **geo)
            if sub == 'sigma2':
                ed, ep, tt = _json_bond(d, 1, where)
                return build_ru_o_pi(*pos, subspace='sigma2', eps_d_sigma=ed[0], eps_p_sigma=ep[0], t_sigma=tt[0],
                                     **geo)
            raise ValueError("%s: bond input supports subspace 'pi4' or 'sigma2'" % where)
        if sub == 'pi4':
            allowed = {'eps_d_eV', 'eps_p_eV', 't_eV', 'd_pi4_eV', 'd_pi4_im_eV', 'd_pi4_dn_eV', 'd_pi4_dn_im_eV',
                       'extract_params'}
        elif sub == 'sigma2':
            allowed = {'eps_d_sigma_eV', 'eps_p_sigma_eV', 't_sigma_eV', 'd_sigma2_eV', 'd_sigma2_im_eV',
                       'd_sigma2_dn_eV', 'd_sigma2_dn_im_eV'}
        elif sub == 'full8':
            allowed = {'d_full8_eV', 'd_full8_im_eV', 'd_full8_dn_eV', 'd_full8_dn_im_eV', 'eps_d_sigma_eV',
                       'eps_p_sigma_eV', 't_sigma_eV', 'eps_d_eV', 'eps_p_eV', 't_eV', 'eps_delta_eV'}
        else:
            raise ValueError("%s: subspace must be 'pi4', 'sigma2' or 'full8'" % where)
        field_keys = KIND_KEYS['RU_O_PI'] - {'ru', 'o', 'o_img', 'side', 'side_img', 'subspace', 'bond', 'frame',
                                            'frame_ref', 'center', 'center_ref', 'shells'}
        wrong = sorted(k for k in d if k in field_keys and k not in allowed)
        if wrong:
            raise ValueError("%s: key(s) %s do not apply to subspace '%s'" % (where, wrong, sub))
        t = [_cnum(x) for x in d.get('t_eV', (0, 0))]
        return build_ru_o_pi(*pos, eps_d=d.get('eps_d_eV', (0, 0)), eps_p=d.get('eps_p_eV', (0, 0)), t=t,
                             d_pi4=_cmat(d, 'd_pi4_eV'), d_pi4_dn=_cmat(d, 'd_pi4_dn_eV'), subspace=sub,
                             d_full8=_cmat(d, 'd_full8_eV'), d_full8_dn=_cmat(d, 'd_full8_dn_eV'),
                             eps_d_sigma=d.get('eps_d_sigma_eV', 0.0), eps_p_sigma=d.get('eps_p_sigma_eV', 0.0),
                             t_sigma=_cnum(d.get('t_sigma_eV', 0.0)), d_sigma2=_cmat(d, 'd_sigma2_eV'),
                             d_sigma2_dn=_cmat(d, 'd_sigma2_dn_eV'), eps_delta=d.get('eps_delta_eV', (0, 0)),
                             extract_params=bool(d.get('extract_params', False)), **geo)
    if kind == 'SIGMA_5S':
        t = d.get('t_eV')
        return build_sigma5s(d['ru'], d['o'], _img(d.get('o_img', (0, 0, 0))), d.get('side', 0),
                             _img(d.get('side_img', (0, 0, 0))), d=_cmat(d, 'd_eV'), d_dn=_cmat(d, 'd_dn_eV'),
                             eps=d.get('eps_eV'), t=t, with_o2s=bool(d.get('with_o2s', False)),
                             frame=d.get('frame', 'moving'), frame_ref=d.get('frame_ref'),
                             center=d.get('center', 'moving'), center_ref=d.get('center_ref'), tag=d.get('tag'),
                             ru_shells=tuple(d.get('ru_shells', ('5S', '4D'))), o_shell=d.get('o_shell', '2P'),
                             o2s_shell=d.get('o2s_shell', '2S'), xhelper=d.get('xhelper'), **common)
    # IAO_CHANNEL / OBSERVER: general group
    atoms = []
    for a in d['atoms']:
        _check_keys(a, ATOM_KEYS, where + ' atoms entry')
        atoms.append((a['atom'], _img(a.get('img', (0, 0, 0))), a.get('element', 'ANY'), list(a['shells'])))
    anchors = d.get('anchors')
    if anchors is not None:
        anchors = tuple((a[0], _img(a[1])) for a in anchors)
    if kind == 'OBSERVER':
        return build_observer(atoms, tag=d.get('tag', 'observer'), frame=d.get('frame', 'global'), anchors=anchors,
                              frame_ref=d.get('frame_ref'), bottom=common['bottom'])
    cl = d['cloc']
    if isinstance(cl, str):
        if cl.lower() != 'identity':
            raise ValueError("%s: cloc must be a matrix or 'identity'" % where)
        cl = None
    nao = sum(2 * shell_l(s) + 1 for _, _, _, sh in atoms for s in sh)
    nch = nao if cl is None else np.asarray(cl, float).shape[1]
    dre = d.get('d_eV', [])
    dim = d.get('d_im_eV')
    if dim is not None and len(dim) != len(dre):
        raise ValueError('%s: d_im_eV needs one matrix per d_eV matrix' % where)
    dl = [np.asarray(x, float) + (0 if dim is None else 1j * np.asarray(dim[i], float)) for i, x in enumerate(dre)]
    spin = d.get('spin')
    nsp = len(dl) if dl else (2 if spin == 'resolved' else 1)
    if not dl:
        dl = [np.zeros((nch, nch), complex) for _ in range(nsp)]
    if len(dl) > 2:
        raise ValueError('%s: at most two D matrices (spin up, down)' % where)
    params = []
    if 'pair' in d:
        p = d['pair']
        _check_keys(p, PAIR_KEYS, where + ' pair')
        r = int(p['rank'])
        if 2 * r != nch:
            raise ValueError('%s: pair rank %d needs nch = %d (paired channels [u_1..u_r, v_1..v_r]); nch = %d'
                             % (where, r, 2 * r, nch))
        vals = {k: p.get(k + '_eV') for k in ('v_pop', 'v_pol', 'v_coh')}
        for k, v in vals.items():
            if v is not None and len(v) != r:
                raise ValueError('%s pair: %s_eV needs %d values' % (where, k, r))
        pd = pair_dmat(r, vals['v_pop'], vals['v_pol'], vals['v_coh'])
        dl = [x + pd for x in dl]
        params += pair_params(r, vals['v_pop'], vals['v_pol'], vals['v_coh'])
    for p in d.get('params', []):
        _check_keys(p, PARAM_KEYS, where + ' params entry')
        b = np.zeros((nch, nch), complex)
        for ent in p['basis']:
            i, j, re_, im_ = int(ent[0]), int(ent[1]), float(ent[2]), float(ent[3]) if len(ent) > 3 else 0.0
            if not (1 <= i <= nch and 1 <= j <= nch):
                raise ValueError('%s param %s: index out of 1..%d' % (where, p['name'], nch))
            b[i - 1, j - 1] += re_ + 1j * im_
        if np.max(np.abs(b - b.conj().T)) > 1e-12:
            raise ValueError('%s param %s: basis must be Hermitian (list both (i,j) and (j,i))' % (where, p['name']))
        sp = int(p.get('spin', 0))
        val = float(p.get('value_eV', 0.0))
        for s in range(len(dl)):
            if sp in (0, s + 1):
                dl[s] = dl[s] + val * b
        if sp == 2 and len(dl) == 1:
            raise ValueError('%s param %s: spin 2 needs a spin-resolved site (two d_eV matrices)' % (where, p['name']))
        params.append((p['name'], sp, val, b))
    return build_iao_channel(atoms, cl, d.get('frame', 'moving'), anchors, dl, d.get('tag', 'IAO'), params or None,
                             center=d.get('center', 'moving'), center_ref=d.get('center_ref'),
                             frame_ref=d.get('frame_ref'), spin=spin,
                             anchor_elements=tuple(d['anchor_elements']) if 'anchor_elements' in d else None,
                             xhelper=d.get('xhelper'), **common)


def check_param_names(sites):
    import re
    for s in sites:
        seen = set()
        for name, sp, val, b in s.params:
            if not re.match(r'^[A-Za-z][A-Za-z0-9_\-]{0,31}$', name):
                raise ValueError('site %s: PARAM name %r (letters, digits, _ and -, <= 32 chars)' % (s.tag, name))
            if (name, sp) in seen:
                raise ValueError('site %s: duplicate PARAM %s spin %d' % (s.tag, name, sp))
            seen.add((name, sp))
            if sp not in (0, 1, 2):
                raise ValueError('site %s: PARAM %s spin must be 0, 1 or 2' % (s.tag, name))
            if sp == 2 and s.spin == 'shared':
                raise ValueError('site %s: PARAM %s spin 2 on a SPIN shared site' % (s.tag, name))


def compile_json(d):
    """JSON dict -> (ofe.in text, [Site], notes).  Unknown top-level / control / site keys raise ValueError."""
    _check_keys(d, TOP_KEYS, 'top level')
    sites = [site_from_json(s, i) for i, s in enumerate(d['sites'], 1)]
    check_param_names(sites)
    notes = []
    txt = write_ofe_in(sites, d.get('control'), comment=d.get('comment'),
                       bottom_o_atoms=tuple(d.get('bottom_o_atoms', BOTTOM_O_DEFAULT)), notes_out=notes)
    return txt, sites, notes


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description='compile a JSON site list into ofe.in, or check an ofe.in')
    sub = ap.add_subparsers(dest='cmd')
    p1 = sub.add_parser('compile'); p1.add_argument('json'); p1.add_argument('-o', '--out', default='ofe.in')
    p2 = sub.add_parser('check'); p2.add_argument('ofe_in')
    a = ap.parse_args(argv)
    if a.cmd == 'compile':
        d = json.load(open(a.json))
        txt, sites, notes = compile_json(d)
        with open(a.out, 'w', newline='\n') as fh:
            fh.write(txt)
        print('wrote %s (%d sites)' % (a.out, len(sites)))
        for n in notes:
            print('NOTE ' + n)
    elif a.cmd == 'check':
        ctl, sites = read_ofe_in(open(a.ofe_in).read())
        for s in sites:
            print(s.kind, s.tag, 'nao', s.nao, 'nch', s.nch, s.frame, s.center, s.spin, 'params', len(s.params))
        for k in ('print_hch', 'print_hch_k'):
            if k in ctl:
                print('control %s = %s' % (k, ctl[k]))
    else:
        ap.print_help()


if __name__ == '__main__':
    main(sys.argv[1:])
