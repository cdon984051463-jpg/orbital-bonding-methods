#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""T5 ofe_calibrate.py (OFE V1.2 tools, 2026-10-26): IAO-SVD channel calibration from an OFE history file.

Closes the broken link of the review (section 3.2): make_channel_input.py only reads the old IC_* records.
This adapter reads the OFE records of a zero-field C = I calibration run (e.g. examples ex4):
  OFE_N_AO site spin row  (lab-frame AO density, real part; spin-summed here: nspin 1 rows are already
                           spin-summed, nspin 2 rows are summed over the two spins -> one C for both spins)
  OFE_FRAME site col      (Q used by the engine at that step, columns x y z)
  OFE_SITE_ATOM / OFE_SITE_ANCHORS / OFE_SITE (group, images, shells, frame mode, anchors)
and calls make_channel_input.channels() (A-B block SVD, unchanged) -> C_lab = [u_1..u_r, v_1..v_r] (or the
--split-b columns); C_LOCAL = Rblk(Q)^T C_lab for moving/frozen frames (as make_channel_input main), C_lab for
global.  Named parameters: paired channels -> v_pop i, v_pol i, v_coh i (ofe_input.pair_params);
split-b channels -> v_n_<col> (E_jj) and v_coh<i>_<atom> ((E_uv + E_vu)/2).  D = sum V B (default 0).
Shell labels L0/L1/L2 are passed to make_channel_input as XS/XP/XD (it only uses the l letter).
Frozen frames get FRAME_REF = the Q of the calibration step (explicit, as V1.1 requires for restart/relax).
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ofe_history as oh  # noqa: E402
import ofe_input as oi  # noqa: E402
import make_channel_input as mci  # noqa: E402


def _mci_label(lab):
    lab = lab.upper()
    if lab[0] == 'L':
        return 'X' + 'SPD'[int(lab[1])]
    return lab


def mci_group(group):
    """[(atom, img, element, shells)] -> make_channel_input group [(atom, img, [labels])]."""
    return [(a, list(img), [_mci_label(s) for s in sh]) for a, img, el, sh in group]


def rblk(group, q):
    """block-diagonal AO rotation for the group (make_channel_input.rot_block per shell)."""
    _, shells = mci.ao_layout(mci_group(group))
    nao = mci.ao_layout(mci_group(group))[0]
    rb = np.zeros((len(nao), len(nao)))
    for st, l in shells:
        rb[st:st + 2 * l + 1, st:st + 2 * l + 1] = mci.rot_block(l, q)
    return rb


def split_params(labels, rank, side_b):
    """named parameters for split-b columns [u_1..u_r, v_i@atom ...]."""
    n = len(labels)
    out = []
    for j, lab in enumerate(labels):
        nm = lab.split('(')[0].replace('@', '_')
        out.append(('v_n_' + nm, 0, 0.0, oi._e(n, j, j)))
    for i in range(rank):
        for j, lab in enumerate(labels):
            if lab.startswith('v%d@' % (i + 1)):
                atom = lab.split('@')[1].split('(')[0]
                out.append(('v_coh%d_%s' % (i + 1, atom), 0, 0.0, oi._e(n, i, j, .5) + oi._e(n, j, i, .5)))
    return out


def calibrate(hist, site, side_a, side_b, rank=None, split_b=False, step=-1, v_pop=None, v_pol=None, v_coh=None,
              tag=None):
    """hist: parsed history (ofe_history.parse_history).  Returns dict(site=Site, C_lab, C_loc, singular_values,
    columns, N_ch_reference, frame_reference, notes)."""
    st = hist['steps'][step]
    nspin = hist['header']['run']['nspin']
    meta = hist['header']['sites'][site]
    group = oh.site_group(hist, site)
    atoms_in = [a for a, _, _, _ in group]
    if len(set(atoms_in)) != len(atoms_in):
        raise ValueError('an atom appears twice in the group (different images): side_a/side_b are ambiguous')
    for a in list(side_a) + list(side_b):
        if a not in atoms_in:
            raise ValueError('side atom %d not in the site group %s' % (a, atoms_in))
    n = oh.n_ao_spin_summed(st, site, nspin)
    if np.max(np.abs(n - n.T)) > 1e-8:
        raise ValueError('OFE_N_AO not symmetric (max %.2e)' % np.max(np.abs(n - n.T)))
    q = st['frame'][site]
    clab, s, labels = mci.channels(n.astype(complex), mci_group(group), list(side_a), list(side_b), rank, split_b)
    frame = meta['frame'].lower()
    cloc = clab if frame == 'global' else rblk(group, q).T @ clab
    r = len(s) if rank is None else rank
    nch = clab.shape[1]
    notes = []
    if not split_b:
        params = oi.pair_params(r, v_pop, v_pol, v_coh)
        d = oi.pair_dmat(r, v_pop, v_pol, v_coh)
    else:
        if any(x is not None for x in (v_pop, v_pol, v_coh)):
            raise ValueError('split-b channels: set values afterwards with ofe_input.set_param (v_n_*, v_coh*_*)')
        params = split_params(labels, r, side_b)
        d = np.zeros((nch, nch))
    anc = hist['header'].get('site_anchors', {}).get(site)
    anchors = None if anc is None else anc['anchors']
    if anchors is not None and not anc['use_x']:
        anchors = (anchors[0], anchors[1], (0, (0, 0, 0)))
    fref = q.copy() if frame == 'frozen' else None
    if frame == 'frozen':
        notes.append('frame frozen: FRAME_REF set to the calibration-step Q')
    site_obj = oi.build_iao_channel(group, cloc, frame, anchors, [d], tag or ('%s_cal' % meta['tag']), params,
                                    frame_ref=fref)
    nch_ref = clab.T @ n @ clab
    return dict(site=site_obj, C_lab=clab, C_loc=cloc, singular_values=s.tolist(), columns=labels,
                N_ch_reference=nch_ref, frame_reference=q, notes=notes, nspin=nspin)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('history')
    ap.add_argument('--site', type=int, default=1)
    ap.add_argument('--side-a', required=True, help='comma list of atom numbers (1-based)')
    ap.add_argument('--side-b', required=True)
    ap.add_argument('--rank', type=int)
    ap.add_argument('--split-b', action='store_true')
    ap.add_argument('--step', type=int, default=-1)
    ap.add_argument('--v-pop', default=''); ap.add_argument('--v-pol', default=''); ap.add_argument('--v-coh', default='')
    ap.add_argument('--tag')
    ap.add_argument('--control', default='{}', help='JSON dict for &ofe_control of the written ofe.in')
    ap.add_argument('-o', '--out', default='ofe_calibrated.in')
    a = ap.parse_args(argv)
    hist = oh.parse_history(open(a.history).read())
    vec = lambda s_: [float(x) for x in s_.split(',')] if s_ else None  # noqa: E731
    res = calibrate(hist, a.site, [int(x) for x in a.side_a.split(',')], [int(x) for x in a.side_b.split(',')],
                    a.rank, a.split_b, a.step, vec(a.v_pop), vec(a.v_pol), vec(a.v_coh), a.tag)
    notes = []
    txt = oi.write_ofe_in([res['site']], json.loads(a.control), comment='calibrated by ofe_calibrate.py from %s site %d'
                          % (os.path.basename(a.history), a.site), notes_out=notes)
    with open(a.out, 'w', newline='\n') as fh:
        fh.write(txt)
    summ = dict(singular_values=res['singular_values'], columns=res['columns'], N_ch_reference=res['N_ch_reference'].tolist(),
                frame_reference=res['frame_reference'].tolist(), history=a.history, site=a.site, notes=res['notes'] + notes,
                params=[p[0] for p in res['site'].params])
    with open(a.out + '.json', 'w') as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(dict(singular_values=summ['singular_values'], columns=summ['columns'], params=summ['params']), indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
