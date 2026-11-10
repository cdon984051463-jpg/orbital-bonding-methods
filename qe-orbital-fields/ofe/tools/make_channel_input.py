#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""IAO_CHANNEL_V1 输入工具（标定 → 通道文件 → D 矩阵 namelist）。

子命令
  template  写“标定用”通道文件：C = 单位阵（nch = nao），配合 D=0 的零场 SCF（tprnfor=.true.）。
            该运行在 iao_channel_field_history.dat 中写出组内 AO 密度 IC_N_AO_LAB_ROW_RE_IM 与坐标系。
  channels  读零场历史文件最后一步：自旋求和 AO 密度 N_ao（实验室系），取 A–B 块 SVD
            → 成对通道 C_lab=[u_1..u_r, v_1..v_r]；moving 模式写 C_loc = Rblk(Q_ref)^T C_lab。
            --split-b 把 v_i 按 B 侧原子拆成归一化分量（三中心 3c-4e：u_σ、v_σ(顶)、v_σ(底)）。
  dmat      由 V_pop/V_pol/V_coh（eV，按成对通道）或 3c 相干场写 &iao_channel_field 的 d_re_eV 行。

AO 顺序与 QE 模块一致：原子按文件顺序，壳层按列出顺序，m 顺序 s；x,y,z；z2,xz,yz,x2-y2,xy（正笛卡尔）。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

LDIM = {'S': 0, 'P': 1, 'D': 2}


def d_basis():
    b = np.zeros((5, 3, 3))
    b[0] = np.diag([-1.0, -1.0, 2.0]) / np.sqrt(6.0)
    b[1][0, 2] = b[1][2, 0] = 1 / np.sqrt(2.0)
    b[2][1, 2] = b[2][2, 1] = 1 / np.sqrt(2.0)
    b[3] = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    b[4][0, 1] = b[4][1, 0] = 1 / np.sqrt(2.0)
    return b


def rot_block(l, q):
    if l == 0:
        return np.ones((1, 1))
    if l == 1:
        return np.asarray(q, float)
    b = d_basis()
    return np.einsum('iab,jba->ij', b, np.einsum('ab,jbc,dc->jad', q, b, q))


def parse_group(spec):
    """'1:0,0,0:5S,4D;113:0,0,0:2P;93:0,0,0:2P' → [(atom, img, [labels])]"""
    out = []
    for item in spec.split(';'):
        a, img, sh = item.split(':')
        out.append((int(a), [int(x) for x in img.split(',')], [s.strip().upper() for s in sh.split(',')]))
    return out


def ao_layout(group):
    rows, shells = [], []
    for g, (a, img, labs) in enumerate(group):
        for lab in labs:
            l = LDIM[lab[1]]
            shells.append((len(rows), l))
            rows += [(g, a, lab, m) for m in range(2 * l + 1)]
    return rows, shells


def write_channel_file(path, tag, group, nch, moving, anchors, c):
    rows, _ = ao_layout(group)
    c = np.asarray(c, float)
    assert c.shape == (len(rows), nch)
    err = np.max(abs(c.T @ c - np.eye(nch)))
    if err > 1e-10:
        raise ValueError(f'C not orthonormal ({err:.2e})')
    lines = ['IAO_CHANNEL_V1', tag[:64], f'{len(group)} {nch} {1 if moving else 0}']
    for a, img, labs in group:
        lines.append(f'{a} {img[0]} {img[1]} {img[2]} {len(labs)} ' + ' '.join(labs))
    if moving:
        lines.append(' '.join(f'{a} {i[0]} {i[1]} {i[2]}' for a, i in anchors))
    for r in c:
        lines.append(' '.join(f'{x:.16e}' for x in r))
    lines.append('END_IAO_CHANNEL_V1')
    Path(path).write_text('\n'.join(lines) + '\n', newline='\n')


def parse_anchors(s):
    if not s:
        return None
    out = []
    for item in s.split(';'):
        a, img = item.split(':')
        out.append((int(a), [int(x) for x in img.split(',')]))
    assert len(out) == 3
    return out


def read_history(path, site=1):
    """最后一个完整 IC_BEGIN_STEP 块：N_ao（每自旋）与坐标系 Q。"""
    blocks, cur = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith('IC_BEGIN_STEP'):
            cur = dict(nao={}, frame={})
        elif cur is None:
            continue
        elif line.startswith('IC_N_AO_LAB_ROW_RE_IM'):
            t = line.split(); s_, sp, j = int(t[1]), int(t[2]), int(t[3])
            if s_ == site:
                v = np.array(t[4:], float); n = len(v) // 2
                cur['nao'].setdefault(sp, {})[j] = v[:n] + 1j * v[n:]
        elif line.startswith('IC_FRAME_X_Y_Z_COLUMNS'):
            t = line.split()
            if int(t[1]) == site:
                cur['frame'][int(t[2])] = np.array(t[3:6], float)
        elif line.startswith('IC_EEXT_TOTAL_RY'):
            blocks.append(cur); cur = None
    if not blocks:
        raise ValueError('no complete IC step in history')
    b = blocks[-1]
    nao = {sp: np.array([rows[j] for j in sorted(rows)]) for sp, rows in b['nao'].items()}
    q = np.column_stack([b['frame'][j] for j in (1, 2, 3)])
    return nao, q


def channels(nao_lab, group, side_a, side_b, rank=None, split_b=False):
    rows, _ = ao_layout(group)
    ra = [i for i, r in enumerate(rows) if r[1] in side_a]
    rb = [i for i, r in enumerate(rows) if r[1] in side_b]
    if np.max(abs(nao_lab.imag)) > 1e-8:
        raise ValueError('complex AO density: wrong image or broken time reversal')
    n = nao_lab.real
    u, s, vt = np.linalg.svd(n[np.ix_(ra, rb)], full_matrices=False)
    r = len(s) if rank is None else rank
    cols, labels = [], []
    for i in range(r):
        ui, vi = u[:, i].copy(), vt[i].copy()
        sg = np.sign(ui[np.argmax(abs(ui))]); ui *= sg; vi *= sg
        cu = np.zeros(len(rows)); cu[ra] = ui
        cols.append(cu); labels.append(f'u{i+1}')
    for i in range(r):
        vi = vt[i] * np.sign(u[:, i][np.argmax(abs(u[:, i]))])
        if not split_b:
            cv = np.zeros(len(rows)); cv[rb] = vi; cols.append(cv); labels.append(f'v{i+1}')
        else:
            for atom in side_b:
                idx = [k for k, row in enumerate(rb) if rows[row][1] == atom]
                part = vi[idx]; nrm = np.linalg.norm(part)
                if nrm < 1e-8:
                    continue
                cv = np.zeros(len(rows)); cv[[rb[k] for k in idx]] = part / nrm
                cols.append(cv); labels.append(f'v{i+1}@{atom}(share={nrm**2:.4f})')
    return np.column_stack(cols), s, labels


def pair_dmat(r, v_pop, v_pol, v_coh):
    d = np.zeros((2 * r, 2 * r))
    for i in range(r):
        d[i, i] += 0.5 * (v_pop[i] + v_pol[i]); d[r + i, r + i] += 0.5 * (v_pop[i] - v_pol[i])
        d[i, r + i] += 0.5 * v_coh[i]; d[r + i, i] += 0.5 * v_coh[i]
    return d


def namelist_lines(d, site=1, spin=1):
    out = []
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if d[i, j] != 0.0:
                out.append(f'  d_re_eV({i+1},{j+1},{site},{spin}) = {d[i, j]:.10f}')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('template'); t.add_argument('--group', required=True); t.add_argument('--anchors')
    t.add_argument('--tag', default='calibration'); t.add_argument('out')
    c = sub.add_parser('channels'); c.add_argument('--group', required=True); c.add_argument('--anchors')
    c.add_argument('--history', required=True); c.add_argument('--site', type=int, default=1)
    c.add_argument('--side-a', required=True); c.add_argument('--side-b', required=True)
    c.add_argument('--rank', type=int); c.add_argument('--split-b', action='store_true')
    c.add_argument('--tag', default='channels'); c.add_argument('out')
    d = sub.add_parser('dmat'); d.add_argument('--rank', type=int, required=True)
    d.add_argument('--v-pop', default=''); d.add_argument('--v-pol', default=''); d.add_argument('--v-coh', default='')
    d.add_argument('--site', type=int, default=1)
    a = p.parse_args()
    if a.cmd == 'template':
        group = parse_group(a.group); anchors = parse_anchors(a.anchors)
        rows, _ = ao_layout(group)
        write_channel_file(a.out, a.tag, group, len(rows), anchors is not None, anchors, np.eye(len(rows)))
    elif a.cmd == 'channels':
        group = parse_group(a.group); anchors = parse_anchors(a.anchors)
        nao, q = read_history(a.history, a.site)
        ntot = sum(nao.values())                       # 自旋求和：同一组 C 用于两个自旋
        sa = [int(x) for x in a.side_a.split(',')]; sb = [int(x) for x in a.side_b.split(',')]
        clab, s, labels = channels(ntot, group, sa, sb, a.rank, a.split_b)
        if anchors is not None:
            _, shells = ao_layout(group)
            rb = np.zeros((clab.shape[0],) * 2)
            for st, l in shells:
                rb[st:st + 2 * l + 1, st:st + 2 * l + 1] = rot_block(l, q)
            cwrite = rb.T @ clab
        else:
            cwrite = clab
        write_channel_file(a.out, a.tag, group, clab.shape[1], anchors is not None, anchors, cwrite)
        nch = clab.T @ ntot.real @ clab
        Path(str(a.out) + '.json').write_text(json.dumps(dict(singular_values=s.tolist(), columns=labels,
            N_ch_reference=nch.tolist(), frame_reference=q.tolist(), history=str(a.history)), indent=1))
        print(json.dumps(dict(singular_values=s.tolist(), columns=labels), indent=1))
    else:
        r = a.rank
        vec = lambda s_: np.array([float(x) for x in s_.split(',')]) if s_ else np.zeros(r)
        dm = pair_dmat(r, vec(a.v_pop), vec(a.v_pol), vec(a.v_coh))
        print('\n'.join(namelist_lines(dm, a.site)))


if __name__ == '__main__':
    main()
