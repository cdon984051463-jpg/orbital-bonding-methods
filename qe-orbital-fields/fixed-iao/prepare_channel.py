# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""Freeze a full-cell IAO density-antiphase channel in the native QE basis.

Usage: prepare_channel.py GRAMS IAO RU O CHANNEL NEW_OUTPUT
CHANNEL is one-based. The reference pair uses the declared zero lattice image.
"""
from pathlib import Path
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'native_backend'))
from qe_iao import read_gram

grams, iao = map(Path, sys.argv[1:3])
ru, oxygen, channel = map(int, sys.argv[3:6])
output = Path(sys.argv[6])
report = json.loads((iao / 'report.json').read_text())
if report['status'] != 'NUMERICAL_GATES_PASS':
    raise ValueError('Require an accepted numerical IAO result')
if not 1 <= channel <= 3:
    raise ValueError('Ru5s4d/O2p channel must be 1, 2 or 3')
output.mkdir(parents=True, exist_ok=False)
selection = dict(source_grams=str(grams.resolve()), source_iao=str(iao.resolve()),
                 Ru=ru, O=oxygen, channel=channel, translation=[0, 0, 0],
                 combination='(Ru_partner - O_partner)/sqrt(2)',
                 interpretation='density-antiphase; not identified as an antibonding energy eigenstate',
                 operator='fixed W0=S0*chi0; DeltaH=lambda W0 W0^dagger',
                 independent_spin_bases=True, per_k=[], per_spin={})
vectors = {}
for spin in report['spins']:
    sp = iao / ('spin' + spin)
    with np.load(sp / f'pair_{ru}_{oxygen}.npz') as pair:
        columns_a, columns_b = pair['a_columns'], pair['b_columns']
        values = pair['singular_values']
        vector = np.zeros(np.load(sp / 'density.npy', mmap_mode='r').shape[0], complex)
        vector[columns_a] = pair['U'][:, channel - 1] / np.sqrt(2)
        vector[columns_b] = -pair['Vh'].conj().T[:, channel - 1] / np.sqrt(2)
    if abs(np.vdot(vector, vector) - 1) > 1e-10:
        raise ValueError('Invalid normalized channel partners')
    vectors[int(spin)] = vector
    selection['per_spin'][spin] = dict(singular_values=values.tolist(), reference_occupation=0.)

for path in sorted(grams.glob('k*.gram')):
    block = read_gram(path)
    spin, ik, nb, nr = (block[k] for k in ('spin', 'ik', 'nb', 'nr'))
    with np.load(iao / f'spin{spin}/k{ik:05d}.npz') as arrays:
        active = arrays['active_bands']
        coefficients = arrays['transform'] @ vectors[spin]
    full = np.zeros(nb + nr, complex)
    full[active] = coefficients[:len(active)]
    full[nb:] = coefficients[len(active):]
    joint = np.block([[block['gcc'], block['gcf']], [block['gcf'].conj().T, block['gff']]])
    norm = np.vdot(full, joint @ full)
    if abs(norm - 1) > 1e-8:
        raise ValueError(f'Full-cell channel norm failed at k={ik}: {norm}')
    overlap = np.column_stack((block['gcc'], block['gcf'])) @ full
    occupation = float(np.dot(block['wg'], abs(overlap)**2))
    selection['per_spin'][str(spin)]['reference_occupation'] += occupation
    selection['per_k'].append(dict(k=ik, spin=spin, norm_error=float(abs(norm-1)),
                                   weighted_occupation=occupation))
    with (output / f'k{ik:05d}.coeff').open('x') as f:
        f.write(f'{ik} {spin} {nb} {nr} {block["nks"]}\n')
        np.savetxt(f, np.column_stack((full.real, full.imag)), fmt='%.17e')
    np.save(output / f'k{ik:05d}.expected.npy', overlap, allow_pickle=False)
(output / 'calibration.json').write_text(json.dumps(selection, indent=2) + '\n')
print(json.dumps(selection, indent=2))
