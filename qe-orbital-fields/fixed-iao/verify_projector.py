# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""Check native projector files against every saved KS overlap.

Usage: verify_projector.py GRAMS COEFFICIENTS PROJECTORS SOURCE_SAVE
"""
from pathlib import Path
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'native_backend'))
from qe_iao import read_gram
from qe_wfc import Wfc

grams, coefficients, projectors, source = map(Path, sys.argv[1:])
rows = []
occupations = {}
for gram_path in sorted(grams.glob('k*.gram')):
    block = read_gram(gram_path)
    ik, spin = block['ik'], block['spin']
    spatial = ik if block['nspin'] == 1 else (ik - 1) % (block['nks'] // 2) + 1
    label = '' if block['nspin'] == 1 else ('up' if spin == 1 else 'dw')
    name = f'wfc{label}{spatial}.dat'
    expected = np.load(coefficients / f'k{ik:05d}.expected.npy')
    native_text = np.loadtxt(projectors / f'k{ik:05d}.overlap')
    native_overlap = native_text[:, 0] + 1j * native_text[:, 1]
    with Wfc(projectors / name) as w, Wfc(source / name) as psi:
        assert w.nbnd == 1 and w.npol == 1 and w.spin == spin and w.ik == spatial
        assert np.array_equal(w.gvec, psi.gvec)
        np.testing.assert_allclose(w.k_cart, psi.k_cart, atol=1e-12, rtol=0)
        np.testing.assert_allclose(w.reciprocal, psi.reciprocal, atol=1e-12, rtol=0)
        dual = w.band(0).ravel()
        readback = np.array([np.vdot(psi.band(n).ravel(), dual) for n in range(psi.nbnd)])
    native_error = float(np.max(abs(native_overlap - expected)))
    readback_error = float(np.max(abs(readback - expected)))
    assert native_error < 1e-9 and readback_error < 1e-9, (ik, native_error, readback_error)
    occupations[spin] = occupations.get(spin, 0.) + float(np.dot(block['wg'], abs(readback)**2))
    rows.append(dict(k=ik, spin=spin, native_overlap_error=native_error,
                     serialized_readback_error=readback_error))
result = dict(status='NATIVE_PROJECTOR_ROUNDTRIP_PASS', per_k=rows,
              source_reference_occupation_per_spin=occupations,
              shared_lambda_conjugate=sum(occupations.values()),
              meaning='W is the PAW dual vector S0*chi0; no additional normalization applied',
              source_contains_other_Ru_d_field=True, new_zero_field_SCF_expected_equal=False,
              new_DFT=False)
(projectors / 'projector_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
