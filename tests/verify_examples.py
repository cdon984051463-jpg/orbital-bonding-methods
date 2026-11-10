# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2"]
# ///
"""Check the real IAO reference, OFE input and saved LOBSTER figure data."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'iao-svd/source'), str(ROOT/'qe-orbital-fields/ofe/tools')]
import principal_density_channels_v2 as pdc
import ofe_input

example = ROOT/'iao-svd/examples/o2s-clean'
problem = pdc.load_problem(example/'config.json')
result = pdc.compute(problem)
pdc.validate_result(problem, result)
reference = json.loads((example/'reference.json').read_text(encoding='utf-8'))
assert np.allclose(result['singular_values'], reference['singular_values'], rtol=0, atol=1e-12)
assert np.allclose(100*result['fractions'], reference['squared_shares_percent'], rtol=0, atol=1e-10)
print('IAO-SVD: four-channel real-data reference passed')

control, sites = ofe_input.read_ofe_in((ROOT/'qe-orbital-fields/ofe/example/ofe.in').read_text(encoding='utf-8'))
assert {site.kind for site in sites} == {'RU_D', 'O_P_BOND', 'RU_O_PI', 'IAO_CHANNEL'}
print('OFE: archived canonical input parsed')

with (ROOT/'lobster-local-levels/example/source_data.csv').open(encoding='utf-8-sig',newline='') as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 8
for row in rows:
    name = f"{row['state']}_{row['channel']}.json"
    saved = json.loads((ROOT/'lobster-local-levels/example/results'/name).read_text(encoding='utf-8'))
    assert saved['numerical_pass']
    for key,value in saved['summary'].items():
        assert float(row[key]) == value, (name,key)
        weighted = sum(k['weight']*k[key] for k in saved['k_resolved'])
        assert weighted == value, (name,key)
print('LOBSTER: all eight saved results reproduce the figure data and k-weighted summaries')
