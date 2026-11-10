# Archive validation — 2026-11-10

This record distinguishes checks run while preparing the repository from the
historical calculations on which the method examples are based.

## Checks run for this repository

| Check | Evidence and scope | Result |
|---|---|---|
| IAO-SVD real-data reference | All four singular values agree with the archived O 2s+2p clean reference within 1e-12; channel shares within 1e-10 percentage points | Pass |
| IAO command-line run | The included periodic density JSON completes the original numerical workflow and writes the channel table, coefficients and validation report in a new directory | Pass |
| IAO numerical preservation | The SVD `compute` function and complete periodic-density/symmetry modules retain the original numerical syntax trees; extracted support-routine bodies are unchanged | Pass |
| Dependency/entry-point imports | PAW density construction and BvK field entry points import the pinned external PyVaspWfc dependency; pair selection and viewer interfaces load from their new relative locations | Pass |
| OFE example | Canonical ten-atom input parses all four site kinds: RU_D, O_P_BOND, RU_O_PI and IAO_CHANNEL | Pass |
| QE source hooks | Both the OFE CPU and fixed-IAO patchers apply their hooks successfully to matching pristine QE 7.5 source copies; original source is not overwritten | Pass |
| Linux build-script syntax | Both Bash build recipes pass a syntax check | Pass |
| Native QE IAO entry | Export/analyze interface imports its shared mathematical routines through the repository-relative path | Pass |
| LOBSTER saved results | Eight full-precision CSV rows agree exactly with their saved JSON summaries and the original per-k weighted sums | Pass |
| LOBSTER plotting | The standalone command regenerates PNG, PDF and SVG from the included source data | Pass |
| Publication contents | Python sources parse; prepared text contains no private machine/cluster paths or active digest machinery; binaries, licensed potentials and private scheduler files are excluded | Pass |

Run the numerical example checks from the repository root:

```text
uv run tests/verify_examples.py
```

The plotting geometry and annotation conventions are retained from the
previously checked four-panel figure. That figure's rendered alignment,
font-size floor and collision checks passed when it was prepared; this archive
also executes the standalone plot command with the same source data.

## Historical validation retained

- IAO default-basis reference: 2026-09-08 full-cell O 2s+2p clean data, including
  the original atom/image selection and full environment density. The example
  retains four mathematical channels. It does not establish field-integral
  convergence or new chemical mechanisms.
- OFE V1.3: the 2026-10-27 matched-engine, pool-layout and small-system relaxation
  comparisons are recorded in `qe-orbital-fields/validation/ofe-v13.md`.
- Fixed IAO field: the original nine six-atom SCF runs, PAW-metric checks,
  complete-projector readback and two Cartesian force finite differences are
  recorded under `qe-orbital-fields/fixed-iao/validation/`.
- LOBSTER: the 2026-11-04 real-matrix replay covered eight state/channel cases
  with 24 k points per state. Its numerical checks remain in each included
  result JSON. The source analysis routine is unchanged.

## Scope of the current validation

The archive task did not submit new DFT jobs or rebuild/run the Linux QE
executables locally. PAW preprocessing/field interfaces were checked for
imports and portable input paths; their full wavefunction calculations were
not repeated for this packaging task. LOBSTER raw-matrix analysis was not
repeated again; its previously replayed numerical data are retained and checked.
New systems, bases, empty-band windows, compilers and parallel layouts require
their own scientific and numerical validation.

The upstream PyVaspWfc import emits non-fatal Python escape-sequence warnings
in some docstrings on the tested interpreter; the imports complete successfully.
