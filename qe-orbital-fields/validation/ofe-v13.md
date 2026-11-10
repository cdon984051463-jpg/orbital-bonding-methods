# Historical OFE V1.3 validation

Source implementation: OFE_V1_3_20261027. The numerical record was produced on
2026-10-27 and reviewed again when preparing this archive. These calculations
were not rerun for the GitHub packaging task.

The original model is the Ru3O7 software regression system with matching
pristine-QE and preceding-engine comparisons.

| Check | Energy difference | Other recorded comparison | Result |
|---|---|---|---|
| CPU V1.3 vs V1.2, fixed geometry | 1.57e-11 Ry | max force difference 6.30e-8 eV/Å; occupation difference 1.46e-9 e | Pass |
| CPU k-pool decomposition, 1 vs 2 pools | 5.14e-11 Ry | max force difference 1.95e-7 eV/Å; occupation difference 9.30e-9 e | Pass |
| CPU zero-field relaxation vs pristine QE | 6.32e-11 Ry | maximum coordinate difference 5.51e-8 Å; both converged in 25 BFGS steps | Pass |
| Nonzero-field CPU vs historical GPU relaxation | 9.26e-10 Ry | maximum coordinate difference 3.32e-8 Å; both converged | Pass |

The V1.3 source fixes the recorded periodic-image continuation and global
condition-number reporting while preserving the corresponding fixed-geometry
response. This evidence applies to the tested system and layouts, not to all
models, all fields or new toolchains. Project-level mechanism claims are not
part of this archive's validation.
