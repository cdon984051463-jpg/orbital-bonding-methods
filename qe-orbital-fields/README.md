# Orbital-selective external fields for Quantum ESPRESSO 7.5

**Tangtao Chen · TU Berlin**

This directory preserves two separately validated implementations:

| Path | Archived implementation | Defined use |
|---|---|---|
| `ofe/` | OFE V1.3, 2026-10-27; compatible V1.2 Python input/calibration tools | Specified atomic-orbital and joint-channel operators, with their established moving/frozen frame rules |
| `fixed-iao/` | Fixed full-cell QE-native IAO projector branch, 2026-10-26 | Fixed-reference static SCF, conjugate population and raw-force checks |

Both use explicit operators and self-consistent response. A common finite-rank
form is

\[
\Delta H_k=W_k D W_k^{\dagger},\qquad
E_{\mathrm{ext}}=\mathrm{Tr}(D N),\qquad
N=W_k^{\dagger}\rho_k W_k
\]

with the appropriate k/spin sums and the branch's PAW metric conventions.
Field parameters are supplied in eV; native QE energies and forces are Ry and
Ry/bohr. The external expectation is already included in the engine energy
bookkeeping. Do not add it again to the printed total energy.

The native-QE IAO construction uses the `QE_UPF_full` reference shells, including
the UPF-defined Ru semicore. This differs from the standard-Ru VASP template in
the separate `iao-svd/` preprocessing path. Their coefficient files and PAW
metrics cannot be interchanged.

## OFE V1.3

The canonical `ofe.in` format supports the archived RU_D, O_P_BOND, RU_O_PI and
IAO_CHANNEL site definitions. The full implementation and detailed parameter
conventions are in `ofe/src/` and `ofe/tools/ofe_input.py`. Local AO-SVD channel
calibration in this engine is distinct from the separately exported full-cell
IAO branch below; names alone do not make their projectors interchangeable.

The validated V1.3 source retains the earlier `OFE_V1_2_20261026` string in its
history header. This archive preserves that source unchanged; the V1.3 source
header, included fixes and repository version identify the archived implementation.

Check the included ten-atom input:

```text
uv run qe-orbital-fields/ofe/tools/ofe_input.py check qe-orbital-fields/ofe/example/ofe.in
```

The ten-atom geometry is an existing Ru3O7 software regression fixture, not an
equilibrium material model. Its `qe.in` and `ofe.in` are matched to each other.
Supply the exact named UPF files in the run directory's `pseudo/` directory.

Build on Linux from an already successful **GNU/MPI CPU CMake build of QE 7.5**
and its matching pristine source tree:

```text
bash qe-orbital-fields/ofe/build.sh QE75_CPU_BUILD QE75_SOURCE NEW_OFE_BUILD
```

The command reads the pristine source/libraries and writes a new output
directory. The established hook strings must match exactly once; the original
source is not changed. The result is `NEW_OFE_BUILD/pw_ofe.x`. The build recipe
uses the existing QE `libqe*.a`, `libmbd.a`, FFTW3 and OpenBLAS interface. A
different compiler/library layout requires an explicit port and new validation.

Run a copied example in its own new calculation directory, using an allocation
appropriate to that calculation:

```text
mpirun -np 4 PATH_TO_BUILD/pw_ofe.x -nk 1 -in qe.in > qe.log
```

This archive provides the CPU build path. The V1.3 source retains its original
conditional GPU code, but a new GPU build is not validated by this archive.
V1.3's historical small-system single-point and relaxation tests are described
in `validation/ofe-v13.md`. They do not establish arbitrary-system convergence.

Input generation and response inspection use the existing tools:
`ofe_input.py` for the canonical input, `ofe_history.py` for the engine history,
and `ofe_calibrate.py` with `make_channel_input.py` for the specified AO-channel
calibration. Each tool has `--help` or prints its direct usage when invoked.
Site atoms, periodic images and frame anchors must match the actual model.

## Fixed full-cell IAO projector branch

This static branch applies

\[
\Delta H_{k\sigma}=\lambda W_{0,k\sigma}W_{0,k\sigma}^{\dagger},\qquad
W_0=S_0\chi_0.
\]

`W0` is a complete dual projector, normalized through the reference PAW metric;
it remains fixed during SCF and the displacement checks. Ordinary L2 or W†SW
renormalization changes the operator and is inappropriate here.

```text
bash qe-orbital-fields/fixed-iao/build.sh QE75_CPU_BUILD QE75_SOURCE NEW_FIXED_IAO_BUILD
```

This builds `pw_iao_fixed.x`, `qe_iao_export.x` and `qe_iao_projector.x` against
the same pristine QE libraries. Copy `fixed-iao/example/qe.in`, `iao_field.in`
and the complete `projectors/` directory into a new run directory; supply its
matching Ru/O UPF files in `pseudo/`. Run `pw_iao_fixed.x` with one k pool.
The supplied projectors belong only to this six-atom reference geometry,
cell, k mesh and PAW setup.

For a new reference, follow this existing sequence:

1. `native_backend/qe_iao.py export` reads a converged `.save` plus its log and
   invokes `qe_iao_export.x` to export native PAW overlaps.
2. `native_backend/qe_iao.py analyze` constructs the full-cell IAO density for
   the declared mesh and pair; use its `--help` for the argument format.
3. `prepare_channel.py` selects the specified density channel and writes its
   reference coefficients. Its current selection is `(u−v)/sqrt(2)`; this is
   not, by itself, a Hamiltonian antibonding eigenstate.
4. `qe_iao_projector.x` writes complete dual projectors; `verify_projector.py`
   independently reads them back. Use exactly these frozen projectors for the
   subsequent field strengths.

The reference `.save` must contain wavefunctions. The example SCF uses `nowf`
and therefore is not an input for generating a new calibration wavefunction.
The Fortran projector input uses `prefix`, `outdir`, `coeff_dir` and
`projector_dir`; these must point to one matching reference, not a mixture.

Supported static scope: CPU, collinear nspin=1/2, general k points, nosym,
semilocal DFT, one k pool/band group/image, fixed cell, `tprnfor=true` and
`tstress=false`. Restart, ionic relaxation, MD, Hubbard, hybrids, meta-GGA,
SOC and stress are outside this fixed-projector implementation's accepted scope.

The conjugate occupation is printed as `IAO_DEDV_ELECTRONS`. Force comparisons
must use `IAO_RAW_FORCE_RY_BOHR`, before QE subtracts a net force. The archived
real SCF validation covers nine runs and two Cartesian force components on a
six-atom model; detailed numerical records are under `fixed-iao/validation/`.

## 中文说明与许可

OFE V1.3 与固定全晶胞 IAO 外势分开归档，各自保持已有的算符、坐标和支持范围。
二者的通道或投影子不能因名称相似而互换。当前整理只验证源码接入、输入工具和
已有数值记录；Linux/MPI 程序的历史验收与本次本地检查分开列出。

源码衍生部分保留 GPL-2.0-or-later，完整文本见 `LICENSE`。仓库不附二进制或 UPF，
也不附特定集群的调度、账号或模块配置。
