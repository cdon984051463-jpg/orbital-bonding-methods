# PAW–IAO pair density-coherence channels

**Tangtao Chen · TU Berlin**

The method constructs the density in a full-cell orthonormal IAO basis, selects
two atomic subspaces and the intended lattice image, and takes the SVD of their
density cross block. The current default atomic shells are Ru 5s+4d, O 2s+2p
and H 1s; the O 2s occupied bands are retained with that template choice.

For weighted Bloch-density matrices

\[
\widetilde D(k)=w_k\sum_n f_{nk}c_{nk}c_{nk}^{\dagger},\qquad
D_{\mu(0),\nu(R)}=\sum_k e^{-2\pi i k\cdot R}\widetilde D_{\mu\nu}(k),
\]

the selected block is decomposed as

\[
D_{AB}(R)=U\,\mathrm{diag}(s_i)\,V^{\dagger},\qquad
q_i=\frac{s_i^2}{\sum_j s_j^2}.
\]

The saved `density_matrices_k.npy` already includes the k weights and electronic
occupations. Do not apply the weights a second time. `s_i` and `q_i` quantify
density coherence and its distribution across the pair's channels. They are
not bond energies, dissociation energies or a unique bond order.

## Included real-data example

Run from the repository root:

```text
uv run iao-svd/source/principal_density_channels_v2.py --config iao-svd/examples/o2s-clean/config.json --output work/iao-clean
```

The input JSON names a weighted density array, fractional k points, IAO columns
and lattice translations. The example is the archived O 2s+2p clean Ru2–O93
case, with four mathematical channels. Its independent archived reference is
in `examples/o2s-clean/reference.json`. The complete 520-column density is
included to preserve the environment and the original atom/image selection.

Outputs include `channel_table.csv`, `channel_coefficients.npz`, the density cross
block and `density-channel-validation.json`. The output directory must be new.
The numerical kernel also retains the original same-cell and eigen-coordinate
readers for existing project inputs; the example and workflow documented here
use the periodic density format above.

## Construct the IAO density from a new VASP calculation

Required calculation files are the matching `WAVECAR`, `POTCAR`, `CONTCAR`,
`OUTCAR`, `IBZKPT`, `KPOINTS` and `INCAR`. Atomic templates are separate, existing
free-atom calculations supplied as `atomic-templates/Ru/`, `O/` and, when
present, `H/`; the template loader requires `WAVECAR`, `POTCAR` and `POSCAR`
in each used template directory. Template PAW datasets must match the
calculation. Standard Ru (ZVAL=8; s1 d7) is the supported Ru template here.
No licensed potential contents or template wavefunctions are distributed.

```text
uv run iao-svd/source/build_iao_density.py --calc inputs/model --templates atomic-templates --output work/density/model
uv run iao-svd/source/prepare_pair_configs.py --density-dir work/density/model --calc inputs/model --atom-a RU_INDEX --atom-b O_INDEX --output work/pairs/target/input --structure-label model
uv run iao-svd/source/principal_density_channels_v2.py --config work/pairs/target/input/config.json --output work/pairs/target/result
```

Replace `RU_INDEX` and `O_INDEX` with the actual 1-based atom indices in
`inputs/model/CONTCAR`. The pair generator uses the IAO labels and that actual
structure to select the columns and lattice images. The sample above establishes
the `input/` and `result/` layout consumed by the field builder below.

PAW preprocessing uses an externally installed, pinned PyVaspWfc dependency.
The required project support routines are included under `support/`; they were
retained from the existing implementation with their numerical bodies unchanged.
The archive changes dependency paths and template locations, and compares
normalized PAW dataset text directly without retaining potential contents in reports.

## Signed real-space fields and viewer

The existing field path is `build_bvk_local_channel_fields.py`, followed by
`build_channel_viewer.py`. After the preceding density/pair commands:

```text
uv run iao-svd/source/build_bvk_local_channel_fields.py --density-dir work/density/model --pair-dir work/pairs/target --calc inputs/model --output work/fields/target
uv run iao-svd/source/build_channel_viewer.py --fields-root work/fields --pairs-root work/pairs --structures-root inputs --output work/channels.html
```

Both commands use the same `inputs/model/CONTCAR` and the `model` structure label.
Use each script's `--help` for its existing grid arguments.
The viewer now takes an explicit `--structures-root`; its local Three.js assets
and their MIT license are under `viewer/`. A separate private base viewer is
not required.

Reconstruct fields only from a matched wavefunction, saved IAO coefficients,
pair coefficients and the declared Born–von Karman mesh. Signed partner
amplitudes and weighted partner-product proxies visualize channel shape.
Quantitative `s_i`, shares and populations come from coefficient space;
absolute field integrals require separate grid/PAW-metric convergence.

## Scope

The full-cell builder supports the previously validated collinear, non-SOC
ISYM=0/1 routes and retains their symmetry, weight, PAW Gram and electron-count
checks. The default O 2s+2p branch and the explicit legacy O 2p branch define
different subspaces. Do not combine their column maps or silently relabel their
channel counts. Near-degenerate channels require subspace comparisons rather
than unique interpretations of individual singular vectors.

## 中文说明

本目录归档全晶胞 PAW–IAO 密度、指定原子及周期镜像的交叉块 SVD，以及真实空间
有符号通道场和查看器的构建代码。默认 O 2s+2p；自带算例保留四条数学通道。
物理比较必须保持原子模板、PAW、能带窗口及目标镜像定义一致。通道相干份额不等于键能。
