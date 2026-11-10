# 基于 LOBSTER 投影的局域两轨道能级—成分—占据分析

**Tangtao Chen · TU Berlin**

本包对应四面板能级图：顶键 πx（dxz–px）、顶键 πy（dyz–py）、顶键 σ（dz²–pz）和底键 σ（Ru1–O93）。每个面板比较 OH 与竖直去 H；两态的重原子几何相同。方法与数值来自 2026-11-04 已实际重放的八组通道，本次仅独立整理计算与出图流程。

英文定位：Pair-local Hamiltonian analysis from LOBSTER projections: local levels, orbital character and projected populations.

## 包含什么、怎样复现图

- `analyze.py`：沿用已有、已核查的矩阵分析入口，未改动其数值实现。
- `plot_levels.py`：从原绘图流程独立提取的四面板出图脚本。
- `example/source_data.csv`：图上所有能级、成分和占据的完整精度数据；能量为 eV，占据为电子数 e，成分为无量纲份额。
- `example/results/`：八组原有逐 k 结果、数值检查及解释边界。
- `example/inputs/`：八组真实配置，含参考局部轴旋转后的 AO 系数。仅把机器上的绝对路径改成相对路径；数值参数与原记录一致。
- `figures/local_levels.png`、`.pdf`、`.svg`：预览及矢量图。
- `validation.json`：本次数据一致性核对、历史数值验收范围和出图检查摘要。

在本包根目录运行（Python ≥3.12；uv 按脚本声明安装依赖）：

```text
uv run plot_levels.py example/source_data.csv figures/local_levels
```

包内数据足以重新生成四面板图。上游 VASP 波函数和 LOBSTER 原始大矩阵不随此包重复分发；从矩阵起点重算时，需要自行提供对应的真实输入。

## 从计算到这张图

1. 准备对应几何的收敛 VASP 静态波函数，明确泛函、PAW、基组、能带范围、自旋与 k 网格。OH 与竖直去 H 的比较保持重原子几何及参考轴一致。
2. 用 LOBSTER 完成局域 AO 投影与矩阵导出，并核查投影质量。本实现限定已验证的 LOBSTER 5.1.1 文本约定、ISPIN=1、无 SOC／非共线。实际输入文件名与版本均须核对。
3. 读取同源的 `hamiltonMatrices.lobster`（正交表示 H⊥）、`overlapMatrices.lobster`（原始 AO 度量 S）、`coefficientMatricesLSO2.lobster`（与 H⊥ 一致的 C⊥）、`EIGENVAL` 和 `lobsterout`，另提供同源费米能。矩阵块按 k 点编号对应，不按文件出现顺序配对。
4. 固定局部轨道方向，恢复原始 AO 度量，对指定两个方向做局部 Löwdin 正交化，构造 2×2 哈密顿量。不能直接截取全局正交矩阵的两行两列。
5. 每个 k 点分别求局域低／高支、Ru／O 伙伴成分，并把完整晶体已占据 KS 态投影到两支上计算占据。
6. 按原始 k 权重分别平均每个物理量，保留逐 k 数据，然后按 CSV 出图。先平均哈密顿量再对角化会改变结果，不采用该顺序。

矩阵输入目录约定：将两态的上述五个输入分别放入 `example/matrices/OH/` 与 `example/matrices/deH/`。例如重算第一组通道：

```text
uv run analyze.py example/inputs/OH_top_pi_x.json recomputed_OH_top_pi_x.json
```

其余七组使用各自的配置和独立输出文件。输出 JSON 的 `summary` 与 `example/source_data.csv` 对应行具有同名字段；`k_resolved` 保留逐点读数。现有 `source_data.csv` 已由这八组结果逐字段核对。

## 核心数学定义

令 B 的两列是两个不同原子上、按既定局部轴构造的实 AO 方向。在每个 k 点：

\[
h=S^{1/2}H^\perp S^{1/2},\quad C=S^{-1/2}C^\perp,
\qquad Q=B(B^\dagger S B)^{-1/2},\quad Q^\dagger S Q=I.
\]

\[
h_2=Q^\dagger hQ=
\begin{pmatrix}\epsilon_a&t\\t^*&\epsilon_b\end{pmatrix},
\qquad h_2v_\pm=E_\pm v_\pm.
\]

\[
A=Q^\dagger SC,\quad D_2=A\operatorname{diag}(f)A^\dagger,
\qquad n_\pm=\sum_k w_k\,v_\pm^\dagger D_2v_\pm.
\]

f 是显式核对过自旋因子的 KS 电子占据；本例使用原配置的 `occupation_factor=2`，使电子数与 NELECT 一致。其他数据必须核对自己的占据约定。占据 n± 不能通过对局域 E± 套费米函数得到。

蓝线为局部正交化后继承 Ru 标签伙伴的对角能 εa；红线为 O 伙伴的 εb；黑线为低／高两支 E±。上支百分数为逐 k 的 |v[a,+]|² 的权重平均，电子数为上式投影占据。能级图的每条线都相对该态自身的 EF。虚线连接用于显示两轨道混合关系。

## 本图的局部坐标与对照

参考 OH 结构中，z 沿 Ru1→O113；x 沿 O113→H115 去掉 z 分量后归一化；y=z×x。去 H 态与底键使用相同参考轴，完整 AO 旋转系数已写入八份配置。

因此底键 σ 的精确定义是“顶键参考轴下的底键 dz²–pz 分量”。本图是逐 k 的 Bloch-AO 对分析，不提供指定周期镜像的实空间单键跃迁 H(R)。

## 验证与解释范围

现有八组真实矩阵重放均通过各自记录的数值门：Hermitian 性、占据态 Gram、电子数、H⊥ 的本征关系、局部谱重构、局部正交性、占据边界及 charge spilling。原有完整检查保留在 `example/results/`。

本例每态 24 个 k 点。已记录的逐项重放对照最大差为 4.8112474×10⁻⁷ eV（能量）与 1.5518372×10⁻⁷ e（占据）。打印的 k 权重和为 1.00000008；原入口保留原始打印权重，历史脚本将其归一化。统一该约定后差降到浮点舍入量级。本包保留原入口的数字，不重标定以贴合截图。

这里的低／高支是局域两轨道模型的 Ritz 能级；“成键／反键”只在该模型耦合项的意义下使用。它们不能直接解释为完整晶体的两条独立能带、整根键的键能或断键能。两态的自适应本征方向可以变化，因此上支占据变化也不能直接当作同一固定轨道的电子去除或力的因果分解。各态 EF 归零不构成跨态绝对能量对齐。

本图没有统计误差棒：它展示确定性的电子结构后处理和 k 权重平均，不是独立重复实验。数值检查通过不替代新基组／空带窗口的收敛验证，也不单独证明化学机制。

## 归档边界

本目录随统一方法仓库归档，署名为 Tangtao Chen，单位 TU Berlin，许可范围见仓库根目录 LICENSE。目前尚未上传 GitHub、Zenodo 或 arXiv。本包不包含 LOBSTER／VASP 程序、POTCAR 或大波函数；使用上游软件须遵守各自许可并引用其方法论文。

- LOBSTER 方法基础：[Maintz et al., J. Comput. Chem. 2016](https://doi.org/10.1002/jcc.24300)。
- 本包新增部分的定位：指定两轨道子空间、局部正交化、逐 k 能级与占据投影及其可复现图示流程。独创性需要相对于相关既有方法单独论证。
