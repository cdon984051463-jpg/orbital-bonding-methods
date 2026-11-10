# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""uv run plot_levels.py example/source_data.csv figures/local_levels"""
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def make_figure(source):
    with Path(source).open(encoding="utf-8-sig", newline="") as handle:
        rows = {(r["state"], r["channel"]): r for r in csv.DictReader(handle)}
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": .8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    blue, red = "#416FA8", "#B4483A"
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 5.4), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.185, top=.79, wspace=.24)
    fig.suptitle("Pair-local Hamiltonian: local levels and projected populations",
                 fontsize=11.5, y=.985)
    channels = ["top_pi_x", "top_pi_y", "top_sigma", "back_sigma"]
    titles = [r"Top $\pi_x$" + "\n" + r"$d_{xz}$–$p_x$",
              r"Top $\pi_y$" + "\n" + r"$d_{yz}$–$p_y$",
              r"Top $\sigma$" + "\n" + r"$d_{z^2}$–$p_z$",
              r"Bottom $\sigma$" + "\nRu1–O93"]
    for j, (axis, channel, title) in enumerate(zip(axes, channels, titles)):
        axis.set(xlim=(0, 1), ylim=(-6.7, 1.1), xticks=[], yticks=range(-6, 2))
        axis.axhline(0, color="#777777", lw=.7, ls="--")
        axis.set_title(title, pad=28, fontsize=10.5)
        axis.annotate("abcd"[j], (0, 1), xycoords="axes fraction",
                      xytext=(-29, 8), textcoords="offset points",
                      fontsize=11, weight="bold", va="bottom")
        for state, centre in [("OH", .24), ("deH", .76)]:
            row = rows[state, channel]
            r = {k: float(v) for k, v in row.items() if k not in ("state", "channel")}
            xa, xb = centre - .15, centre + .15
            ea, eb = r["epsilon_a_relative_EF_eV"], r["epsilon_b_relative_EF_eV"]
            lo, hi = r["E_lower_relative_EF_eV"], r["E_upper_relative_EF_eV"]
            for endpoint, energy in [(xa, ea), (xb, eb)]:
                for level in (lo, hi):
                    axis.plot([endpoint, centre], [energy, level],
                              color="#B0B5BA", lw=.7, ls=":")
            for x, energy, color in [(xa, ea, blue), (xb, eb, red),
                                      (centre, lo, "black"), (centre, hi, "black")]:
                axis.plot([x-.052, x+.052], [energy, energy], color=color, lw=1.8)
            axis.text(centre, lo-.20, f'{r["n_lower_e"]:.2f} e',
                      ha="center", va="top", fontsize=8)
            label_y = .21 if hi > -.85 else hi + .20
            axis.text(centre, label_y, f'{r["n_upper_e"]:.2f} e',
                      ha="center", va="bottom", fontsize=8)
            axis.text(centre, label_y+.35, f'{100*r["a_fraction_upper"]:.0f}% Ru',
                      ha="center", va="bottom", fontsize=8)
            axis.text(centre, 1.02, "OH" if state == "OH" else "de-H",
                      transform=axis.get_xaxis_transform(), ha="center",
                      va="bottom", fontsize=8.5)
    axes[0].set_ylabel(r"Energy relative to each $E_F$ (eV)")
    fig.legend(handles=[
        Line2D([], [], color=blue, lw=1.8, label="Ru d partner"),
        Line2D([], [], color=red, lw=1.8, label="O p partner"),
        Line2D([], [], color="black", lw=1.8, label="Lower / upper local branch"),
    ], loc="lower center", bbox_to_anchor=(.5, .105), ncol=3, fontsize=8)
    fig.text(.5, .075, "Numbers: projected populations; upper-branch percentages: Ru partner weight.",
             ha="center", fontsize=8)
    fig.text(.5, .035, "Per-k two-orbital Ritz levels, then k-weighted means; these are not isolated crystal bands.",
             ha="center", fontsize=8)
    return fig


if __name__ == "__main__":
    source, prefix = sys.argv[1:]
    fig = make_figure(source)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(f"{prefix}.{extension}", dpi=600)
    plt.close(fig)
