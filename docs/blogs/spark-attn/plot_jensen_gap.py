"""Render the blog's exponential Jensen-gap illustration with Matplotlib."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(corrected=False):
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
    })
    blue, orange, ink = "#2364AA", "#C76A16", "#23364D"
    samples = np.array([-1.0, 2.0])
    values = np.exp(samples)
    mean_x = samples.mean()
    f_mean, mean_f = np.exp(mean_x), values.mean()

    fig, ax = plt.subplots(figsize=(9, 6.2), layout="constrained")
    fig.set_facecolor("white")
    x = np.linspace(-1.4, 2.13, 400)
    ax.plot(x, np.exp(x), color=blue, linewidth=3, zorder=3)
    ax.plot(samples, values, color=orange, linewidth=1.8, alpha=0.75)
    ax.scatter(samples, values, s=65, color=ink, zorder=5)
    ax.vlines(mean_x, 0, mean_f, colors="#ABB5C2", linestyles="dashed")
    ax.hlines([f_mean, mean_f], -1.3, mean_x,
              colors=[blue, orange], linestyles="dotted", linewidth=1.4)
    ax.scatter([mean_x, mean_x], [f_mean, mean_f],
               s=110, c=[blue, orange], edgecolors="white", linewidths=1.4, zorder=6)
    ax.annotate("", xy=(mean_x + 0.1, mean_f), xytext=(mean_x + 0.1, f_mean),
                arrowprops={"arrowstyle": "<->", "color": orange, "lw": 2})
    ax.text(mean_x + 0.22, (mean_f + f_mean) / 2,
            "Jensen gap", va="center", color=orange, weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    ax.text(-1.28, mean_f + 0.2,
            r"$\mathbb{E}[\exp(x)]$",
            color=orange, fontsize=15)
    ax.text(-1.28, f_mean + 0.2, r"$\exp(\mathbb{E}[x])$",
            color=blue, fontsize=15)
    ax.text(1.35, 6.85, r"$\exp(x)$", color=blue, fontsize=17)
    ax.annotate(r"$(x_1, \exp(x_1))$", (samples[0], values[0]),
                xytext=(-22, 22), textcoords="offset points", color=ink)
    ax.annotate(r"$(x_2, \exp(x_2))$", (samples[1], values[1]),
                xytext=(-12, 15), textcoords="offset points", ha="right", color=ink)
    ax.set_xticks([-1, mean_x, 2],
                  [r"$x_1=-1$", r"$\mathbb{E}[x]=0.5$", r"$x_2=2$"])
    if corrected:
        green = "#23866B"
        corrected_x = np.log(mean_f)
        weights = np.array([samples[1] - corrected_x, corrected_x - samples[0]])
        weights /= samples[1] - samples[0]
        assert np.isclose(weights.sum(), 1) and np.all(weights >= 0)
        assert np.isclose(np.exp(weights @ samples), mean_f)
        ax.hlines(mean_f, mean_x, corrected_x, colors=green,
                  linestyles="dashed", linewidth=1.6)
        ax.vlines(corrected_x, 0, mean_f, colors=green,
                  linestyles="dashed", linewidth=1.6)
        ax.scatter([corrected_x], [mean_f], s=90,
                   color=green, edgecolors="white", zorder=6)
        ax.text(corrected_x, -0.35, r"$\mathbb{E}_w[x]$", ha="center",
                va="top", color=green, fontsize=14)
        ax.text(-1.28, 5.7,
                r"$\exp(\mathbb{E}_w[x])=\mathbb{E}[\exp(x)]$",
                color=green, fontsize=14)
        ax.text(-1.28, 5.12,
                rf"$w_1={weights[0]:.3f},\quad w_2={weights[1]:.3f}$",
                color=green, fontsize=12)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_xlim(-1.4, 2.35)
    ax.set_ylim(0, 8.7)
    ax.set_xlabel("Logit, $x$", color=ink, labelpad=12)
    ax.set_ylabel("Exponentiated score", color=ink, labelpad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ABB5C2")
    ax.tick_params(colors=ink, length=0, pad=8)
    title = ("Reweighting the mean to match the expected exponential" if corrected
             else "Exponentiating the mean underestimates the mean exponential")
    ax.set_title(title,
                 loc="left", fontsize=15, color=ink, pad=24, weight="bold")
    footer = (rf"Calibrated weights:  $\mathbb{{E}}_w[x]={corrected_x:.3f},\quad"
              rf"\exp(\mathbb{{E}}_w[x])=\mathbb{{E}}[\exp(x)]={mean_f:.2f}$"
              if corrected else
              rf"Two equally weighted logits:  $\exp(\mathbb{{E}}[x])={f_mean:.2f}"
              rf"\;<\;\mathbb{{E}}[\exp(x)]={mean_f:.2f}$")
    fig.supxlabel(
        footer,
        fontsize=12, color=ink,
    )
    for extension in ("svg", "png"):
        name = "reweight_correction" if corrected else "jensen_gap"
        fig.savefig(out / f"{name}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
