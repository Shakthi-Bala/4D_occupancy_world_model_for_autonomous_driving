"""Static result charts for the README from results/*.json.

    python scripts/plot_results.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES, OUT = Path("results"), Path("assets")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

# Published numbers (Occ3D-nuScenes val, GT occupancy input), mIoU / IoU at 1 / 2 / 3 s.
PAPERS = {
    "DOME-O": {"miou": [35.11, 25.89, 20.29], "iou": [43.99, 35.36, 29.74], "color": AQUA},
    "OccWorld-O": {"miou": [25.78, 15.14, 10.51], "iou": [34.63, 25.07, 20.18], "color": YELLOW},
}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def end_label(ax, x, y, text, color):
    ax.plot([x], [y], "o", ms=5, color=color, mec=SURFACE, mew=1.5, zorder=5)
    ax._end_labels = getattr(ax, "_end_labels", []) + [(x, y, text)]


def place_end_labels(ax, min_gap=0.055):
    """Direct labels at line ends, nudged apart vertically so they never overlap."""
    items = sorted(getattr(ax, "_end_labels", []), key=lambda t: t[1])
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ys = [y for _, y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap * (y1 - y0))
    for (x, _, text), ty in zip(items, ys):
        ax.text(x + 0.015 * (x1 - x0), ty, text, va="center", fontsize=9, color=INK)


def forecast_chart():
    ours = json.load(open(RES / "forecast_ours.json"))
    warp = json.load(open(RES / "forecast_warp.json"))
    copy = json.load(open(RES / "forecast_copy.json"))
    hz = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), facecolor=SURFACE)
    for ax, key, title in zip(axes, ("miou", "iou"), ("Semantic mIoU (%)", "Occupancy IoU (%)")):
        style(ax)
        for name, r, c in (("Ours", ours, BLUE), ("Ego-warp", warp, ORANGE), ("Copy last frame", copy, MAGENTA)):
            ax.plot(hz, r[key], color=c, lw=2.2 if name == "Ours" else 2)
            end_label(ax, hz[-1], r[key][-1], name, c)
        for name, p in PAPERS.items():
            ax.plot([1, 2, 3], p[key], color=p["color"], lw=2, ls=(0, (4, 2)), marker="o", ms=5,
                    mec=SURFACE, mew=1.5)
            end_label(ax, 3.0, p[key][-1], name + " (paper)", p["color"])
        ax.set_xlim(0.4, 3.9)
        ax.set_xticks(hz)
        ax.set_xticklabels([f"{h:g}s" for h in hz])
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("forecast horizon", fontsize=9, color=INK2)
        place_end_labels(ax)
    fig.tight_layout()
    fig.savefig(OUT / "forecast_horizon.png", dpi=160, facecolor=SURFACE)


def planning_chart():
    ci = json.load(open(RES / "planning_ci.json"))
    rows = [("prior+oracle", "GT future (oracle)"), ("prior+wm", "World model (ours)"),
            ("prior+static", "Static world"), ("prior", "Imitation prior only")]
    panels = (("col_obst", "Collision with any obstacle (%)"), ("col_agent", "Collision with an agent (%)"),
              ("off", "Off-road (%)"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.6), facecolor=SURFACE, sharey=True)
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for i, (m, _) in enumerate(rows):
            v, (lo, hi) = ci[m][key]["mean"], ci[m][key]["ci95"]
            c = BLUE if m == "prior+wm" else ("#c9c8c2" if m == "prior+oracle" else "#8d8c86")
            ax.barh(i, v, height=0.62, color=c)
            ax.plot([lo, hi], [i, i], color=INK, lw=1.2)
            ax.text(hi + 0.01 * max(r[key]["ci95"][1] for r in ci.values()), i, f"{v:.2f}", va="center",
                    fontsize=8.5, color=INK)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlim(0, 1.18 * max(ci[m][key]["ci95"][1] for m, _ in rows))
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([label for _, label in rows], fontsize=9, color=INK)
    fig.text(0.01, 0.01, "Same 32 candidate trajectories per scene; only the safety cost changes. "
             "Lines: 95% bootstrap CI over 4,519 val scenes.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "planning_safety.png", dpi=160, facecolor=SURFACE)


def ablation_chart():
    curves = json.load(open(RES / "training_curves.json"))
    warp = json.load(open(RES / "forecast_warp.json"))["miou_1s_2s_3s_avg"][3]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    style(ax)
    for key, label, c in (("wm", "Warp + residual (ours)", BLUE), ("wm_nowarp", "No warp (adaLN only)", ORANGE)):
        ep = [r["epoch"] for r in curves[key]]
        v = [r["val_miou_avg"] for r in curves[key]]
        ax.plot(ep, v, color=c, lw=2)
        ax.plot([ep[-1]], [v[-1]], "o", ms=5, color=c, mec=SURFACE, mew=1.5, zorder=5)
        ax.text(ep[-1] + 0.4, v[-1], label, va="center", fontsize=9, color=INK)
    for y, label in ((warp, "ego-warp, 0 parameters"), (27.10, "DOME-O (paper)")):
        ax.axhline(y, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.annotate(label, (1, y), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlim(0.5, 33)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("training epoch", fontsize=9, color=INK2)
    ax.set_title("Forecast mIoU, avg 1-3 s (%)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_warp.png", dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    forecast_chart()
    ablation_chart()
    if (RES / "planning_ci.json").exists():
        planning_chart()
