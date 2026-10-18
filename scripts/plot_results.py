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
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9, color=INK)


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
    fig.tight_layout()
    fig.savefig(OUT / "forecast_horizon.png", dpi=160, facecolor=SURFACE)


def planning_chart(name="planning"):
    res = json.load(open(RES / f"{name}.json"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for key, label, c in (("wm", "World-model cost", BLUE), ("static", "Static-world cost", ORANGE),
                          ("oracle", "GT-future cost (oracle)", AQUA)):
        pts = res["sweep"][key]
        xs = [p["col_obstacle"][3] for p in pts]
        ys = [p["l2"][3] for p in pts]
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 4), textcoords="offset points", fontsize=9, color=INK)
    p = res["overall"]["prior"]
    ax.plot([p["col_obstacle"][3]], [p["l2"][3]], "s", ms=7, color=INK2, mec=SURFACE, mew=1.5)
    ax.annotate("imitation prior only", (p["col_obstacle"][3], p["l2"][3]), xytext=(6, -12),
                textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("collision rate with any obstacle, avg 1-3 s (%)", fontsize=9, color=INK2)
    ax.set_title("L2 to logged trajectory, avg 1-3 s (m)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "planning_tradeoff.png", dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    forecast_chart()
    if (RES / "planning.json").exists():
        planning_chart()
