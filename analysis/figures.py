"""Compact paper figures: standalone panels, Times-style text, redundant line/hatch cues."""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = ("#32699c", "#238885", "#80518f")
STYLES = (("o", "-"), ("s", "--"), ("^", ":"))
POLICIES = ("hotspot", "residents", "environment", "joint")
POLICY_COLORS = ("#32699c", "#80518f", "#238885", "#b95833")
POLICY_NAMES = (
    "Hotspot Policing",
    "Resident Cooperation",
    "Environmental Improvement",
    "Joint Intervention",
)
HATCHES = ("//", "..", "xx", "\\\\")


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )


def save(fig, root, name):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / (name + ".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_pe(summary, output):
    style()
    levels = ("low", "medium", "high")
    filenames = {
        "abm_random": "pe_response_random_pe",
        "plain_llm": "pe_response_plain_llm",
        "crimemind": "pe_response_rat",
        "crimsense": "pe_response_crimsense",
        "no_action_filter": "ablation_no_action_filter",
        "rule_guardianship": "ablation_rule_guardianship",
        "merged_commit_target": "ablation_merged_commit_target",
    }
    for method, values in summary["equal_city_average"].items():
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        for p, color, (marker, linestyle) in zip(levels, COLORS, STYLES):
            cells = [values["cells"][p + "__" + e] for e in levels]
            y = np.array([r["mean"] for r in cells], float)
            ci = np.array(
                [
                    r["ci95"] if r["ci95"] is not None else [r["mean"], r["mean"]]
                    for r in cells
                ],
                float,
            )
            ax.errorbar(
                range(3),
                y,
                yerr=np.maximum(0, np.array([y - ci[:, 0], ci[:, 1] - y])),
                color=color,
                marker=marker,
                linestyle=linestyle,
                mfc="white",
                ms=5,
                lw=1.7,
                capsize=2,
            )
        ax.set(
            xticks=range(3),
            xticklabels=["Low", "Medium", "High"],
            ylim=(-2, 102),
            yticks=[0, 25, 50, 75, 100],
            xlabel="Safety Level",
            ylabel="Crime Choice Rate (%)",
        )
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        save(fig, output, filenames[method])
    fig = plt.figure(figsize=(5.4, 0.45))
    handles = [
        Line2D([0], [0], color=c, marker=m, ls=l, mfc="white", lw=1.7, label=p.title())
        for p, c, (m, l) in zip(levels, COLORS, STYLES)
    ]
    fig.legend(
        handles=handles, loc="center", ncol=3, frameon=False, title="Crime Tendency"
    )
    save(fig, output, "pe_shared_legend")


def plot_interventions(result, output):
    style()
    data = result["summary"]
    fig, ax = plt.subplots(figsize=(5.7, 3))
    zones = ("selected", "citywide", "adjacent", "rest")
    x = np.arange(4)
    for i, (arm, color, label, hatch) in enumerate(
        zip(POLICIES, POLICY_COLORS, POLICY_NAMES, HATCHES)
    ):
        cells = [data[arm]["spatial_change"][z] for z in zones]
        ax.bar(
            x + (i - 1.5) * 0.18,
            [v["mean"] for v in cells],
            width=0.17,
            yerr=[v["sd"] or 0 for v in cells],
            capsize=2,
            facecolor="white",
            edgecolor=color,
            hatch=hatch,
            lw=1.1,
            label=label,
        )
    ax.axhline(0, color="#666666", lw=0.9)
    ax.set(
        xticks=x,
        xticklabels=["Selected\nAreas", "Citywide", "Adjacent\nAreas", "Rest of\nCity"],
        ylabel="Crime Choices − Control (count)",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    save(fig, output, "rq3_policy_spatial")
    fig, ax = plt.subplots(figsize=(5.7, 2.8))
    control = data["control"]["windows"]
    w = result["window"]
    x = (np.arange(len(control)) + 0.5) * w + 0.5
    all_arms = ("control",) + POLICIES
    for arm, color, label, (marker, linestyle) in zip(
        all_arms,
        ("#75818b",) + POLICY_COLORS,
        ("No Intervention",) + POLICY_NAMES,
        (("o", "-"), ("s", "--"), ("D", ":"), ("^", "-."), ("v", (0, (3, 1, 1, 1)))),
    ):
        values = data[arm]["windows"]
        ax.errorbar(
            x,
            [v["mean"] for v in values],
            yerr=[v["sd"] or 0 for v in values],
            color=color,
            label=label,
            marker=marker,
            ls=linestyle,
            lw=1.5,
            ms=4,
            mfc="white",
            capsize=2,
        )
    ax.axvline(result["activation_step"] + 0.5, color="#888888", lw=1, ls="--")
    ax.set(xlabel="Steps", ylabel="Crime Choice Rate (%)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save(fig, output, "rq3_policy_results")
