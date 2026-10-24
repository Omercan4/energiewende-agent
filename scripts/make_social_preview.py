"""Make docs/img/social-preview.png (1280 x 640): the card GitHub and LinkedIn show when the repo link is shared.

Run (needs matplotlib): python scripts/make_social_preview.py
The numbers come from eval/results.md (last run) and eval/questions.yaml, so the card matches the evaluation.
"""

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BG, CARD, LINE = "#0d1117", "#161b22", "#30363d"
FG, MUTED, BLUE, GREEN, ORANGE = "#e6edf3", "#8b949e", "#58a6ff", "#3fb950", "#d29922"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]


def last_run():
    """Numbers accuracy and 'all correct' of the last row of eval/results.md."""
    row = [r for r in (ROOT / "eval" / "results.md").read_text().splitlines() if r.startswith("| ") and "---" not in r][-1]
    cells = [c.strip() for c in row.strip("|").split("|")]
    return float(cells[3]), float(cells[5])


def box(ax, cx, cy, w, h, lines, edge=LINE, color=FG, size=11.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=10", fc=CARD, ec=edge, lw=1.6))
    ax.text(cx, cy, "\n".join(lines), color=color, fontsize=size, ha="center", va="center", linespacing=1.45)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.4))


def main():
    numbers, all_correct = last_run()
    questions = len(re.findall(r"^- id:", (ROOT / "eval" / "questions.yaml").read_text(), flags=re.M))

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280)
    ax.set_ylim(640, 0)
    ax.axis("off")

    ax.text(64, 70, "PORTFOLIO PROJECT  |  LLM AGENT WITH EVALUATION", color=BLUE, fontsize=13, fontweight="bold")
    ax.text(64, 135, "Energiewende Agent", color=FG, fontsize=38, fontweight="bold")
    subtitle = "An LLM agent that decides between live data, a price forecast and Bundestag papers, and names its sources."
    ax.text(64, 178, "\n".join(textwrap.wrap(subtitle, 46)), color=MUTED, fontsize=17, va="top", linespacing=1.4)

    stats = [
        (f"{questions}", "evaluation questions\ntracked in MLflow", BLUE),
        (f"{numbers * 100:.0f} %", "of numbers correct\n(computed by tools)", GREEN),
        (f"{all_correct * 100:.0f} %", "of answers fully\ncorrect (LLM judge)", GREEN),
    ]
    for i, (big, small, color) in enumerate(stats):
        x = 64 + i * 205
        ax.text(x, 372, big, color=color, fontsize=34, fontweight="bold")
        ax.text(x, 392, small, color=MUTED, fontsize=11.5, va="top", linespacing=1.4)

    ax.text(64, 520, "Python computes the numbers, the LLM explains them", color=FG, fontsize=13)
    ax.text(64, 590, "LangGraph  |  LlamaIndex  |  FastAPI  |  MLflow  |  Docker  |  Azure", color=MUTED, fontsize=13)
    ax.text(1216, 590, "github.com/Omercan4/energiewende-agent", color=MUTED, fontsize=12, ha="right")

    # the flow: question -> agent -> three kinds of tools -> answer with sources
    cx = 960
    box(ax, cx, 118, 190, 42, ["Question"], size=12)
    arrow(ax, cx, 140, cx, 172)
    box(ax, cx, 200, 330, 52, ["LangGraph agent", "the LLM picks the tools"], edge=BLUE, size=12)
    tools = [
        (800, ["Live data", "SMARD, weather", "numbers by Python"], GREEN),
        (960, ["Price forecast", "LightGBM model", "with 80 % interval"], ORANGE),
        (1120, ["Bundestag RAG", "99 papers, filters", "party, paper type"], BLUE),
    ]
    for x, lines, color in tools:
        arrow(ax, cx + (x - cx) * 0.55, 228, x, 292)
        box(ax, x, 335, 150, 90, lines, edge=color, size=10.5)
        arrow(ax, x, 382, cx + (x - cx) * 0.55, 446)
    box(ax, cx, 478, 330, 50, ["Answer + sources", "sources built by code, not by the LLM"], edge=GREEN, size=11)

    out = ROOT / "docs" / "img" / "social-preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    print("wrote", out)


if __name__ == "__main__":
    main()
