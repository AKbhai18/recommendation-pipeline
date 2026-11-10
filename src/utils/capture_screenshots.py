"""Capture execution "screenshots" for each pipeline stage.

The build environment is headless (no desktop / no browser), so instead of
photographing a terminal window we render the REAL console output of each stage
into a terminal-styled PNG. Each image is produced by actually executing the
stage as a subprocess and capturing its stdout+stderr, so the screenshots
reflect a genuine run (not mock text).

Images are written to ``reports/screenshots/`` and later embedded in the final
PDF report.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SHOTS = ROOT / "reports" / "screenshots"

# Terminal colour theme.
TERM_BG = "#0b1021"
TERM_FG = "#e2e8f0"
TERM_PROMPT = "#4ade80"
TERM_TITLEBAR = "#1e293b"


def _run(cmd: list) -> str:
    """Run a command from the project root, return combined stdout+stderr."""
    proc = subprocess.run(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.stdout


def _render_terminal(title: str, command: str, output: str, out_path: Path,
                     max_lines: int = 46) -> None:
    lines = output.rstrip("\n").split("\n")
    # keep the most informative tail if the output is very long
    if len(lines) > max_lines:
        lines = ["... (earlier output truncated) ..."] + lines[-(max_lines - 1):]

    # wrap long lines so nothing runs off the image
    wrapped: list = []
    for ln in lines:
        wrapped.extend(textwrap.wrap(ln, width=118) or [""])
    body = wrapped[:max_lines]

    n = len(body) + 4
    fig_h = max(3.2, 0.26 * n + 0.8)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor(TERM_BG)
    ax.set_facecolor(TERM_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # title bar with the classic three dots
    ax.add_patch(plt.Rectangle((0, 0.955), 1, 0.045, color=TERM_TITLEBAR,
                               transform=ax.transAxes, zorder=1))
    for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        ax.add_patch(plt.Circle((0.015 + i * 0.016, 0.977), 0.006,
                                color=c, transform=ax.transAxes, zorder=2))
    ax.text(0.5, 0.977, title, ha="center", va="center",
            color="#94a3b8", fontsize=10, family="monospace",
            transform=ax.transAxes, zorder=2)

    y = 0.92
    dy = 0.9 / max(len(body) + 2, 1)
    ax.text(0.01, y, "$ ", color=TERM_PROMPT, fontsize=10.5,
            family="monospace", va="top", transform=ax.transAxes)
    ax.text(0.035, y, command, color="#f8fafc", fontsize=10.5,
            family="monospace", va="top", transform=ax.transAxes)
    y -= dy * 1.4
    for ln in body:
        colour = TERM_FG
        low = ln.lower()
        if "error" in low or "fail" in low or "invalid" in low or "quarantin" in low:
            colour = "#fca5a5"
        elif "success" in low or "passed" in low or "succeeded" in low or "done" in low:
            colour = "#86efac"
        elif ln.strip().startswith("==="):
            colour = "#60a5fa"
        ax.text(0.01, y, ln, color=colour, fontsize=9.5,
                family="monospace", va="top", transform=ax.transAxes)
        y -= dy

    SHOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=TERM_BG)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


STAGES = [
    ("01_ingestion", "Stage 1 - Data Ingestion",
     "python -m src.ingestion.ingest",
     [sys.executable, "-m", "src.ingestion.ingest"]),
    ("02_validation", "Stage 2 - Data Validation",
     "python -m src.validation.validate",
     [sys.executable, "-m", "src.validation.validate"]),
    ("03_preparation", "Stage 3 & 4 - Data Preparation and Transformation",
     "python -m src.preparation.prepare",
     [sys.executable, "-m", "src.preparation.prepare"]),
    ("04_features", "Stage 5 - Feature Management",
     "python -m src.features.build_features",
     [sys.executable, "-m", "src.features.build_features"]),
    ("05_training_evaluation", "Stage 6 & 7 - Model Training and Evaluation",
     "python -m src.training.train",
     [sys.executable, "-m", "src.training.train"]),
    ("06_orchestration", "Stage 8 - Orchestration (full DAG run)",
     "python -m src.orchestrate",
     [sys.executable, "-m", "src.orchestrate"]),
]


def main():
    for key, title, shown_cmd, cmd in STAGES:
        print(f"Running: {shown_cmd}")
        output = _run(cmd)
        _render_terminal(title, shown_cmd, output, SHOTS / f"{key}.png")


if __name__ == "__main__":
    main()
