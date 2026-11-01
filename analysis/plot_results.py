import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "analysis" / "outputs"
INPUT = OUTPUTS / "evaluation_results.csv"


def make_chart(names, numbers, ylabel, title, output, suffix, precision):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = list(range(len(names)))
    bars = ax.bar(x, numbers)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(numbers) * 1.18)
    for bar, value in zip(bars, numbers):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(numbers) * 0.02,
                f"{value:.{precision}f}{suffix}", ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUTS / output, dpi=300)
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    if not INPUT.is_file():
        raise SystemExit("First run: python -m analysis.evaluate_results")

    with INPUT.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 5 or len({r["Algorithm"] for r in rows}) != 5:
        raise SystemExit("Expected exactly 5 distinct algorithms")

    labels = [row["Algorithm"] for row in rows]
    fitness = [float(row["Fitness"]) for row in rows]
    times = [float(row["Evaluation Time (s)"]) for row in rows]

    make_chart(
        labels, fitness, "Fitness (lower is better)",
        "ESA Morphing Rovers - Five Final Algorithms",
        "fitness_comparison.png", "", 4
    )
    make_chart(
        labels, times, "One complete ESA evaluation (seconds)",
        "ESA Morphing Rovers - Evaluation Time of Saved Solutions",
        "evaluation_time.png", " s", 2
    )
    print("Note: evaluation time is not total algorithm optimization time.")


if __name__ == "__main__":
    main()
