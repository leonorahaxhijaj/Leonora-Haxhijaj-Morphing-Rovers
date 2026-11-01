import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "analysis" / "outputs"
INPUT = OUTPUTS / "comparison_results.csv"
CSV_OUTPUT = OUTPUTS / "comparison_summary.csv"
CHART_OUTPUT = OUTPUTS / "comparison_summary.png"


def main():
    if not INPUT.is_file():
        raise SystemExit("First run: python -m analysis.compare_algorithms")

    with INPUT.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 5:
        raise SystemExit("Expected 5 algorithms in comparison_results.csv")

    baseline_values = [float(item["Fitness"]) for item in rows
                       if item["Algorithm"] == "Hill Climbing"]
    if len(baseline_values) != 1:
        raise SystemExit("Hill Climbing entry is missing or duplicated")
    baseline = baseline_values[0]

    output_rows = []
    for item in rows:
        score = float(item["Fitness"])
        change = 100 * (baseline - score) / baseline
        output_rows.append({
            "Rank": item["Rank"],
            "Algorithm": item["Algorithm"],
            "Fitness": f"{score:.12f}",
            "Improvement vs Hill Climbing (%)": f"{change:.4f}",
        })

    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "Rank", "Algorithm", "Fitness", "Improvement vs Hill Climbing (%)"
        ])
        writer.writeheader()
        writer.writerows(output_rows)

    labels = [r["Algorithm"] for r in output_rows][::-1]
    values = [float(r["Improvement vs Hill Climbing (%)"])
              for r in output_rows][::-1]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, values)
    ax.set_title("ESA Morphing Rovers - Improvement over Hill Climbing")
    ax.set_xlabel("Fitness improvement (%) - higher is better")
    ax.set_xlim(0, max(values) * 1.20 + 1)
    for bar, val in zip(bars, values):
        ax.text(val + 0.25, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center")
    fig.tight_layout()
    fig.savefig(CHART_OUTPUT, dpi=300)
    plt.close(fig)

    print("Saved: comparison_summary.csv")
    print("Saved: comparison_summary.png")
    print("Percentages compare final saved solutions, not repeated optimization runs.")


if __name__ == "__main__":
    main()
