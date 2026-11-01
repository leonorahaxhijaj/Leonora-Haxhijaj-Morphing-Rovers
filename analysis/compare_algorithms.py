import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "analysis" / "outputs"
INPUT = OUTPUTS / "evaluation_results.csv"
OUTPUT = OUTPUTS / "comparison_results.csv"


def main():
    if not INPUT.is_file():
        raise SystemExit("First run: python -m analysis.evaluate_results")

    with INPUT.open(newline="", encoding="utf-8-sig") as file:
        data = list(csv.DictReader(file))

    if len(data) != 5 or len({row["Algorithm"] for row in data}) != 5:
        raise SystemExit("Expected exactly 5 distinct algorithms in evaluation_results.csv")

    ranked = sorted(data, key=lambda row: float(row["Fitness"]))
    best = float(ranked[0]["Fitness"])

    output_rows = []
    print("Final configuration ranking (lower Fitness is better)")
    print("=" * 70)
    for position, item in enumerate(ranked, start=1):
        score = float(item["Fitness"])
        gap = 100 * (score - best) / best
        output_rows.append({
            "Rank": position,
            "Algorithm": item["Algorithm"],
            "Fitness": f"{score:.12f}",
            "Gap to Best (%)": f"{gap:.4f}",
        })
        print(f"{position}. {item['Algorithm']}: {score:.9f} | gap: {gap:.2f}%")

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "Rank", "Algorithm", "Fitness", "Gap to Best (%)"
        ])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nSaved: {OUTPUT.name}")
    print("This is a ranking of saved best configurations, not a fixed-budget algorithm benchmark.")


if __name__ == "__main__":
    main()
