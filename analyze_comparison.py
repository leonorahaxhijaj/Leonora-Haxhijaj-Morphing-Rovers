
import csv
import statistics
import matplotlib.pyplot as plt

# ESA Morphing Rovers - Statistical Analysis

results = {}

# Lexojme rezultatet e 3 ekzekutimeve
with open("comparison_results.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        algorithm = row["Algorithm"]
        fitness = float(row["Fitness"])

        if algorithm not in results:
            results[algorithm] = []

        results[algorithm].append(fitness)


algorithms = []
averages = []
deviations = []

print("ESA Morphing Rovers - Statistical Analysis\n")

# Llogarisim mesataren dhe devijimin standard
for algorithm, values in results.items():

    average = statistics.mean(values)
    deviation = (
        statistics.stdev(values)
        if len(values) > 1 else 0.0
    )

    algorithms.append(algorithm)
    averages.append(average)
    deviations.append(deviation)

    print(f"Algorithm: {algorithm}")
    print(f"Runs: {len(values)}")
    print(f"Average Fitness: {average:.4f}")
    print(f"Standard Deviation: {deviation:.4f}")
    print(f"Best Fitness: {min(values):.4f}")
    print()


# Ruajme statistikat ne CSV
with open(
    "comparison_summary.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Algorithm",
        "Runs",
        "Average Fitness",
        "Standard Deviation",
        "Best Fitness"
    ])

    for i, algorithm in enumerate(algorithms):
        writer.writerow([
            algorithm,
            len(results[algorithm]),
            averages[i],
            deviations[i],
            min(results[algorithm])
        ])


# Krijojme grafikun krahasues
fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(
    algorithms,
    averages,
    yerr=deviations,
    capsize=6
)

# Vija e konfigurimit fillestar te ESA-s
ax.axhline(
    y=7.9848,
    linestyle="--",
    label="ESA Baseline"
)

ax.set_title(
    "ESA Morphing Rovers - Algorithm Comparison"
)
ax.set_ylabel("Average Fitness (Lower is Better)")
ax.set_xlabel("Optimization Algorithm")

ax.set_ylim(
    0,
    max(7.9848, max(
        avg + std
        for avg, std in zip(averages, deviations)
    )) + 1
)

for bar, value in zip(bars, averages):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{value:.4f}",
        ha="center"
    )

ax.legend()
plt.xticks(rotation=10, ha="right")
plt.tight_layout()

plt.savefig("comparison_summary.png", dpi=300)
plt.close()

print("Statistical analysis completed!")
print("Saved: comparison_summary.csv")
print("Saved: comparison_summary.png")

