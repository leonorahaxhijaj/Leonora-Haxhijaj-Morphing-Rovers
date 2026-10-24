
import csv
import matplotlib.pyplot as plt

# Lexojme rezultatet nga CSV
algorithms = []
fitness_values = []
evaluation_times = []

with open("evaluation_results.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        algorithms.append(row["Algorithm"])
        fitness_values.append(float(row["Fitness"]))
        evaluation_times.append(
            float(row["Evaluation Time (s)"])
        )


# GRAFIKU 1 - Krahasimi i Fitness
plt.figure(figsize=(11, 6))

bars = plt.bar(
    algorithms,
    fitness_values,
    color="steelblue"
)

plt.title("ESA Morphing Rovers - Fitness Comparison")
plt.xlabel("Algorithm")
plt.ylabel("Fitness (Lower is Better)")
plt.xticks(rotation=15, ha="right")
plt.ylim(0, max(fitness_values) * 1.15)

for bar, value in zip(bars, fitness_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("fitness_comparison.png", dpi=300)
plt.close()


# GRAFIKU 2 - Koha e vleresimit
plt.figure(figsize=(11, 6))

bars = plt.bar(
    algorithms,
    evaluation_times,
    color="seagreen"
)

plt.title("ESA Morphing Rovers - Evaluation Time")
plt.xlabel("Algorithm Configuration")
plt.ylabel("Evaluation Time (seconds)")
plt.xticks(rotation=15, ha="right")
plt.ylim(0, max(evaluation_times) * 1.15)

for bar, value in zip(bars, evaluation_times):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{value:.2f}s",
        ha="center"
    )

plt.tight_layout()
plt.savefig("evaluation_time.png", dpi=300)
plt.close()

print("Graphs generated successfully!")
print("Saved: fitness_comparison.png")
print("Saved: evaluation_time.png")
