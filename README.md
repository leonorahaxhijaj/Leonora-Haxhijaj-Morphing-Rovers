# Leonora-Haxhijaj-Morphing-Rovers

## Project Overview

This project is based on the ESA SpOC 2 - Morphing Rovers challenge.

The objective is to optimize a rover that can switch between four different configurations while navigating different terrains.

The rover is controlled by a neural network, and its performance is evaluated using a fitness function.

## Technologies

- Python 3.11
- NumPy
- PyTorch
- TorchVision
- Matplotlib
- ImageIO

## Implemented Features

- ESA rover simulation
- Single-scenario testing
- Evaluation across 30 scenarios
- Random Mutation / Hill Climbing optimization
- Saving the best rover configuration

## Initial Results

| Experiment | Fitness |
|------------|---------|
| ESA baseline | 7.9848 |
| Optimized rover | 7.4179 |

Improvement: 7.10%

Lower fitness values indicate better performance.

## Data

The simulation requires six terrain maps, coordinates.txt,
and example_rover.npy in the data/ directory.

These files are available from the original ESA SpOC2
repository under Morphing Rovers/data.

## Acknowledgements

The original challenge, simulation code, and datasets
were developed by the European Space Agency (ESA).

Original repository:
https://github.com/esa/SpOC2

The optimization experiments and scripts in this
repository are part of this project's development.


## Final Optimization Results

The ESA SpOC 2 - Morphing Rovers project evaluates
different optimization techniques for improving rover
performance across 30 simulation scenarios.

The following configurations were obtained through
successive optimization experiments:

| Configuration | Fitness |
|---------------|---------|
| ESA Baseline | 7.9848 |
| Hill Climbing | 7.4179 |
| Adaptive Hill Climbing | 7.3750 |
| Genetic Algorithm V1 | 7.3147 |
| Genetic Algorithm V2 | 7.2203 |
| Best Recovered GA Rover | 6.7131 |

The best configuration achieved a 15.93% reduction
in fitness compared to the original ESA baseline.

Lower fitness values indicate better performance.

## Algorithm Comparison

Three optimization methods were also compared
using the same initial ESA rover and six fitness
evaluations per algorithm in each run.

Each algorithm was tested with three random seeds.

| Algorithm | Average Fitness | Standard Deviation |
|-----------|-----------------|--------------------|
| Hill Climbing | 7.8001 | 0.1948 |
| Adaptive Hill Climbing | 7.5074 | 0.3113 |
| Genetic Algorithm | 7.3336 | 0.5735 |

Genetic Algorithm achieved the lowest average
fitness in these experiments, although its
results showed greater variability.

The comparison is preliminary because only
three independent runs were performed.

## Results and Visualizations

The repository includes:

- evaluation_results.csv
- comparison_results.csv
- comparison_summary.csv
- fitness_comparison.png
- evaluation_time.png
- comparison_summary.png

The best recovered rover configuration is saved
as ga_best_rover_run3.npy.

## Conclusion

The experiments demonstrate that optimization
methods can improve the rover's simulated
performance compared to the ESA baseline.

Genetic Algorithm produced the best observed
solution, while Adaptive Hill Climbing also
showed promising performance.

Further experiments with more independent runs
are needed for stronger statistical conclusions.
