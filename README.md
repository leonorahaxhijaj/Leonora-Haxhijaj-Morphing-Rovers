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
