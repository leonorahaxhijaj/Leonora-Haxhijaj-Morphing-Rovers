
from morphing_udp import udp, Rover

chromosome = udp.example()

print("Number of parameters:", len(chromosome))

rover = Rover(chromosome)

distance, time_ratio = udp.run_single_scenario(rover, 0, 0)

fitness = (1 + distance) * time_ratio

print("Relative distance:", float(distance))
print("Relative time:", float(time_ratio))
print("Fitness:", float(fitness))

print("\nTesting all scenarios...")

result = udp.fitness(chromosome)

print("Average fitness:", result[0])
