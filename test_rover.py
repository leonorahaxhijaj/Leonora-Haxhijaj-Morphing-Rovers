
from morphing_udp import udp, Rover

# Marrim konfigurimin shembull te ESA-s
chromosome = udp.example()

print("Numri i parametrave:", len(chromosome))

# Krijojme roverin
rover = Rover(chromosome)

# Testojme harten e pare, skenarin e pare
distance, time_ratio = udp.run_single_scenario(rover, 0, 0)

# Llogarisim fitness
fitness = (1 + distance) * time_ratio

print("Distanca relative:", float(distance))
print("Koha relative:", float(time_ratio))
print("Fitness:", float(fitness))


# Testimi i plote ne 30 skenare
print("\nDuke testuar te gjithe skenaret...")

result = udp.fitness(chromosome)

print("Fitness mesatar:", result[0])
