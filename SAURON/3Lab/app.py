import matplotlib.pyplot as plt
import numpy as np

class Car:
    def __init__(self, position, speed):
        self.position = position
        self.speed = speed

    def move(self, time):
        self.position += self.speed * time

def simulate_traffic(cars, time):
    positions = []
    for car in cars:
        car.move(time)
        positions.append(car.position)
    return positions

# Створення системи з п'ятьма автомобілями на різних швидкостях
cars = [Car(0, speed) for speed in np.linspace(5, 25, 5)]

# Симуляція руху на 10 хвилин
times = np.arange(0, 10, 0.5)  # кожні 0.5 хвилини
positions = [simulate_traffic(cars, 0.5) for _ in times]

# Візуалізація результатів
for i, car_positions in enumerate(zip(*positions)):
    plt.plot(times, car_positions, label=f'Car {i+1} Speed {5*(i+1)} km/h')

plt.title('Position of Cars Over Time')
plt.xlabel('Time (minutes)')
plt.ylabel('Position (km)')
plt.legend()
plt.show()
