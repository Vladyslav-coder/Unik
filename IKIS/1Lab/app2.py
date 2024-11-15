import numpy as np
import matplotlib.pyplot as plt

# Функція для генерації випадкових величин
def generate_normal_variables(n, mean, std):
    return np.random.normal(mean, std, n)

# Функція для генерації корельованих випадкових величин
def generate_correlated_variables(n, k, rv1, rv2):
    return k * rv1 + (1 - k) * rv2

# Параметри випадкових величин
n = 100
mean, std = 0, 1   # для нормального розподілу

# Випадкові величини
normal_rv = generate_normal_variables(n, mean, std)
uniform_rv = generate_normal_variables(n, mean, std)  # друга ВВ теж нормальна для кореляції

# Вибір значень k для нормального розподілу
k_values_normal = np.linspace(0, 1, 5)

# Кореляція та візуалізація
plt.figure(figsize=(10, 6))
for i, k in enumerate(k_values_normal):
    correlated_normal = generate_correlated_variables(n, k, normal_rv, uniform_rv)
    plt.scatter(normal_rv, correlated_normal, alpha=0.6, label=f'k={k:.2f}', color='orange')

plt.title('Correlation with Normal Distribution')
plt.xlabel('Normal RV')
plt.ylabel('Correlated Normal RV')
plt.legend()
plt.show()
