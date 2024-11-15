import numpy as np
import matplotlib.pyplot as plt

# Функція для генерації випадкових величин
def generate_uniform_variables(n, a, b):
    return np.random.uniform(a, b, n)

# Функція для генерації корельованих випадкових величин
def generate_correlated_variables(n, k, rv1, rv2):
    return k * rv1 + (1 - k) * rv2

# Параметри випадкових величин
n = 100
a, b = 0, 1  # для рівномірного розподілу

# Випадкові величини
uniform_rv = generate_uniform_variables(n, a, b)
normal_rv = generate_uniform_variables(n, a, b)  # друга ВВ теж рівномірна для кореляції

# Вибір значень k для рівномірного розподілу
k_values_uniform = np.linspace(0.5, 1, 5)

# Кореляція та візуалізація
plt.figure(figsize=(10, 6))
for i, k in enumerate(k_values_uniform):
    correlated_uniform = generate_correlated_variables(n, k, uniform_rv, normal_rv)
    plt.scatter(uniform_rv, correlated_uniform, alpha=0.6, label=f'k={k:.2f}')

plt.title('Correlation with Uniform Distribution')
plt.xlabel('Uniform RV')
plt.ylabel('Correlated Uniform RV')
plt.legend()
plt.show()
