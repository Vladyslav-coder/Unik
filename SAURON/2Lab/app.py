import numpy as np
import matplotlib.pyplot as plt

# Функція для створення синусоїди
def generate_sine_wave(freq, sample_rate, end_time):
    t = np.linspace(0, end_time, int(sample_rate * end_time), endpoint=False)
    sine_wave = np.sin(2 * np.pi * freq * t)
    return t, sine_wave

# Функція для квантування сигналу
def quantize_signal(signal, num_levels):
    min_val = np.min(signal)
    max_val = np.max(signal)
    step = (max_val - min_val) / num_levels
    quantized_signal = np.round((signal - min_val) / step) * step + min_val
    return quantized_signal, step

# Параметри синусоїди
freq = 1 / (2 * np.pi)  # Частота, щоб період був 2*pi (один повний цикл sin за 2*pi)
end_time = 15  # Кінець діапазону для x
sample_rate = 1000  # Частота дискретизації

t, sine_wave = generate_sine_wave(freq, sample_rate, end_time)

# Квантування синусоїди
num_levels = 10  # Кількість рівнів квантування
quantized_signal, step = quantize_signal(sine_wave, num_levels)

# Візуалізація
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, sine_wave, label='Original Sine Wave')
plt.title('Original Sine Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(t, quantized_signal, label='Quantized Sine Wave', color='orange')
plt.title('Quantized Sine Wave')
plt.xlabel('Time [s]')
plt.ylabel('Quantized Amplitude')

plt.tight_layout()
plt.show()
