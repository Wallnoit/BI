import numpy as np
import matplotlib.pyplot as plt

# Crear un rango de probabilidades
probabilities = np.linspace(0.01, 0.99, 100)

# Calcular los logaritmos de las probabilidades y de 1 - P
log_probabilities = np.abs(np.log(probabilities))
log_complement_probabilities = np.abs(np.log(1 - probabilities))

# Graficar los logaritmos de las probabilidades
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(probabilities, log_probabilities, label='|log(P)|', color='green')  # Cambio de color a verde
plt.title('Valor Absoluto del Logaritmo de la Probabilidad (|log(P)|)')
plt.xlabel('Probabilidad')
plt.ylabel('|log(P)|')
plt.grid(True)
plt.legend()

# Graficar los logaritmos de 1 - P
plt.subplot(1, 2, 2)
plt.plot(probabilities, log_complement_probabilities, label='|log(1 - P)|', color='purple')  # Cambio de color a morado
plt.title('Valor Absoluto del Logaritmo de 1 - Probabilidad (|log(1 - P)|)')
plt.xlabel('Probabilidad')
plt.ylabel('|log(1 - P)|')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
