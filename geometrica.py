import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import geom, mode

# Parámetro de la distribución geométrica
p = 0.08

# Tamaños de las muestras
sample_sizes = [10**2, 10**3, 10**4, 10**5]

# Generar muestras aleatorias
samples = [geom.rvs(p, size=size) for size in sample_sizes]

# a) Generar muestras aleatorias de tamaños 10^2, 10^3, 10^4 y 10^5
for size, sample in zip(sample_sizes, samples):
    print(f"Muestra de tamaño {size}:")
    print(sample[:10], '...')  # Mostrar los primeros 10 elementos para ilustrar

# b) Hacer un diagrama de cajas para cada una de las muestras generadas
plt.figure(figsize=(10, 6))
sns.boxplot(data=samples)
plt.xticks(ticks=np.arange(len(sample_sizes)), labels=sample_sizes)
plt.title("Diagrama de cajas para diferentes tamaños de muestras")
plt.xlabel("Tamaño de la muestra")
plt.ylabel("Valores")
plt.show()

# c) Realizar un histograma de las muestras generadas
plt.figure(figsize=(15, 10))
for i, sample in enumerate(samples):
    plt.subplot(2, 2, i+1)
    plt.hist(sample, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Histograma (Tamaño de muestra = {sample_sizes[i]})')
    plt.xlabel('Valores')
    plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# d) Hallar la mediana y la moda de cada muestra
for size, sample in zip(sample_sizes, samples):
    mediana = np.median(sample)
    moda_result = mode(sample)
    moda = moda_result.mode if isinstance(moda_result.mode, np.ndarray) else moda_result.mode.item()
    print(f"Tamaño de la muestra {size} - Mediana: {mediana}, Moda: {moda}")

# e) Hallar la media empírica de cada muestra y compararla con la esperanza teórica
esperanza_teorica = 1 / p
for size, sample in zip(sample_sizes, samples):
    media_empirica = np.mean(sample)
    print(f"Tamaño de la muestra {size} - Media empírica: {media_empirica}, Esperanza teórica: {esperanza_teorica}")

# f) Hallar la varianza empírica de cada muestra y compararla con la varianza teórica
varianza_teorica = (1 - p) / (p**2)
for size, sample in zip(sample_sizes, samples):
    varianza_empirica = np.var(sample)
    print(f"Tamaño de la muestra {size} - Varianza empírica: {varianza_empirica}, Varianza teórica: {varianza_teorica}")
