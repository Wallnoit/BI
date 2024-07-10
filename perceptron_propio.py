import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# %%
data = pd.read_csv('Test_set_1.csv')
data = np.array(data.iloc[:, :-1] )
data

# %%
clases = pd.read_csv('Test_set_1.csv')
clases = np.array(clases.iloc[:, -1])
clases

# %%
def activacion(pesos, x, b):
    z = pesos*x
    if z.sum() + b > 0:
        return 1
    else:
        return 0

# %%
n = data.shape[0];
n

# %%
#cantidad de caracteristicas
n = data.shape[1]
pesos = np.random.uniform(-1, 1, size=n)
b = np.random.uniform(-1, 1)
tasa_aprendizaje = 0.1
epocas = 100

for epoca in range(epocas):
    error_total = 0
    for i in range(data.shape[0]):
        prediccion = activacion(pesos, data[i], b)
        error = clases[i] - prediccion
        error_total += error**2
        for j in range(n):
          pesos[j] += tasa_aprendizaje * data[i][j] * error
        b += tasa_aprendizaje * error  
    print(error_total, end=" ")



activacion(pesos,[2,2,2,3,450,32,0,1,0,49,6,87,60.5,32,1000,40,7,2,5,9,3,423.5,7,74.55], b)