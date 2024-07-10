import pandas as pd
import numpy as np

# Cargar el dataset existente
dataset_existente = pd.read_csv('prediccionFactCompras.csv')

# Generar etiquetas aleatorias (1 o -1)
etiquetas = np.random.choice([-1, 1], size=len(dataset_existente))

# Agregar las etiquetas al dataset existente
dataset_existente['Etiqueta'] = etiquetas

# Guardar el dataset con las nuevas etiquetas
dataset_existente.to_csv('prediccionFactComprasE.csv', index=False)

print("Etiquetas agregadas exitosamente.")
