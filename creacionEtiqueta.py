import pandas as pd
import numpy as np

# Cargar los datos desde un archivo CSV
input_file = 'DataSet-BI/DataSet-FactCompras.csv'  # Reemplaza con el nombre de tu archivo CSV
data = pd.read_csv(input_file)

# Generar la columna 'etiqueta' con valores aleatorios de 1 y -1
np.random.seed(0)  # Fijar la semilla para reproducibilidad
data['etiqueta'] = np.random.choice([1, -1], size=len(data))

# Guardar el resultado en un nuevo archivo CSV
output_file = 'prediccionFactCompras.csv'  # Nombre del archivo de salida
data.to_csv(output_file, index=False)

print(f"Archivo guardado en {output_file}")
