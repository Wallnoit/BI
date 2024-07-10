import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el dataset
data = pd.read_csv('prediccionEstadoEmpleado_2.csv')

# Nombre de la columna de etiquetas
columna_etiquetas = 'pred_est_emp'

# Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%), asegurando el balance de etiquetas
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42, stratify=data[columna_etiquetas])

# Verificar que los conjuntos estén balanceados
print(f"Distribución de etiquetas en el conjunto de entrenamiento:\n{train_set[columna_etiquetas].value_counts()}")
print(f"Distribución de etiquetas en el conjunto de prueba:\n{test_set[columna_etiquetas].value_counts()}")

# Guardar los conjuntos de entrenamiento y prueba en archivos CSV
nombre_archivo_train_set = 'dataset_entrenamiento_estadoEmpleados.csv'
train_set.to_csv(nombre_archivo_train_set, index=False)

nombre_archivo_test_set = 'dataset_prueba_estadoEmpleados.csv'
test_set.to_csv(nombre_archivo_test_set, index=False)
