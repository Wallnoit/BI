import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar el dataset
df = pd.read_csv('numerico/prediccionNumHijos.csv')

# Seleccionar solo las columnas numéricas
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Excluir la columna que no se debe normalizar
columna_no_normalizar = 'Pre_N_H'  # Reemplaza 'nombre_columna_a_excluir' con el nombre de la columna que no quieres normalizar
numeric_columns = numeric_columns.drop(columna_no_normalizar)

# Inicializar el escalador
scaler = MinMaxScaler()

# Ajustar y transformar solo las columnas numéricas
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Copiar la columna no normalizada al DataFrame normalizado
df[columna_no_normalizar] = df[columna_no_normalizar]

# Guardar el dataset normalizado
df.to_csv('normalizado/prediccionNumHijos.csv', index=False)

print('Dataset normalizado guardado en "tu_archivo_normalizado.csv"')
