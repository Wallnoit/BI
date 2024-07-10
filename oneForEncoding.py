import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Leer el dataset desde un archivo CSV
df = pd.read_csv('DataSet-Bi-predicciones/prediccionFactCompras.csv')

# Definir la columna que queremos ignorar para la codificación y normalización
ignore_column = 'etiqueta'

# Verificar que la columna a ignorar existe y eliminarla temporalmente del DataFrame
if ignore_column in df.columns:
    etiqueta_column = df[ignore_column]
    df = df.drop(columns=[ignore_column])
else:
    raise KeyError(f"'{ignore_column}' not found in DataFrame columns")

# Seleccionar solo las columnas de tipo objeto (texto)
text_columns = df.select_dtypes(include=['object']).columns

# Inicializar el OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Aplicar el OneHotEncoder a las columnas de texto seleccionadas
encoded_data = encoder.fit_transform(df[text_columns])

# Convertir el resultado a un DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(text_columns))

# Concatenar el DataFrame original sin las columnas de texto originales con el DataFrame codificado
df = pd.concat([df.drop(columns=text_columns).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Rellenar los datos nulos con el valor de 0.001
df = df.fillna(0.001)

# Normalizar los datos excepto la columna 'etiqueta'
scaler = MinMaxScaler()
numeric_columns = df.columns

# Aplicar la normalización
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Volver a agregar la columna 'etiqueta'
df[ignore_column] = etiqueta_column.values

# Guardar el DataFrame resultante en un nuevo archivo CSV
df.to_csv('oneForEncoding/prediccionFactCompras.csv', index=False)

# Mostrar un resumen del DataFrame resultante
print(df.head())
