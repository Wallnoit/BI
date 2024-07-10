import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generar un dataset linealmente separable con 21,000 muestras
n_samples = 21000
X, y = make_classification(n_samples=n_samples, n_features=8, n_informative=8, n_redundant=0, n_clusters_per_class=1, class_sep=2, random_state=42)

# Convertir etiquetas de 0 y 1 a -1 y 1
y = np.where(y == 0, -1, 1)

# Crear un DataFrame con las características especificadas
df = pd.DataFrame(X, columns=['valor_pago', 'Cantidad', 'subtotal', 'Total', 'numero_factura', 
                              'LimiteCreditoProveedor', 'DiasCreditoProveedor', 'CostoProducto'])

# Añadir columnas categóricas ficticiasZ
df['NombreProveedor'] = np.random.choice(['Proveedor A', 'Proveedor B', 'Proveedor C'], size=df.shape[0])
df['TelefonoProveedor'] = np.random.choice(['123456789', '987654321', '555555555'], size=df.shape[0])
df['RucProveedor'] = np.random.choice(['1234567890', '9876543210', '5555555555'], size=df.shape[0])
df['CiudadProveedor'] = np.random.choice(['Ciudad X', 'Ciudad Y', 'Ciudad Z'], size=df.shape[0])
df['DescripcionProducto'] = np.random.choice(['Producto A', 'Producto B', 'Producto C'], size=df.shape[0])
df['GrupoProducto'] = np.random.choice(['Grupo 1', 'Grupo 2', 'Grupo 3'], size=df.shape[0])
df['MarcaProducto'] = np.random.choice(['Marca A', 'Marca B', 'Marca C'], size=df.shape[0])
df['StockProducto'] = np.random.choice(['Stock A', 'Stock B', 'Stock C'], size=df.shape[0])
df['SubgrupoProducto'] = np.random.choice(['Subgrupo A', 'Subgrupo B', 'Subgrupo C'], size=df.shape[0])

# Añadir la columna de etiqueta
df['etiqueta'] = y

# Seleccionar las columnas de tipo objeto (texto)
text_columns = df.select_dtypes(include=['object']).columns

# Inicializar el OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Aplicar el OneHotEncoder a las columnas de texto seleccionadas
encoded_data = encoder.fit_transform(df[text_columns])

# Convertir el resultado a un DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(text_columns))

# Concatenar el DataFrame original sin las columnas de texto originales con el DataFrame codificado
df = pd.concat([df.drop(columns=text_columns).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Normalizar las características numéricas
scaler = MinMaxScaler()
numeric_columns = df.columns.difference(['etiqueta'])

# Aplicar la normalización
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Guardar el DataFrame resultante en un nuevo archivo CSV
df.to_csv('linearly_separable_dataset_21000.csv', index=False)

# Mostrar un resumen del DataFrame resultante
print(df.head())

# Visualizar las dos primeras características para verificar la separabilidad
plt.scatter(df['valor_pago'], df['Cantidad'], c=df['etiqueta'], cmap='bwr', alpha=0.7)
plt.xlabel('valor_pago')
plt.ylabel('Cantidad')
plt.title('Linearly Separable Dataset with 21,000 Samples')
plt.colorbar(label='Clase')
plt.show()
