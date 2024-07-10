import pandas as pd

# Cargar el dataset CSV
df = pd.read_csv('DataSet-Bi-predicciones/prediccionNumHijos.csv')

# Iterar sobre cada columna del DataFrame
for columna in df.columns:
    # Verificar si la columna contiene datos de texto
    if df[columna].dtype == 'object':
        # Convertir los valores de texto a números utilizando codificación ordinal
        df[columna] = pd.factorize(df[columna])[0]+1

# Guardar el DataFrame modificado en un nuevo archivo CSV
df.to_csv('numerico/prediccionNumHijos.csv',index=False)