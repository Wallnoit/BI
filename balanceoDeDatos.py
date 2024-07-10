import pandas as pd
from sklearn.utils import resample
import numpy as np

nombre_archivo = 'prediccionEstadoEmpleado.csv'

# Cargar el dataset
df = pd.read_csv('oneForEncoding/' + nombre_archivo)

# Verificar la distribución inicial
print("Distribución inicial de clases:")
print(df['pred_est_emp'].value_counts())

# Determinar el número total de registros
total_registros = len(df)

# Determinar el número de categorías
num_categorias = df['pred_est_emp'].nunique()

# Calcular la cantidad objetivo base de registros por categoría
cantidad_objetivo_base = total_registros // num_categorias

# Definir un rango de variación más pequeño (por ejemplo, ±500 registros)
variacion = 500

# Crear un diccionario para almacenar la cantidad objetivo ajustada por categoría
cantidad_objetivo_ajustada = {}

# Generar la cantidad objetivo ajustada para cada categoría
for categoria in df['pred_est_emp'].unique():
    cantidad_objetivo_ajustada[categoria] = cantidad_objetivo_base + np.random.randint(-variacion, variacion + 1)

# DataFrame para el dataset balanceado
df_balanced = pd.DataFrame()

for categoria in df['pred_est_emp'].unique():
    df_categoria = df[df['pred_est_emp'] == categoria]
    objetivo_ajustado = cantidad_objetivo_ajustada[categoria]
    if len(df_categoria) > objetivo_ajustado:
        # Submuestrear si hay más registros que el objetivo ajustado
        df_categoria_resampled = resample(df_categoria, replace=False, n_samples=objetivo_ajustado, random_state=42)
    else:
        # Sobremuestrear si hay menos registros que el objetivo ajustado
        df_categoria_resampled = resample(df_categoria, replace=True, n_samples=objetivo_ajustado, random_state=42)
    
    df_balanced = pd.concat([df_balanced, df_categoria_resampled]) # type: ignore

# Ajustar para tener exactamente el número de registros original
if len(df_balanced) > total_registros:
    df_balanced = df_balanced.sample(n=total_registros, random_state=42)
elif len(df_balanced) < total_registros:
    deficit = total_registros - len(df_balanced)
    df_supplement = resample(df_balanced, replace=True, n_samples=deficit, random_state=42)
    df_balanced = pd.concat([df_balanced, df_supplement]) # type: ignore

# Mezclar aleatoriamente las filas del dataset balanceado
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Verificar la nueva distribución y el número total de registros
print("Nueva distribución de clases:")
print(df_balanced['pred_est_emp'].value_counts())
print(f"Total de registros balanceados: {len(df_balanced)}")

# Guardar el dataset balanceado y mezclado
df_balanced.to_csv('DataSet-final/' + nombre_archivo, index=False)
