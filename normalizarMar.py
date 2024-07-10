
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Asegúrate de que la ruta del archivo sea correcta
file_path = 'testing.csv'
df = pd.read_csv(file_path)

# Estandarizar los datos
scaler = StandardScaler()
df_encoded_scaled = scaler.fit_transform(df_encoded) # type: ignore

# Normalizar los datos
min_max_scaler = MinMaxScaler()
df_encoded_normalized = min_max_scaler.fit_transform(df_encoded_scaled)

# Aplicar t-SNE con menos iteraciones
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(df_encoded_normalized)

# Crear un DataFrame con los resultados de t-SNE
tsne_df = pd.DataFrame(X_tsne, columns=['TSNE_Component_1', 'TSNE_Component_2'])
tsne_df['genre'] = y.values # type: ignore

# Verificar el DataFrame antes de exportar
print(tsne_df.head())

# Exportar el DataFrame resultante a un archivo CSV
output_file_path = 'encoded_and_tsne_results.csv'
tsne_df.to_csv(output_file_path, index=False, sep=',')
print(f'El archivo CSV resultante ha sido guardado en {output_file_path}')

# Adicionalmente, si necesitas el DataFrame combinado antes de t-SNE:
# Guardar el DataFrame combinado y normalizado antes de t-SNE
combined_df = pd.DataFrame(df_encoded_normalized, columns=df_encoded.columns) # type: ignore
combined_df['genre'] = y.values # type: ignore
combined_output_file_path = 'combined_encoded_data.csv'
combined_df.to_csv(combined_output_file_path, index=False, sep=',')
print(f'El archivo CSV combinado ha sido guardado en {combined_output_file_path}')