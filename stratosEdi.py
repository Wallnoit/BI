import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# Cargar el dataset
file_path = 'generated_dataset.csv'
data = pd.read_csv(file_path)

# Columnas de géneros
genre_columns = [
    'NombreGenero_Blues', 'NombreGenero_Classical', 'NombreGenero_Country',
    'NombreGenero_Electronic', 'NombreGenero_HipHop', 'NombreGenero_Jazz',
    'NombreGenero_Metal', 'NombreGenero_Pop', 'NombreGenero_Reggae',
    'NombreGenero_Rock'
]

# Crear una sola columna 'Genero'
data['Genero'] = data[genre_columns].idxmax(axis=1).apply(lambda x: x.replace('NombreGenero_', ''))

# Extraer características y etiquetas
X = data.drop(columns=genre_columns + ['Genero'])
y = data['Genero']

# Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Crear un DataFrame con los resultados de t-SNE
tsne_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['Genero'] = y

# Graficar los resultados de t-SNE
plt.figure(figsize=(10, 7))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Genero', data=tsne_df, palette='bright')
plt.title('t-SNE visualization')
plt.legend(loc='best')
plt.show()