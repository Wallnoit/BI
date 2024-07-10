
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Cargar datos normalizados desde el archivo CSV
data = pd.read_csv("oneForEncoding/prediccionCiudadProveedor.csv")

# Extraer características (X) y etiquetas (y) del DataFrame
X = data.drop(columns=['Prediccion_Ciudad_Prov']).values
y = data['Prediccion_Ciudad_Prov'].values

# Inicializar y ajustar t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Crear un mapeo de colores para las clases
unique_labels = np.unique(y)  # type: ignore # Obtener etiquetas únicas
colors = plt.cm.get_cmap('tab10', len(unique_labels))  # Crear un colormap con suficientes colores

# Mapear cada etiqueta única a un color
color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

# Asignar colores a los puntos según sus etiquetas
point_colors = [color_map[label] for label in y]

# Graficar los resultados de t-SNE
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=point_colors)
plt.title('Visualización de t-SNE con datos normalizados')
plt.xlabel('Componente 1 de t-SNE')
plt.ylabel('Componente 2 de t-SNE')

# Crear una leyenda personalizada
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=str(label))
           for label, color in color_map.items()]
plt.legend(title='Clase', handles=handles)

plt.show()
