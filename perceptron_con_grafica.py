import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

class Perceptron:
    '''Clasificador perceptron.'''
    def _init_(self, eta=0.5, n_iter=50, random_state=1):
        self.eta = eta  # Tasa de aprendizaje (entre 0.0 y 1.0)
        self.n_iter = n_iter  # Número de veces que va a pasar el conjunto de datos
        self.random_state = random_state  # Semilla del generador de número aleatorios

    def fit(self, X, y):
        '''Ajuste de los datos'''
        rgen = np.random.RandomState(self.random_state)  # Generamos número aleatorios
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # Número aleatorios con desviación estándar 0.01
        self.errores_ = []  # Lista vacía para los errores
        print('Pesos iniciales', self.w_)

        for epoch in range(self.n_iter):  # Ciclo que se repite según el número de iteraciones
            errores = 0
            for xi, etiqueta in zip(X, y):  # Ciclo que se repite según el número de muestras
                actualizacion = self.eta * (etiqueta - self.predice(xi))
                self.w_[1:] += actualizacion * xi
                self.w_[0] += actualizacion
                errores += int(actualizacion != 0)
            self.errores_.append(errores)
            print(f'Pesos en epoch {epoch + 1}: {self.w_}')
        return self

    def entrada_neta(self, X):
        '''Cálculo de la entrada neta'''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predice(self, X):
        '''Etiqueta de clase de retorno después del paso unitario'''
        return np.where(self.entrada_neta(X) >= 0.0, 1, -1)
    

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predice(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

# Cargar los datos
train_data1 = pd.read_csv("fold_1_train.csv")
train_data2 = pd.read_csv("fold_2_train.csv")
train_data3 = pd.read_csv("fold_3_train.csv")
train_data4 = pd.read_csv("fold_4_train.csv")
train_data5 = pd.read_csv("fold_5_train.csv")

test_data1 = pd.read_csv("fold_1_test.csv")
test_data2 = pd.read_csv("fold_2_test.csv")
test_data3 = pd.read_csv("fold_3_test.csv")
test_data4 = pd.read_csv("fold_4_test.csv")
test_data5 = pd.read_csv("fold_5_test.csv")

# Combinar datos de entrenamiento
train_data = pd.concat([train_data1, train_data2, train_data3, train_data4, train_data5])
X_train = train_data.drop(columns=['NombreGenero']).values
y_train = train_data['NombreGenero'].values

# Combinar datos de prueba
test_data = pd.concat([test_data1, test_data2, test_data3, test_data4, test_data5])
X_test = test_data.drop(columns=['NombreGenero']).values
y_test = test_data['NombreGenero'].values

# Convertir y_train y y_test a valores binarios si no lo son ya
unique_labels = np.unique(y_train)
if len(unique_labels) == 2:
    y_train = np.where(y_train == unique_labels[0], -1, 1)
    y_test = np.where(y_test == unique_labels[0], -1, 1)
else:
    raise ValueError("La columna de etiquetas debe tener exactamente 2 valores únicos para la clasificación binaria.")

# Reducir dimensiones con t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_train_2d = tsne.fit_transform(X_train)

# Crear y entrenar el perceptrón
ppn = Perceptron(eta=0.01, n_iter=1000)
ppn.fit(X_train_2d, y_train)

# Reducir dimensiones del conjunto de prueba con t-SNE
X_test_2d = tsne.fit_transform(X_test)
predictions = ppn.predice(X_test_2d)

# Calcular precisión
accuracy = np.mean(predictions == y_test) * 100
print(f"Precisión en el conjunto de prueba: {accuracy:.2f}%")

# Visualizar las regiones de decisión
plot_decision_regions(X_train_2d, y_train, classifier=ppn)
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.legend(loc='upper left')
plt.show()