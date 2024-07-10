import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron:
    '''Clasificador perceptron.'''
    def __init__(self, eta=0.5, n_iter=50, random_state=1):
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

# Load the data
train_data1 = pd.read_csv("dataset_entrenamiento_compras.csv")

test_data1 = pd.read_csv("dataset_prueba_compras.csv")
# Combine training data
train_data = pd.concat([train_data1])
X_train = train_data.drop(['etiqueta'], axis=1).values
y_train = train_data['etiqueta'].values

# Combine test data
test_data = pd.concat([test_data1])
X_test = test_data.drop(['etiqueta'], axis=1).values
y_test = test_data['etiqueta'].values

# Convert y_train and y_test to binary values if they are not already
unique_labels = np.unique(y_train)
if len(unique_labels) == 2:
    y_train = np.where(y_train == unique_labels[0], -1, 1)
    y_test = np.where(y_test == unique_labels[0], -1, 1)
else:
    raise ValueError("The label column must have exactly 2 unique values for binary classification.")

# Create and train the perceptron
ppn = Perceptron(eta=0.01, n_iter=1000)
ppn.fit(X_train, y_train)

# Predictions on the test set
predictions = ppn.predice(X_test)

# Calculate accuracy
accuracy = np.mean(predictions == y_test) * 100
print(f"Precisión en el conjunto de prueba: {accuracy:.2f}%")

# Visualize decision regions (optional, for 2D data only)
if X_train.shape[1] == 2:
    plot_decision_regions(X_train, y_train, classifier=ppn)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='upper left')
    plt.show()