import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

class Perceptron:
    '''Clasificador perceptron.'''
    def __init__(self, tasa_aprendizaje=0.5, epocas=50, random_state=1):
        self.tasa_aprendizaje = tasa_aprendizaje  # Tasa de aprendizaje (entre 0.0 y 1.0)
        self.epocas = epocas  # Número máximo de iteraciones
        self.random_state = random_state  # Semilla del generador de número aleatorios

    def fit(self, X, y):
        '''Ajuste de los datos'''
        rgen = np.random.RandomState(self.random_state)  # Generamos número aleatorios
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # Número aleatorios con desviación estándar 0.01
        self.errores_ = []  # Lista vacía para los errores
        print('Pesos iniciales', self.w_)
        
        for epoch in range(self.epocas):  # Ciclo que se repite según el número de épocas
            errores = 0

            for xi, etiqueta in zip(X, y):  # Ciclo que se repite según el número de muestras
                actualizacion = self.tasa_aprendizaje * (etiqueta - self.predice(xi))
                self.w_[1:] += actualizacion * xi
                self.w_[0] += actualizacion
                errores += int(actualizacion != 0)
            
            self.errores_.append(errores)
            print(f'Pesos en época {epoch + 1}: {self.w_}')

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

# Lista de archivos de entrenamiento y prueba
train_files = ['Train_set_1.csv', 'Train_set_2.csv', 'Train_set_3.csv']
test_files = ['Test_set_1.csv', 'Test_set_2.csv', 'Test_set_3.csv']

# Seleccionar dos categorías para la clasificación binaria
selected_categories = [1, -1]  # Cambiar según las categorías de interés

# Iterar sobre los archivos y entrenar/probar el modelo
for train_file, test_file in zip(train_files, test_files):
    # Cargar los datos de entrenamiento y prueba
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Seleccionar la etiqueta
    label = 'pred_est_emp'

    # Filtrar los datos para incluir solo las dos categorías seleccionadas
    train_data = train_data[train_data[label].isin(selected_categories)]
    test_data = test_data[test_data[label].isin(selected_categories)]

    # Verificar los valores únicos en las etiquetas
    unique_labels = train_data[label].unique()
    print(f"Valores únicos en {train_file}: {unique_labels}")

    if len(unique_labels) != 2:
        raise ValueError(f"La columna de etiquetas en {train_file} debe tener exactamente 2 valores únicos para la clasificación binaria.")
    
    # Convertir etiquetas categóricas a etiquetas binarias
    train_data[label] = np.where(train_data[label] == unique_labels[0], -1, 1)
    test_data[label] = np.where(test_data[label] == unique_labels[0], -1, 1)

    # Seleccionar solo las características numéricas
    numeric_features = train_data.select_dtypes(include=[np.number]).columns.tolist()
    X_train = train_data[numeric_features].values
    y_train = train_data[label].values
    X_test = test_data[numeric_features].values
    y_test = test_data[label].values

    # Imputar valores faltantes
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Reducción de dimensionalidad a 2D utilizando PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Crear y entrenar el perceptrón
    ppn = Perceptron(tasa_aprendizaje=0.01, epocas=50)  # Iterar 50 veces en cada conjunto de datos
    ppn.fit(X_train_pca, y_train)

    # Predicciones en el conjunto de prueba
    predictions = ppn.predice(X_test_pca)

    # Calcular precisión
    accuracy = np.mean(predictions == y_test) * 100
    print(f"Precisión para el archivo {test_file}: {accuracy:.2f}%")

    # Visualizar regiones de decisión
    plot_decision_regions(X_train_pca, y_train, classifier=ppn)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='upper left')
    plt.title(f'Regiones de decisión para {test_file}')
    plt.show()
