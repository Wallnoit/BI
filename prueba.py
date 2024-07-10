import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Función de activación
def activacion(pesos, x, b):
    z = np.dot(pesos, x)
    return 1 if z + b > 0 else 0

# Función para entrenar el perceptrón y calcular el error
def entrenar_perceptron(X_train, y_train, tasa_aprendizaje, epocas):
    n = X_train.shape[1]
    pesos = np.random.uniform(-1, 1, size=n)
    b = np.random.uniform(-1, 1)

    for epoca in range(epocas):
        error_total = 0
        for i in range(X_train.shape[0]):
            prediccion = activacion(pesos, X_train[i], b)
            error = y_train[i] - prediccion
            error_total += error**2
            for j in range(n):
                pesos[j] += tasa_aprendizaje * X_train[i][j] * error
            b += tasa_aprendizaje * error
        
        if epoca % 10 == 0 or epoca == epocas - 1:
            print(f"Época {epoca + 1}/{epocas}: Error total = {error_total}")
    
    return pesos, b

# Función para evaluar el perceptrón
def evaluar_perceptron(X, y, pesos, b):
    aciertos = 0
    for i in range(X.shape[0]):
        prediccion = activacion(pesos, X[i], b)
        if prediccion == y[i]:
            aciertos += 1
    return aciertos / X.shape[0]

def procesar_datasets(dataset_entrenamiento, dataset_prueba):
    # Cargar los datasets
    data_entrenamiento = pd.read_csv(dataset_entrenamiento)
    data_prueba = pd.read_csv(dataset_prueba)

    # Obtener el número de columnas menos 1 para la última columna de etiquetas
    num_columnas_entrenamiento = len(data_entrenamiento.columns)
    num_columnas_prueba = len(data_prueba.columns)
    
    columna_etiquetas_entrenamiento = data_entrenamiento.columns[num_columnas_entrenamiento - 1]  # Última columna
    columna_etiquetas_prueba = data_prueba.columns[num_columnas_prueba - 1]  # Última columna

    # Filtrar las etiquetas deseadas (por ejemplo, 1 y 2)
    etiquetas_deseadas = [1, 2]
    subset_entrenamiento = data_entrenamiento[data_entrenamiento[columna_etiquetas_entrenamiento].isin(etiquetas_deseadas)]
    subset_prueba = data_prueba[data_prueba[columna_etiquetas_prueba].isin(etiquetas_deseadas)]

    print("Subset Entrenamiento:")
    print(subset_entrenamiento.head())
    print("\nSubset Prueba:")
    print(subset_prueba.head())

    # Dividir el dataset de entrenamiento en conjunto de entrenamiento (80%) y conjunto de prueba (20%)
    X_train = subset_entrenamiento.iloc[:, :-1].values
    y_train = subset_entrenamiento.iloc[:, -1].values
    X_test = subset_prueba.iloc[:, :-1].values
    y_test = subset_prueba.iloc[:, -1].values

    # Verificar que los conjuntos de datos sean distintos
    if np.array_equal(X_train, X_test) and np.array_equal(y_train, y_test):
        print("Advertencia: Los datos de entrenamiento y prueba son idénticos.")
    else:
        print("Los datos de entrenamiento y prueba son distintos.")

    # Entrenar el perceptrón
    print(f"\nEntrenando con {dataset_entrenamiento}")
    pesos, b = entrenar_perceptron(X_train, y_train, tasa_aprendizaje=0.1, epocas=100)

    # Evaluar accuracies
    accuracy_entrenamiento = evaluar_perceptron(X_train, y_train, pesos, b)
    accuracy_prueba = evaluar_perceptron(X_test, y_test, pesos, b)

    print(f'Accuracy de entrenamiento = {accuracy_entrenamiento * 100:.2f}%')
    print(f'Accuracy de prueba = {accuracy_prueba * 100:.2f}%')

# Llamar a la función con los nombres de los archivos de entrada
procesar_datasets('dataset_entrenamiento_compras.csv', 'dataset_prueba_compras.csv')
