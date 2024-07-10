import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_files = ['Train_set_1.csv', 'Train_set_2.csv', 'Train_set_3.csv']
test_files = ['Test_set_1.csv', 'Test_set_2.csv', 'Test_set_3.csv']

etiqueta = 'pred_est_emp'

train_data = []
test_data = []

# Cargar los datos de los archivos CSV
for file in train_files:
    train_data.append(pd.read_csv(file))

for file in test_files:
    test_data.append(pd.read_csv(file))

# Lista para almacenar los pesos del modelo
weights = []

# Procesar cada conjunto de datos
for i in range(3):
    # Dividir en características (X) y variable objetivo (y)
    X_train = train_data[i].drop(etiqueta, axis=1)
    y_train = train_data[i][etiqueta]

    X_test = test_data[i].drop(etiqueta, axis=1)
    y_test = test_data[i][etiqueta]

   # Convertir etiquetas de {1, -1} a {1, 0}
    y_train = y_train.replace({1: 1, -1: 0})
    y_test = y_test.replace({1: 1, -1: 0})

    # Para simplificar, vamos a seleccionar solo una característica para la visualización de la curva sigmoide
    # Suponiendo que la primera columna es 'feature1'
    X_train_single_feature = X_train.iloc[:, 0].values.reshape(-1, 1)
    X_test_single_feature = X_test.iloc[:, 0].values.reshape(-1, 1)

    # Estandarizar la característica
    scaler = StandardScaler()
    X_train_single_feature = scaler.fit_transform(X_train_single_feature)
    X_test_single_feature = scaler.transform(X_test_single_feature)

    # Crear el modelo de regresión logística
    model = LogisticRegression()

    # Entrenar el modelo
    model.fit(X_train_single_feature, y_train)

    # Guardar los pesos del modelo
    weights.append(model.coef_)

    # Hacer predicciones
    y_pred = model.predict(X_test_single_feature)

    # Evaluar el rendimiento del modelo
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Modelo {i+1}:")
    print("Pesos del modelo:", model.coef_)
    print("Precisión del modelo:", accuracy)
    print("Matriz de confusión:")
    print(conf_matrix)
    print("Reporte de clasificación:")
    print(class_report)

# Graficar la curva sigmoide del primer modelo como ejemplo
X_range = np.linspace(X_train_single_feature.min(), X_train_single_feature.max(), 300).reshape(-1, 1)
y_prob = model.predict_proba(X_range)[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(X_train_single_feature, y_train, color='blue', label='Datos de Entrenamiento')
plt.scatter(X_test_single_feature, y_test, color='green', label='Datos de Prueba')
plt.plot(X_range, y_prob, color='red', linewidth=2, label='Curva Sigmoide')
plt.xlabel('Característica Estandarizada')
plt.ylabel('Probabilidad')
plt.title('Curva Sigmoide de la Regresión Logística')
plt.legend()
plt.show()
