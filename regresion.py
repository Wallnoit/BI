import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar datos (asegúrate de tener un archivo CSV con datos adecuados)
data = pd.read_csv('estandarizado.csv')

# Dividir en características (X) y variable objetivo (y)
X = data.drop('Etiqueta', axis=1)
y = data['Etiqueta']

# Convertir etiquetas de {1, 2} a {0, 1}
y = y.replace({1: 0, 2: 1})

# Para simplificar, vamos a seleccionar solo una característica para la visualización de la curva sigmoide
# Suponiendo que la primera columna es 'feature1'
X_single_feature = X.iloc[:, 0].values.reshape(-1, 1)

# Estandarizar la característica
scaler = StandardScaler()
X_single_feature = scaler.fit_transform(X_single_feature)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_single_feature, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión logística
model = LogisticRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Precisión del modelo:", accuracy)
print("Matriz de confusión:")
print(conf_matrix)
print("Reporte de clasificación:")
print(class_report)

# Graficar la curva sigmoide
# Generar un rango de valores para la característica
X_range = np.linspace(X_single_feature.min(), X_single_feature.max(), 300).reshape(-1, 1)

# Predecir las probabilidades utilizando el modelo entrenado
y_prob = model.predict_proba(X_range)[:, 1]

# Dibujar la curva sigmoide
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Datos de Entrenamiento')
plt.scatter(X_test, y_test, color='green', label='Datos de Prueba')
plt.plot(X_range, y_prob, color='red', linewidth=2, label='Curva Sigmoide')
plt.xlabel('Característica Estandarizada')
plt.ylabel('Probabilidad')
plt.title('Curva Sigmoide de la Regresión Logística')
plt.legend()
plt.show()