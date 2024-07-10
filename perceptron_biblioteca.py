import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Paso 1: Cargar los datos
train_data1 = pd.read_csv("Train_set_1.csv")
train_data2 = pd.read_csv("Train_set_2.csv")
train_data3 = pd.read_csv("Train_set_3.csv")

test_data1 = pd.read_csv("Test_set_1.csv")
test_data2 = pd.read_csv("Test_set_2.csv")
test_data3 = pd.read_csv("Test_set_3.csv")

# Paso 2: Preprocesamiento de datos
# Aquí debes realizar cualquier preprocesamiento necesario, como normalización de datos

# Separar características (X) de etiquetas (y)
X_train = pd.concat([train_data1, train_data2, train_data3]).drop(columns=["pred_est_emp"])
y_train = pd.concat([train_data1, train_data2, train_data3])["pred_est_emp"]

X_test = pd.concat([test_data1, test_data2, test_data3]).drop(columns=["pred_est_emp"])
y_test = pd.concat([test_data1, test_data2, test_data3])["pred_est_emp"]

# Paso 3: Entrenamiento del perceptrón
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# Paso 4: Evaluación del modelo
y_pred = perceptron.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
