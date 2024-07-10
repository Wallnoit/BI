import pandas as pd

from sklearn.model_selection import KFold

import math

nombre_directorio = ''
nombre_archivo = 'prediccionFactComprasE.csv'

nivel_confianza = 0.95

# Especifica el valor de N deseado
N_deseado = 439344

# Calcula el error correspondiente para el valor de N dado
error_calculado = math.sqrt((math.log(2) - math.log(1 - nivel_confianza)) / (2 * N_deseado))

# Imprime el error calculado
print(f"Para N={N_deseado}, el error calculado es {error_calculado * 100:.2f}%")

# Cargar el archivo CSV
df = pd.read_csv(''+nombre_archivo)

# Definir la columna que contiene las clases
columna_clases = 'pred_est_emp'

# Inicializar el objeto KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Iterar sobre las divisiones k-fold
for i, (train_index, test_index) in enumerate(kf.split(df)):
    train_set = df.iloc[train_index]
    test_set = df.iloc[test_index]
    
    # Mostrar la longitud de cada parte
    print(f"Iteración {i+1}:")
    print("Train set:", len(train_set))
    print("Test set:", len(test_set))
    
    # Guardar los conjuntos en archivos CSV
    train_set.to_csv(nombre_directorio+""+f'CTrain_set_{i+1}.csv', index=False)
    test_set.to_csv(nombre_directorio+""+f'CTest_set_{i+1}.csv', index=False)
    print()







