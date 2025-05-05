import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from sklearn.preprocessing import MinMaxScaler

# Cargar los pesos desde el archivo CSV exportado desde R
pesos = pd.read_csv("D:/Geotermicos/Datos/pesos_modelo.csv", delimiter=';')

# Verificar los tipos de datos
print("Tipos de datos en el archivo CSV:")
print(pesos.dtypes)

# Mostrar las primeras filas del archivo CSV
print(pesos.head())

# Si las columnas contienen cadenas separadas por ';', conviértelas a flotantes
if pesos['capa1_pesos'].dtype == 'object':
    pesos['capa1_pesos'] = pesos['capa1_pesos'].str.split(';').explode().astype(float)
    pesos['capa1_bias'] = pesos['capa1_bias'].str.split(';').explode().astype(float)
    pesos['capa2_pesos'] = pesos['capa2_pesos'].str.split(';').explode().astype(float)
    pesos['capa2_bias'] = pesos['capa2_bias'].str.split(';').explode().astype(float)
    pesos['salida_pesos'] = pesos['salida_pesos'].str.split(';').explode().astype(float)
    pesos['salida_bias'] = pesos['salida_bias'].str.split(';').explode().astype(float)

# Crear el modelo con la misma arquitectura que en R
modelo = Sequential([
    Input(shape=(11,)),  # Define la entrada explícitamente
    Dense(5, activation='relu'),                     # Capa oculta 1 (11 entradas x 5 neuronas)
    Dense(4, activation='relu'),                     # Capa oculta 2 (5 entradas x 4 neuronas)
    Dense(1, activation='linear')                    # Capa de salida (4 entradas x 1 neurona)
])

# Asignar los pesos y sesgos al modelo
modelo.layers[0].set_weights([
    np.array(pesos['capa1_pesos']).reshape(11, 5),  # Pesos de la primera capa
    np.array(pesos['capa1_bias'])                  # Sesgos de la primera capa
])
modelo.layers[1].set_weights([
    np.array(pesos['capa2_pesos']).reshape(5, 4),  # Pesos de la segunda capa
    np.array(pesos['capa2_bias'])                 # Sesgos de la segunda capa
])
modelo.layers[2].set_weights([
    np.array(pesos['salida_pesos']).reshape(4, 1),  # Pesos de la capa de salida
    np.array(pesos['salida_bias'])                 # Sesgos de la capa de salida
])

print("Pesos asignados correctamente al modelo.")

# Cargar los datos de entrada para realizar predicciones
datos = pd.read_csv("D:/Geotermicos/Datos/dataset.csv", delimiter=';')

# Seleccionar las 11 variables de entrada
X = datos[['Tem', 'Mr', 'Sd', 'Cp', 'n', 'IP', 'LOI', 'MBI', 'C_Mpa', 'T_Mpa', 'Duw']]

# Normalizar los datos (si los datos en R fueron normalizados)
scaler = MinMaxScaler()
X_normalizado = scaler.fit_transform(X)

# Realizar predicciones
predicciones = modelo.predict(X_normalizado)

# Mostrar las predicciones
print("Predicciones:")
print(predicciones)