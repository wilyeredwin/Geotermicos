import numpy as np
import pandas as pd

# Rutas de los archivos CSV
ruta_pesos = "D:/Geotermicos/Datos/Pesos.csv"
ruta_sesgos = "D:/Geotermicos/Datos/Sesgos.csv"

# Cargar los pesos y sesgos desde los archivos CSV y reemplazar valores NA con 0
pesos = pd.read_csv(ruta_pesos, header=None).fillna(0).to_numpy()
sesgos = pd.read_csv(ruta_sesgos, header=None).fillna(0).to_numpy()

# Ajustar pesos y sesgos para reflejar la estructura de la red (5 neuronas en la primera capa, 4 en la segunda)
pesos_capa_1 = pesos[:9, :5]  # Pesos de la primera capa (9 entradas, 5 neuronas)
pesos_capa_2 = pesos[9:14, :4]  # Pesos de la segunda capa (5 entradas, 4 neuronas)
pesos_final = pesos[14:, :1]  # Pesos de la capa de salida (4 entradas, 1 neurona)

sesgos_capa_1 = sesgos[:5, 0]  # Sesgos de la primera capa (5 neuronas)
sesgos_capa_2 = sesgos[5:9, 0]  # Sesgos de la segunda capa (4 neuronas)
sesgos_final = sesgos[9:, 0]  # Sesgos de la capa de salida (1 neurona)

# Organizar pesos y sesgos en listas
pesos_organizados = [pesos_capa_1, pesos_capa_2, pesos_final]
sesgos_organizados = [sesgos_capa_1, sesgos_capa_2, sesgos_final]

# Función para realizar la propagación hacia adelante
def forward_propagation(entrada, pesos, sesgos):
    """
    Realiza la propagación hacia adelante en una red neuronal con dos capas ocultas.
    :param entrada: Vector de entrada (numpy array).
    :param pesos: Lista de matrices de pesos por capa.
    :param sesgos: Lista de vectores de sesgos por capa.
    :return: Salida de la red neuronal.
    """
    activacion = entrada
    for i in range(len(pesos)):
        z = np.dot(activacion, pesos[i]) + sesgos[i]  # z = activación * pesos + sesgo
        if i < len(pesos) - 1:
            activacion = np.maximum(0, z)  # Función de activación ReLU
        else:
            activacion = z  # Salida lineal en la última capa
    return activacion

# Ejemplo de datos de entrada para realizar predicciones
nuevos_datos = np.array([[300, 50, 20, 3, 2, 0.1, 10, 20, 25]])  # Ejemplo de entrada

# Normalización de los datos
rango = nuevos_datos.max(axis=0) - nuevos_datos.min(axis=0)
rango[rango == 0] = 1  # Evitar división por cero asignando un rango de 1
nuevos_datos_normalizados = (nuevos_datos - nuevos_datos.min(axis=0)) / rango

# Realizar la predicción
prediccion = forward_propagation(nuevos_datos_normalizados, pesos_organizados, sesgos_organizados)
print(f"Predicción de cemento: {prediccion[0]}")