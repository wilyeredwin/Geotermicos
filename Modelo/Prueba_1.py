import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Solicitar la ruta y el nombre del archivo al usuario
ruta_archivo = "D:/Geotermicos/Datos/Datos.csv" # Cambia esto a la ruta deseada)

try:
    # Leer el archivo .csv
    datos = pd.read_csv(ruta_archivo, delimiter=';')  # Asegúrate de usar el delimitador correcto (; en este caso)
    
    # Mostrar los datos cargados
    print("Datos cargados del archivo CSV:")
    print(datos.head())

    # Separar características (X) y variable objetivo (y)
    X = datos[['Temperatura', 'Resiliencia',
               'Esfuerzo desviador', 'Confinamiento', 'Traccion', 
               'Compresion', 'Densidad seca', 'Porosidad', 'Plasticidad', 
               'Materia organica', 'Indice azul de metileno', 'Conductividad termica', 
               'Contenido de agua']]
    y = datos['Contenido de cemento']

    # Normalizar las características
    scaler = MinMaxScaler()
    X_normalizado = scaler.fit_transform(X)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y, test_size=0.2, random_state=42)

    # Crear el modelo de red neuronal
    modelo = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Capa oculta 1
        Dense(32, activation='relu'),                                  # Capa oculta 2
        Dense(1, activation='linear')                                  # Capa de salida
    ])

    # Compilar el modelo
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Entrenar el modelo
    historia = modelo.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

    # Evaluar el modelo
    resultados = modelo.evaluate(X_test, y_test)
    print(f"Pérdida (MSE): {resultados[0]}, Error Absoluto Medio (MAE): {resultados[1]}")

    # Graficar la pérdida durante el entrenamiento
    plt.plot(historia.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(historia.history['val_loss'], label='Pérdida de validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()

    # Predicción en nuevos datos (ejemplo)
    nuevos_datos = [[25, 150, 3.8, 100, 2.7, 1.8, 2.5, 0.4, 0.3, 0.2, 0.1, 1.2, 0.5]]  # Ejemplo: Resiliencia, Tracción, Compresión
    nuevos_datos_normalizados = scaler.transform(nuevos_datos)
    prediccion = modelo.predict(nuevos_datos_normalizados)
    print(f"Contenido de cemento predicho: {prediccion[0][0]}")

except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta especificada: {ruta_archivo}")
except Exception as e:
    print(f"Se produjo un error al leer el archivo o procesar los datos: {e}")