import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
"""Este script implementa el entrenamiento de una 
   red neuronal artificial (RNA) 
   usando Python y Keras/TensorFlow 
   para predecir el contenido óptimo de cemento necesario
   para estabilizar suelos, a partir de variables de entrada
   relevantes en ingeniería civil."""
# Solicitar la ruta y el nombre del archivo al usuario
ruta_archivo = "D:/Geotermicos/Datos/Datos_R.csv" # Cambia esto a la ruta deseada)

try:
    # Leer el archivo .csv
    datos = pd.read_csv(ruta_archivo, delimiter=';')  # Asegúrate de usar el delimitador correcto (; en este caso)
    
    # Mostrar los datos cargados
    print("Datos cargados del archivo CSV:")
    print(datos.head())

    # Separar características (X) y variable objetivo (y)
    """Se seleccionan las columnas relevantes como variables 
       de entrada (X): propiedades del suelo, esfuerzos, temperatura, etc."""
    X = datos[['Mr', 'Sd', 'Cp', 'C_Mpa', 'T_Mpa', 'IP', 'LOI', 'MBI', 'Tem']]  # Asegúrate de que estas columnas existan en tu archivo
    y = datos['C_S']  # Cambia esto al nombre de la columna de tu variable objetivo

    # Normalizar las características
    """Se normalizan las variables de entrada con MinMaxScaler
       para que todas estén en el mismo rango y el modelo 
       aprenda mejor."""
    scaler = MinMaxScaler()
    X_normalizado = scaler.fit_transform(X)

    # Dividir los datos en entrenamiento y prueba
    """Los datos se dividen en conjuntos de entrenamiento
       y prueba (80%/20%) usando train_test_split."""
    X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y, test_size=0.2, random_state=42)

    # Crear el modelo de red neuronal
    modelo = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Capa oculta 1
        Dense(32, activation='relu'),                                  # Capa oculta 2
        Dense(1, activation='linear')                                  # Capa de salida
    ])

    # Compilar el modelo
    modelo.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Entrenar el modelo
    historia = modelo.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

    # Evaluar el modelo
    resultados = modelo.evaluate(X_test, y_test)
    print(f"Pérdida (MSE): {resultados[0]}, Error Absoluto Medio (MAE): {resultados[1]}")

    # Realizar predicciones en el conjunto de prueba
    y_pred = modelo.predict(X_test)

    # Crear un DataFrame con los valores reales (y_true) y los valores predichos (y_pred)
    resultados_df = pd.DataFrame({
        'y_true': y_test.values,  # Convertir y_test a un array si es necesario
        'y_pred': y_pred.flatten()  # Asegurarse de que y_pred sea un array unidimensional
    })

    # Guardar los resultados en un archivo CSV
    resultados_csv_path = "D:/Geotermicos/Datos/Predicciones.csv"
    resultados_df.to_csv(resultados_csv_path, index=False)
    print(f"Resultados guardados exitosamente en '{resultados_csv_path}'")

    # Graficar la pérdida durante el entrenamiento
    plt.plot(historia.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(historia.history['val_loss'], label='Pérdida de validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()
    #'Tem', 'Mr', 'Sd', 'Cp', 'n', 'IP', 'LOI', 'MBI',
    #    'C_Mpa', 'T_Mpa'
    # Predicción en nuevos datos (ejemplo)
    nuevos_datos = [[300, 50, 20, 3, 2, 0.1, 10, 20, 25]]  # Ejemplo: Resiliencia, Tracción, Compresión
    nuevos_datos_normalizados = scaler.transform(nuevos_datos)
    prediccion = modelo.predict(nuevos_datos_normalizados)
    resultado = prediccion[0][0]
    print(f"Predicción de cemento: {resultado}")
    # Mostrar la predicción
    print(f"Contenido de cemento predicho: {prediccion[0][0]}")


except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta especificada: {ruta_archivo}")
except Exception as e:
    print(f"Se produjo un error al leer el archivo o procesar los datos: {e}")

# Guardar el modelo entrenado
modelo.save("D:/Geotermicos/Modelo/modelo_entrenado.h5")
print("Modelo guardado exitosamente en 'D:/Geotermicos/Modelo/modelo_entrenado.h5'")