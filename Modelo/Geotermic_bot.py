import logging
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, filters
from tensorflow.keras.models import load_model
from telegram.ext import CallbackContext
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pytz import timezone

# Configuración del registro
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Estados del bot
TRAFFIC, RESISTANCE, SOIL, TEMPERATURE = range(4)

# Registrar la métrica 'mse'
mse = MeanSquaredError()

# Cargar el modelo entrenado
modelo = load_model("D:/Geotermicos/Modelo/modelo_entrenado.h5", custom_objects={'mse': mse})
print("Modelo cargado exitosamente.")

# Cargar el escalador
scaler = MinMaxScaler()
# Simula datos para ajustar el escalador (debe coincidir con los datos originales)
datos_ejemplo = pd.read_csv("D:/Geotermicos/Datos/Datos_R.csv", delimiter=';')
X = datos_ejemplo[['Mr', 'Sd', 'Cp', 'C_Mpa', 'T_Mpa', 'IP', 'LOI', 'MBI', 'Tem']]
scaler.fit(X)

# Calcular el valor del MSE
mse_value = mse.result().numpy()

# Mostrar los resultados en notación científica
print(f"Pérdida (MSE): {mse_value:.2e}")

# Función de inicio
async def start(update: Update, context) -> int:
    await update.message.reply_text(
        "¡Hola Ing. Civil! Soy un bot para ayudarte con el diseño de la mezcla para suelos mejorados "
        "con cemento para la construcción de pavimentos.\n"
        "Primero, responde las siguientes preguntas para que pueda realizar la predicción."
    )
    await update.message.reply_text(
        "1. ¿Qué esfuerzos por tráfico debe soportar la capa?\n\n"
        "Responde el Módulo Resiliente en MPa, el Esfuerzo Desviador en MPa y la Presión de confinamiento en MPa, separados por comas: Mr, Sd, Cp. \n"
        "Debes tener en cuenta que:\n"
        " -el Módulo Resiliente debe estar entre 10 y 400 MPa,\n"
        " -el Esfuerzo Desviador debe estar entre 10 y 67 MPa \n"
        " -y la Presión de confinamiento debe estar entre 10 y 42 MPa.")
    return TRAFFIC

# Pregunta 1: Tráfico
async def ask_traffic(update: Update, context) -> int:
    try:
        texto = update.message.text
        valores = list(map(float, texto.split(',')))

        if len(valores) != 3:
            await update.message.reply_text("Por favor, ingresa exactamente 3 valores separados por comas (Mr, Sd, Cp).")
            return TRAFFIC

        # Validar los rangos de los valores
        if not (10 <= valores[0] <= 400):
            await update.message.reply_text("El Módulo Resiliente (Mr) debe estar entre 10 y 400 MPa.")
            return TRAFFIC
        if not (10 <= valores[1] <= 67):
            await update.message.reply_text("El Esfuerzo Desviador (Sd) debe estar entre 10 y 67 MPa.")
            return TRAFFIC
        if not (10 <= valores[2] <= 42):
            await update.message.reply_text("La Presión de Confinamiento (Cp) debe estar entre 10 y 42 MPa.")
            return TRAFFIC

        # Guardar los valores en el contexto del usuario
        context.user_data['traffic'] = valores
        await update.message.reply_text(
            "2. ¿Cuál es la resistencia de diseño a compresión y tracción?\n\n"
            " -Recuerda que el valor de la Resistencia a la compresión debe estar entre 1 y 4 MPa  \n"
            " -y la resistencia a la tracción indirecta debe estar entre 0.5 y 6 MPa. \n"
            "Responde únicamente con los valores en MPa en el siguiente orden, separados por comas: C, T."
        )
        return RESISTANCE
    except ValueError:
        await update.message.reply_text("Por favor, ingresa valores numéricos separados por comas.")
        return TRAFFIC

# Pregunta 2: Resistencia
async def ask_resistance(update: Update, context) -> int:
    try:
        texto = update.message.text
        valores = list(map(float, texto.split(',')))

        if len(valores) != 2:
            await update.message.reply_text("Por favor, ingresa exactamente 2 valores separados por comas: Compresión (MPa), Tracción (MPa).")
            return RESISTANCE

        # Validar los rangos de los valores
        if not (1 <= valores[0] <= 4):
            await update.message.reply_text("El valor de la Resistencia a la Compresión (C) debe estar entre 1 y 4 MPa.")
            return RESISTANCE
        if not (0.5 <= valores[1] <= 6):
            await update.message.reply_text("El valor de la Resistencia a la Tracción (T) debe estar entre 0.5 y 6 MPa.")
            return RESISTANCE

        # Guardar los valores en el contexto del usuario
        context.user_data['resistance'] = valores
        await update.message.reply_text(
            "3. ¿Qué tipo de suelo deseas mejorar?\n\n"
            "Responde en el siguiente orden separados por comas: IP, LOI, MBI.\n"
            "Teniendo en cuenta que: \n"
            " -El Índice de Plasticidad (IP) debe estar entre 0 y 0.9, \n"
            " -el contenido de Materia orgánica (LOI) entre 0 y 20 \n"
            " -y el Índice de Azul de Metileno (MBI) entre 0 y 100.", parse_mode="Markdown"
        )
        return SOIL
    except ValueError:
        await update.message.reply_text("Por favor, ingresa valores numéricos separados por comas.")
        return RESISTANCE

# Pregunta 3: Tipo de suelo
async def ask_soil(update: Update, context) -> int:
    try:
        texto = update.message.text
        valores = list(map(float, texto.split(',')))

        if len(valores) != 3:
            await update.message.reply_text("Por favor, ingresa exactamente 3 valores separados por comas (IP, LOI, MBI).")
            return SOIL

        # Validar los rangos de los valores
        if not (0 <= valores[0] <= 0.9):
            await update.message.reply_text("El Índice de Plasticidad (IP) debe estar entre 0 y 0.9.")
            return SOIL
        if not (0 <= valores[1] <= 20):
            await update.message.reply_text("El contenido de Materia Orgánica (LOI) debe estar entre 0 y 20.")
            return SOIL
        if not (0 <= valores[2] <= 100):
            await update.message.reply_text("El Índice de Azul de Metileno (MBI) debe estar entre 0 y 100.")
            return SOIL

        # Guardar los valores en el contexto del usuario
        context.user_data['soil'] = valores
        await update.message.reply_text(
            "4. ¿Qué temperatura tendrás para curar el suelo cemento?\n\n" 
            "Responde en grados Centígrados."
        )
        return TEMPERATURE
    except ValueError:
        await update.message.reply_text("Por favor, ingresa valores numéricos separados por comas.")
        return SOIL

# Pregunta 4: Temperatura de curado
async def ask_tem(update: Update, context) -> int:
    try:
        texto = update.message.text
        valores = list(map(float, texto.split(',')))

        if len(valores) != 1:
            await update.message.reply_text("Por favor, ingresa un único valor en grados Celsius.")
            return TEMPERATURE

        # Paso 1: Validar y guardar la temperatura
        await update.message.reply_text("Paso 1/4: Validando la temperatura ingresada... ✅")
        context.user_data['temperature'] = valores

        # Paso 2: Preparar los datos de entrada
        await update.message.reply_text("Paso 2/4: Preparando los datos para el modelo... ⏳")
        entradas = (
            context.user_data['traffic'] +
            context.user_data['resistance'] +
            context.user_data['soil'] +
            context.user_data['temperature']
        )
        valores_normalizados = scaler.transform([entradas])

        # Paso 3: Realizar la predicción
        await update.message.reply_text("Paso 3/4: Calculando el valor predicho... 🔄")
        prediccion = modelo.predict(valores_normalizados)
        resultado = prediccion[0][0]

        # Paso 4: Calcular métricas y mostrar resultados
        await update.message.reply_text("Paso 4/4: Calculando métricas de evaluación... 📊")
        predicciones_csv_path = "D:/Geotermicos/Datos/Predicciones.csv"
        predicciones_df = pd.read_csv(predicciones_csv_path)

        # Extraer y_true y y_pred del archivo CSV
        y_true = predicciones_df['y_true'].values
        y_pred = predicciones_df['y_pred'].values

        # Calcular el MSE usando los valores del archivo CSV
        mse_metric = MeanSquaredError()
        mse_metric.update_state(y_true, y_pred)
        mse_value = mse_metric.result().numpy()

        # Calcular el MAE usando los valores del archivo CSV
        mae_metric = MeanAbsoluteError()
        mae_metric.update_state(y_true, y_pred)
        mae_value = mae_metric.result().numpy()

        # Enviar el mensaje con los resultados finales
        await update.message.reply_text(
            f"✅ Proceso completado.\n\n"
            f"El contenido de cemento en peso es:\n```\n{resultado:.2f}\n```"
            f"es decir que por cada Tonelada de suelo seco, se requiere {resultado:.2f} Toneladas de cemento.\n\n"
            f"Pérdida (MSE) calculada:\n```\n{mse_value:.2e}\n```\n"
            f"Error Absoluto Medio (MAE) calculado:\n```\n{mae_value:.2e}\n```\n\n"
            "Recuerda que este valor es una estimación y puede variar según las condiciones específicas de tu proyecto.\n"
            "Si deseas realizar otra predicción, simplemente envía /start para reiniciar el proceso.\n"
            "Éxitos con tu pavimento. ¡Hasta luego!",
            parse_mode="Markdown"
        )
        return ConversationHandler.END
    except FileNotFoundError:
        await update.message.reply_text("No se encontró el archivo de predicciones. Asegúrate de que exista en la ruta especificada.")
        return TEMPERATURE
    except ValueError:
        await update.message.reply_text("Por favor, ingresa valores numéricos separados por comas.")
        return TEMPERATURE

# Función para cancelar la conversación
async def cancel(update: Update, context) -> int:
    await update.message.reply_text("Operación cancelada. ¡Hasta luego!")
    return ConversationHandler.END

# Configuración del bot
def main():
    # Reemplaza 'YOUR_TOKEN' con el token de tu bot
    application = Application.builder().token("7959624365:AAHWHu7tQ2t1ZW_32PN1d9j4VnImY5jy9mg").build()

    # Crear el manejador de la conversación
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            TRAFFIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_traffic)],
            RESISTANCE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_resistance)],
            SOIL: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_soil)],
            TEMPERATURE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_tem)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    # Registrar el manejador
    application.add_handler(conv_handler)

    # Iniciar el bot
    application.run_polling()

if __name__ == '__main__':
    main()