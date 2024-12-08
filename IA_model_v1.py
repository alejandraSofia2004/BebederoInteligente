import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Puedes cambiar el modelo
from sklearn.metrics import mean_squared_error

# 1. Cargar los datos (reemplaza 'datos.csv' con tu archivo)
try:
    datos = pd.read_csv('datos.csv')
except FileNotFoundError:
    print("Error: Archivo 'datos.csv' no encontrado. Asegúrate de que el archivo exista en el mismo directorio que el script o proporciona la ruta completa.")
    exit()

# Verificar columnas necesarias
columnas_necesarias = ['temperatura', 'caudal', 'tiempo', 'consumo_agua'] # Asegúrate que coincidan con tus nombres de columna
if not all(col in datos.columns for col in columnas_necesarias):
    print(f"Error: El archivo CSV debe contener las columnas: {columnas_necesarias}")
    exit()


# 2. Preprocesamiento de datos (opcional, pero recomendado)
: Eliminar filas con valores faltantes
datos = datos.dropna()  

# 3. Dividir los datos en entrenamiento y prueba
X = datos[['temperatura', 'caudal', 'tiempo']] # Características
y = datos['consumo_agua']  # Variable objetivo (lo que queremos predecir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% entrenamiento, 20% prueba


# 4. Entrenar el modelo de IA (Regresión Lineal en este ejemplo)
modelo = LinearRegression()
modelo.fit(X_train, y_train)


# 5. Evaluar el modelo
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio: {mse}")

nueva_temperatura = 25  
nuevo_caudal = 10       
nuevo_tiempo = 30      

nuevos_datos = pd.DataFrame({'temperatura': [nueva_temperatura], 'caudal': [nuevo_caudal], 'tiempo': [nuevo_tiempo]})

prediccion_consumo = modelo.predict(nuevos_datos)
print(f"Predicción de consumo para temperatura={nueva_temperatura}, caudal={nuevo_caudal}, tiempo={nuevo_tiempo}: {prediccion_consumo[0]}")



import joblib
nombre_archivo_modelo = "modelo_bebedero_inteligente.pkl"
joblib.dump(modelo, nombre_archivo_modelo)
print(f"Modelo guardado como '{nombre_archivo_modelo}'")

