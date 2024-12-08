import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

# 1. Cargar el modelo entrenado
try:
    modelo = joblib.load("modelo_bebedero_inteligente.pkl")
except FileNotFoundError:
    print("Error: No se encontró el archivo del modelo. Asegúrate de que 'modelo_bebedero_inteligente.pkl' esté en el mismo directorio.")
    exit()

# 2. Inicializar Firebase
cred = credentials.Certificate("ruta/a/tu/archivo/credenciales/firebase.json")  
firebase_admin.initialize_app(cred)
db = firestore.client()


# 3. Obtener datos de Firebase (asumiendo una colección llamada 'mediciones')
mediciones_ref = db.collection('mediciones')
docs = mediciones_ref.get()  # Obtiene todos los documentos de la colección

datos = []
for doc in docs:
    datos.append(doc.to_dict())

df = pd.DataFrame(datos)  # Crea un DataFrame de Pandas con los datos de Firebase



if df.empty:
    print("Error: No se encontraron datos en la colección 'mediciones' de Firebase.")
    exit()


columnas_necesarias = ['temperatura', 'caudal', 'tiempo']
if not all(col in df.columns for col in columnas_necesarias):
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
    print(f"Error: Faltan las siguientes columnas en los datos de Firebase: {columnas_faltantes}")
    exit()

X = df[['temperatura', 'caudal', 'tiempo']]
predicciones = modelo.predict(X)


batch = db.batch()
for i, doc in enumerate(docs):
    doc_ref = mediciones_ref.document(doc.id)
    batch.update(doc_ref, {'consumo_predicho': float(predicciones[i])})  

batch.commit()

print("Predicciones realizadas y guardadas en Firebase.")

print(predicciones)