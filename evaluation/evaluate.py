
# ==========================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==========================================================

import os
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# 2. CONFIGURACIÓN DE RUTAS Y CATEGORÍAS
# ==========================================================

# Ruta del conjunto de imágenes de prueba
RUTA_TEST = r"C:\Users\Antonio\Desktop\ANTOTE\MASTER BIG DATA IA\SEGUNDO SEMESTRE\PROYECTO\NF3\test"

# Ruta del modelo VGG16
RUTA_MODELO = r"C:\Users\Antonio\Downloads\Proyecto-Antonio\src\models\vgg16.keras"

# Orden de las seis categorías del modelo
CLASES = [
    "1. Enfeksiyonel",
    "2. Ekzama",
    "3. Akne",
    "4. Pigment",
    "5. Benign",
    "6. Malign"
]


# ==========================================================
# 3. CARGA DEL MODELO VGG16
# ==========================================================

print("\nCargando modelo VGG16...")

modelo = tf.keras.models.load_model(
    RUTA_MODELO,
    compile=False
)

print("Modelo cargado correctamente.")
print("Entrada:", modelo.input_shape)
print("Salida:", modelo.output_shape)



# ==========================================================
# 4. CARGA Y PREPROCESAMIENTO DE IMÁGENES
# ==========================================================

print("\nPreparando imágenes de test...")

# Listas para guardar las imágenes y sus categorías reales
imagenes = []
etiquetas_reales = []

# Recorrer las seis categorías
for indice, clase in enumerate(CLASES):

    ruta_clase = os.path.join(RUTA_TEST, clase)

    # Obtener los archivos de imagen
    archivos = sorted([
        f for f in os.listdir(ruta_clase)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for archivo in archivos:

        ruta_imagen = os.path.join(ruta_clase, archivo)

        try:
            # Abrir la imagen y convertirla a RGB
            imagen = Image.open(ruta_imagen).convert("RGB")

            # Redimensionar a 224 x 224 píxeles
            imagen = imagen.resize((224, 224))

            # Convertir a matriz NumPy
            imagen = np.array(imagen, dtype=np.float32)

            # Guardar imagen y categoría real
            imagenes.append(imagen)
            etiquetas_reales.append(indice)

        except Exception as e:
            print(f"Error al procesar {ruta_imagen}: {e}")

# Convertir las listas a matrices NumPy
imagenes = np.array(imagenes, dtype=np.float32)
etiquetas_reales = np.array(etiquetas_reales)

# Aplicar el preprocesamiento específico de VGG16
imagenes = preprocess_input(imagenes)

print("Imágenes preparadas correctamente.")
print("Total de imágenes:", len(imagenes))
print("Dimensiones:", imagenes.shape)



# ==========================================================
# 5. PREDICCIÓN DE LAS IMÁGENES DE TEST
# ==========================================================

print("\nRealizando predicciones con VGG16...")

# El modelo analiza las imágenes en lotes de 32
predicciones = modelo.predict(
    imagenes,
    batch_size=32,
    verbose=1
)

# Obtener la categoría con mayor puntuación para cada imagen
etiquetas_predichas = np.argmax(predicciones, axis=1)

print("\nPredicciones completadas.")
print("Número de predicciones:", len(etiquetas_predichas))
print("Dimensiones de las predicciones:", predicciones.shape)


# ==========================================================
# 6. CÁLCULO Y GUARDADO DE MÉTRICAS
# ==========================================================

print("\nCalculando métricas de evaluación...")

# Calcular la exactitud global
accuracy = accuracy_score(
    etiquetas_reales,
    etiquetas_predichas
)

# Generar el informe detallado
informe = classification_report(
    etiquetas_reales,
    etiquetas_predichas,
    labels=list(range(len(CLASES))),
    target_names=CLASES,
    zero_division=0
)

# Obtener las métricas en formato estructurado
informe_dict = classification_report(
    etiquetas_reales,
    etiquetas_predichas,
    labels=list(range(len(CLASES))),
    target_names=CLASES,
    output_dict=True,
    zero_division=0
)

# Crear la carpeta donde guardaremos los resultados
RUTA_RESULTADOS = os.path.join(
    os.path.dirname(__file__),
    "results"
)

os.makedirs(RUTA_RESULTADOS, exist_ok=True)

# Guardar las métricas principales en JSON
metricas = {
    "accuracy": accuracy,
    "precision_macro": informe_dict["macro avg"]["precision"],
    "recall_macro": informe_dict["macro avg"]["recall"],
    "f1_macro": informe_dict["macro avg"]["f1-score"],
    "precision_weighted": informe_dict["weighted avg"]["precision"],
    "recall_weighted": informe_dict["weighted avg"]["recall"],
    "f1_weighted": informe_dict["weighted avg"]["f1-score"],
    "total_imagenes": len(etiquetas_reales)
}

ruta_json = os.path.join(RUTA_RESULTADOS, "metrics.json")

with open(ruta_json, "w", encoding="utf-8") as archivo:
    json.dump(metricas, archivo, indent=4, ensure_ascii=False)

# Guardar el informe completo por categorías
ruta_informe = os.path.join(
    RUTA_RESULTADOS,
    "classification_report.txt"
)

with open(ruta_informe, "w", encoding="utf-8") as archivo:
    archivo.write("INFORME DE EVALUACIÓN DEL MODELO VGG16\n\n")
    archivo.write(informe)

# Mostrar resultados en el terminal
print("\n========== RESULTADOS DE EVALUACIÓN ==========")
print(f"Accuracy: {accuracy:.4f}")
print("\nInforme por categorías:")
print(informe)

print("\nArchivos guardados en:", RUTA_RESULTADOS)
print("- metrics.json")
print("- classification_report.txt")

print("\nINICIO DEL APARTADO 7")
# ==========================================================
# 7. GENERACIÓN DE LA MATRIZ DE CONFUSIÓN
# ==========================================================

print("\nGenerando matriz de confusión...")

# Calcular la matriz
matriz = confusion_matrix(
    etiquetas_reales,
    etiquetas_predichas,
    labels=list(range(len(CLASES)))
)

# Crear la representación gráfica
fig, ax = plt.subplots(figsize=(12, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=CLASES
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)

plt.title("Matriz de confusión - Modelo VGG16")
plt.tight_layout()

# Guardar la imagen
ruta_matriz = os.path.join(
    RUTA_RESULTADOS,
    "confusion_matrix.png"
)

plt.savefig(ruta_matriz, dpi=300, bbox_inches="tight")
plt.close()

print("Matriz de confusión guardada correctamente.")


# ==========================================================
# 8. EXPORTACIÓN DE PREDICCIONES A CSV
# ==========================================================

print("\nExportando predicciones a CSV...")

import pandas as pd

# Crear una tabla con los resultados
datos_predicciones = {
    "categoria_real": [
        CLASES[indice] for indice in etiquetas_reales
    ],
    "categoria_predicha": [
        CLASES[indice] for indice in etiquetas_predichas
    ],
    "acierto": (
        etiquetas_reales == etiquetas_predichas
    )
}

# Añadir las puntuaciones de las seis categorías
for i, clase in enumerate(CLASES):
    datos_predicciones[f"probabilidad_{clase}"] = predicciones[:, i]

# Crear el DataFrame
df_predicciones = pd.DataFrame(datos_predicciones)

# Guardar el archivo CSV
ruta_csv = os.path.join(
    RUTA_RESULTADOS,
    "predictions.csv"
)

df_predicciones.to_csv(
    ruta_csv,
    index=False,
    encoding="utf-8-sig"
)

print("CSV guardado correctamente.")
print("Total de registros:", len(df_predicciones))
