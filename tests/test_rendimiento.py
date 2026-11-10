
import time
import os
import psutil
import numpy as np
import tensorflow as tf

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "vgg16.keras"

print("Cargando modelo VGG16...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado correctamente")

# Imagen sintética de entrada: 224x224 RGB
image = np.random.randint(
    0, 256, size=(1, 224, 224, 3)
).astype(np.float32)

# Preprocesamiento VGG16
image = tf.keras.applications.vgg16.preprocess_input(image)

# Calentamiento
for _ in range(5):
    model.predict(image, verbose=0)

# Medición
num_runs = 20
times = []

for _ in range(num_runs):
    start = time.perf_counter()
    model.predict(image, verbose=0)
    end = time.perf_counter()
    times.append((end - start) * 1000)

process = psutil.Process(os.getpid())
ram_mb = process.memory_info().rss / (1024 ** 2)

print("\n===== RESULTADOS DE RENDIMIENTO VGG16 =====")
print(f"Iteraciones: {num_runs}")
print(f"Tiempo medio: {np.mean(times):.2f} ms")
print(f"Tiempo mínimo: {np.min(times):.2f} ms")
print(f"Tiempo máximo: {np.max(times):.2f} ms")
print(f"RAM del proceso: {ram_mb:.2f} MB")
