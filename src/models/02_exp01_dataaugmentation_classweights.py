import numpy as np
import os
import logging
import tensorflow as tf
import json
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, Sequential
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetV2B0 
import time
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)
import numpy as np
import sklearn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)
import pandas as pd
from sklearn.metrics import precision_score, recall_score

logging.getLogger("tensorflow").setLevel(logging.ERROR)
print(np.__version__)

print(tf.__version__, tf.sysconfig.get_build_info())

print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
print(tf.config.list_physical_devices("GPU"))

x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
print(tf.cast(x, tf.float16))

# %% Cell 2


#Configuramos directorio de resultados y metadatos, y modelos
RESULTS_DIR = "../src/models/resultados"
MODELOS_DIR = "../src/models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

# Cargar metadatos 
metadata_path = Path(MODELOS_DIR) / "metadata.json"

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Mostrar contenido
print(json.dumps(metadata, indent=4, ensure_ascii=False))

IMG_SIZE = metadata["image_size"]
BATCH_SIZE = metadata["batch_size"]
CLASS_NAMES = metadata["class_names"]
NUM_CLASSES = len(CLASS_NAMES)
SEED = metadata["seed"]
MODELOS_DIR = metadata["MODELOS_DIR"]
RESULTS_DIR = metadata["RESULTS_DIR"]
EPOCHS = metadata["EPOCHS"]
ruta_dataset = metadata["DATASET_DIR"]
# Nuevas rutas locales
ruta_train = ruta_dataset + "/train"
ruta_val = ruta_dataset + "/val"
ruta_test = ruta_dataset + "/test"

baseline = tf.keras.models.load_model(
    f"{MODELOS_DIR}/baseline_efficientnetv2b0.keras"
)

baseline.summary()

# %% Cell 4
for layer in baseline.layers:
    print(layer.name, layer.__class__.__name__, layer.trainable)


BATCH_SIZE = 16  # A 32 nos quedamos si memoria
X_SIZE = 224
Y_SIZE = 224
IMG_SIZE = (X_SIZE, Y_SIZE) # A 224 nos quedamos sin recursos
SEED = 42

# Cargar TRAIN (Se queda listo para el entrenamiento)
train_ds = tf.keras.utils.image_dataset_from_directory(
    ruta_train,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=True,
    seed=SEED
)

#Usamos class_names en validación y test para segurar que se usan los mismos índices
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

# Cargar VAL (Solo se llamará al final de cada época)
val_ds = tf.keras.utils.image_dataset_from_directory(
    ruta_val,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    shuffle=False
)

# Cargar TEST (Solo ocupará memoria al final de todo el entrenamiento)
test_ds = tf.keras.utils.image_dataset_from_directory(
    ruta_test,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    shuffle=False
)

# Optimización de flujo (SIN .cache(), solo prefetch)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("¡Datasets configurados para lectura bajo demanda desde el disco duro!")


# 1. Leer el archivo JSON existente
with open(metadata_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 2. Añadir los nuevos datos
data["class_names"] = class_names

# 3. Guardar el JSON actualizado (indent=4 lo hace legible)
with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.04),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )
], name="data_augmentation")

# Congelamos el backbone
baseline.get_layer("efficientnetv2-b0").trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

x = data_augmentation(inputs)
outputs = baseline(x)

model = Model(
    inputs=inputs,
    outputs=outputs,
    name="efficientnetv2b0_aug"
)

# Obtener las etiquetas del conjunto de entrenamiento
y_train = np.concatenate([
    labels.numpy()
    for images, labels in train_ds
])

# Como las etiquetas están en formato one-hot, convertimos a etiquetas enteras
if y_train.ndim > 1 and y_train.shape[-1] > 1:
    y_train = np.argmax(y_train, axis=1)
else:
    y_train = y_train.reshape(-1)

# Calcular pesos equilibrados
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(c): float(w)
    for c, w in zip(classes, weights)
}

# Mostrar resultados
print("Pesos por clase:")
for i, weight in class_weights.items():
    print(f"{class_names[i]}: {weight:.4f}")

base_model = EfficientNetV2B0 (
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

#Congelamos la red preentrenada
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Extracción de características
x = data_augmentation(inputs)
x = base_model(x, training=False)

# Clasificador
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(inputs, outputs, name="efficientnetv2b0_CW_y_DA")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

RESULTS_DIR = "./resultados"
os.makedirs(RESULTS_DIR, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        f"{MODELOS_DIR}/efficientnetv2b0_CW_y_DA.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    CSVLogger(
        f"{RESULTS_DIR}/efficientnetv2b0_CW_y_DA_history.csv"
    )
]

inicio = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS, 
    class_weight=class_weights,
    callbacks=callbacks
)

duracion = time.time() - inicio

print(f"Tiempo de entrenamiento: {duracion / 60:.2f} minutos")

# Nombre del modelo experimental
MODEL_PATH = Path(MODELOS_DIR) / "efficientnetv2b0_CW_y_DA.keras"

# Guardar el modelo completo
model.save(MODEL_PATH)

print(f"Modelo guardado en: {MODEL_PATH}")
print(f"Tamaño: {MODEL_PATH.stat().st_size / (1024**2):.2f} MB")

def plot_history(history):
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(hist["accuracy"], label="Train")
    axes[0].plot(hist["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Época")
    axes[0].legend()

    axes[1].plot(hist["loss"], label="Train")
    axes[1].plot(hist["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Época")
    axes[1].legend()

    plt.show()

plot_history(history)

# Cargar el mejor modelo guardado
best_model = tf.keras.models.load_model(
    f"{MODELOS_DIR}/efficientnetv2b0_CW_y_DA.keras"
)

# Predicciones sobre validación
y_true = np.concatenate([
    labels.numpy() for images, labels in val_ds
])

y_prob = best_model.predict(val_ds)
y_pred = np.argmax(y_prob, axis=1)

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
))

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
print("Balanced accuracy:", balanced_accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average="macro"))
print("Recall:", recall_score(y_true, y_pred, average="macro"))     

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(12, 12))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=False
)

plt.title("Baseline EfficientNetB0 DA y CW- Validación")
plt.tight_layout()
plt.show()

resultado_baseline = {
    "modelo": "EfficientNetv2B0",
    "archivo": "efficientnetv2b0_CW_y_DA.keras",
    "tipo": "Transfer learning - capas congeladas",
    "resolucion": "224x224",
    "batch_size": BATCH_SIZE,
    "learning_rate": 0.001,
    "epochs": len(history.history["loss"]),
    "tiempo_minutos": round(duracion / 60, 2),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_true, y_pred)
    )
}

resultados = pd.DataFrame([resultado_baseline])

resultados.to_csv(
    f"{RESULTS_DIR}/comparativa_modelos.csv",
    index=False
)
