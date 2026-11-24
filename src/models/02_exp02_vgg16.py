import json
from pathlib import Path 
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight


# Cargar metadatos 
metadata_path = "metadata.json"
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(json.dumps(metadata, indent=4, ensure_ascii=False))

IMG_SIZE = metadata["image_size"]
BATCH_SIZE = metadata["batch_size"]
CLASS_NAMES = metadata["class_names"]
NUM_CLASSES = len(CLASS_NAMES)
SEED = metadata["seed"]
MODELOS_DIR = metadata["MODELOS_DIR"]
RESULTS_DIR = metadata["RESULTS_DIR"]
EPOCHS  = metadata["EPOCHS"]
ruta_dataset = metadata["DATASET_DIR"]
ruta_train = ruta_dataset + "/train"
ruta_val = ruta_dataset + "/val"
ruta_test = ruta_dataset + "/test"

keras.utils.set_random_seed(SEED)

# Configurar crecimiento dinámico de memoria GPU
# Evita que tf reserve toda la memoria de la GPU al inicio, permitiendo que otros procesos puedan usarla.
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("Memory growth activado para", len(gpus), "GPU(s)")

    except RuntimeError as e:
        print("Error al configurar memory growth:", e)

else:
    print("No se detectan GPU")


print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))


NUM_CLASSES = len(CLASS_NAMES)
CLASS_IDS = np.arange(NUM_CLASSES)
MALIGN_IDX = CLASS_NAMES.index("6. Malign")

# Usa la misma resolución que en el baseline.
IMG_SIZE = (IMG_SIZE[0], IMG_SIZE[1])  # (224, 224)
BATCH_SIZE = min(BATCH_SIZE, 8)  # VGG16 consume bastante más memoria que MobileNetV2.

USE_AUGMENTATION = False    
USE_CLASS_WEIGHTS = False

HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = EPOCHS - HEAD_EPOCHS
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5

BEST_MODEL_PATH = Path(MODELOS_DIR) / "vgg16.keras"

assert Path(ruta_train).is_dir(), f"No existe {ruta_train}"
assert Path(ruta_val).is_dir(), f"No existe {ruta_val}"

def load_partition(directory, shuffle):
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

train_raw = load_partition(ruta_train, shuffle=True)
val_raw = load_partition(ruta_val, shuffle=False)

train_paths = list(train_raw.file_paths)
val_paths = list(val_raw.file_paths)

# Las etiquetas de entrenamiento se usan solamente para contar clases.
y_train = np.concatenate([
    labels.numpy() for _, labels in train_raw
])
y_val = np.concatenate([
    labels.numpy() for _, labels in val_raw
])

distribution = pd.DataFrame({
    "clase": CLASS_NAMES,
    "train": np.bincount(y_train, minlength=NUM_CLASSES),
    "val": np.bincount(y_val, minlength=NUM_CLASSES),
})
print(distribution.to_string(index=False))

# Comprueba el soporte del informe original.
expected_val_support = np.array([159, 74, 175, 111, 1014, 823])
actual_val_support = np.bincount(y_val, minlength=NUM_CLASSES)

assert np.array_equal(actual_val_support, expected_val_support), (
    "La distribución de validación no coincide con el baseline. "
    "Revisa la partición y el orden de clases."
)

# Esta comprobación no garantiza que sean las mismas imágenes:
# para eso hay que comprobar la partición original.
pd.DataFrame({
    "path": val_paths,
    "label": y_val,
}).to_csv(Path(RESULTS_DIR) / "validation_manifest.csv", index=False)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_raw.prefetch(AUTOTUNE)
val_ds = val_raw.prefetch(AUTOTUNE)

assert np.all(np.bincount(y_train, minlength=NUM_CLASSES) > 0), (
    "Faltan clases en entrenamiento."
)

if USE_CLASS_WEIGHTS:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=CLASS_IDS,
        y=y_train,
    )
    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(CLASS_IDS, weights)
    }
else:
    class_weights = None

print("Class weights:", class_weights)


keras.backend.clear_session()
keras.utils.set_random_seed(SEED)

augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05, fill_mode="reflect"),
    layers.RandomZoom(0.10, fill_mode="reflect"),
], name="augmentation")

base_model = keras.applications.VGG16(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")
x = augmentation(inputs) if USE_AUGMENTATION else inputs
x = layers.Lambda(
    keras.applications.vgg16.preprocess_input,
    name="vgg16_preprocessing",
)(x)

# VGG16 no incorpora capas BatchNormalization.
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="class_probabilities",
)(x)

model = keras.Model(inputs, outputs, name="VGG16_skin")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=HEAD_LR),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.summary()

class ValidationMacroF1(keras.callbacks.Callback):
    def __init__(self, dataset, y_true, save_path):
        super().__init__()
        self.dataset = dataset
        self.y_true = np.asarray(y_true)
        self.save_path = str(save_path)
        self.best = -np.inf
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}

        probabilities = self.model.predict(self.dataset, verbose=0)
        predictions = probabilities.argmax(axis=1)

        macro_f1 = f1_score(
            self.y_true,
            predictions,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        )

        logs["val_macro_f1"] = float(macro_f1)
        self.records.append({
            "validation_check": len(self.records) + 1,
            "val_macro_f1": float(macro_f1),
        })

        print(f" — val_macro_f1: {macro_f1:.4f}")

        if macro_f1 > self.best:
            self.best = float(macro_f1)
            self.model.save(self.save_path)
            print(f"   Mejor checkpoint guardado: {macro_f1:.4f}")


macro_callback = ValidationMacroF1(
    dataset=val_ds,
    y_true=y_val,
    save_path=BEST_MODEL_PATH,
)

print(f"El mejor checkpoint se guardará en: {BEST_MODEL_PATH}")

def make_callbacks():
    # macro_callback debe ir antes de los callbacks que leen val_macro_f1.
    return [
        macro_callback,
        keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(),
)

model = keras.models.load_model(BEST_MODEL_PATH)

base_model = next(
    layer for layer in model.layers
    if isinstance(layer, keras.Model)
    and "vgg16" in layer.name.lower()
)

base_model.trainable = True

# Descongelar únicamente el último bloque convolucional (block5).
# Los bloques anteriores conservan las características generales aprendidas en ImageNet.
for layer in base_model.layers:
    layer.trainable = layer.name.startswith("block5_")

# Es necesario recompilar tras cambiar qué capas son entrenables.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(),
)

print("Mejor macro F1 de ambas fases:", macro_callback.best)

best_model = keras.models.load_model(BEST_MODEL_PATH)

probabilities = best_model.predict(val_ds, verbose=1)
y_pred = probabilities.argmax(axis=1)

print(classification_report(
    y_val,
    y_pred,
    labels=CLASS_IDS,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
))

recall_per_class = recall_score(
    y_val,
    y_pred,
    labels=CLASS_IDS,
    average=None,
    zero_division=0,
)

new_metrics = {
    "accuracy": float(accuracy_score(y_val, y_pred)),
    "macro_f1": float(f1_score(
        y_val, y_pred,
        labels=CLASS_IDS,
        average="macro",
        zero_division=0,
    )),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_val, y_pred)
    ),
    "recall_malign": float(recall_per_class[MALIGN_IDX]),
}

for name, value in new_metrics.items():
    print(f"{name}: {value:.6f}")

report = classification_report(
    y_val,
    y_pred,
    labels=CLASS_IDS,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report).T.to_csv(
    Path(RESULTS_DIR) / "classification_report.csv"
)

prediction_table = pd.DataFrame({
    "path": val_paths,
    "y_true": y_val,
    "y_pred": y_pred,
})
for idx, class_name in enumerate(CLASS_NAMES):
    prediction_table[f"prob_{class_name}"] = probabilities[:, idx]

prediction_table.to_csv(
    Path(RESULTS_DIR) / "validation_predictions.csv",
    index=False,
)

with open(Path(RESULTS_DIR) / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(new_metrics, f, indent=2, ensure_ascii=False)


# Nombre del modelo experimental
MODEL_PATH = Path(MODELOS_DIR) / "vgg16.keras"

# Exportar el mejor checkpoint seleccionado por Macro F1 en validación.
best_model = keras.models.load_model(BEST_MODEL_PATH)
best_model.save(MODEL_PATH)

print(f"Modelo guardado en: {MODEL_PATH}")
print(f"Tamaño: {MODEL_PATH.stat().st_size / (1024**2):.2f} MB")


cm = confusion_matrix(y_val, y_pred, labels=CLASS_IDS)
cm_normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=axes[0],
)
axes[0].set_title("VGG16 — Validación: recuentos")

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".1%",
    cmap="Blues",
    vmin=0,
    vmax=1,
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=axes[1],
)
axes[1].set_title("VGG16 — Normalizada por clase real")

for ax in axes:
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig(Path(RESULTS_DIR) / "confusion_matrices.png", dpi=160)
plt.show()

# %% Cell 13
from sklearn.metrics import precision_score, recall_score

# Ajusta esta ruta respecto al directorio de ejecución del notebook.
PREVIOUS_RESULTS_PATH = Path(RESULTS_DIR) / "comparativa_modelos.csv"

# Comprobamos que existe el fichero.
if not PREVIOUS_RESULTS_PATH.is_file():
    raise FileNotFoundError(
        f"No se encuentra el CSV: {PREVIOUS_RESULTS_PATH.resolve()}"
    )

previous_results = pd.read_csv(PREVIOUS_RESULTS_PATH)

metric_columns = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "precision_malign",
    "recall_malign",
]

required_columns = ["modelo", "archivo"] + metric_columns

# Añadimos las columnas que falten, inicializadas a NaN.
new_columns = [
    col for col in ["precision_malign","recall_malign"]
    if col not in previous_results.columns
]

if new_columns:
    for col in new_columns:
        previous_results[col] = np.nan

    # Actualizamos el CSV.
    previous_results.to_csv(PREVIOUS_RESULTS_PATH, index=False)

    print(f"Columnas añadidas al CSV: {new_columns}")

# Comprobamos que el esquema ya es correcto.
missing_columns = set(required_columns) - set(previous_results.columns)

if missing_columns:
    raise ValueError(
        f"Faltan columnas en el CSV: {sorted(missing_columns)}"
    )

# Completamos las métricas de VGG16.
# Se calculan todas las clases para que la posición de Malign sea explícita.
class_ids = np.arange(len(CLASS_NAMES))
malign_idx = CLASS_NAMES.index("6. Malign")

precision_per_class = precision_score(
    y_val,
    y_pred,
    labels=class_ids,
    average=None,
    zero_division=0,
)

recall_per_class = recall_score(
    y_val,
    y_pred,
    labels=class_ids,
    average=None,
    zero_division=0,
)

vgg16_results = pd.DataFrame([{
    "modelo": "VGG16",
    "archivo": str(BEST_MODEL_PATH),
    "accuracy": float(new_metrics["accuracy"]),
    "macro_f1": float(new_metrics["macro_f1"]),
    "balanced_accuracy": float(new_metrics["balanced_accuracy"]),
    "recall_malign": float(recall_per_class[malign_idx]),
    "precision_malign": float(precision_per_class[malign_idx]),
}])

vgg16_results.to_csv(
    f"{PREVIOUS_RESULTS_PATH}",
    mode="a",
    header=False,
    index=False
)

# 3. Guardar también VGG16 por separado.
# Este archivo permitirá comparar desde otro notebook.
vgg16_csv = Path(RESULTS_DIR) / "vgg16_metricas_val.csv"
vgg16_csv.parent.mkdir(parents=True, exist_ok=True)

vgg16_results.to_csv(vgg16_csv, index=False)
