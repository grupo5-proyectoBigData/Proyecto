
# Ayuda al diagnóstico de enfermedades en la piel
Aplicación destinada a determinar el tipo de lesión de la piel con el objetivo de facilitar su priorización y orientar el tratamiento.

## 👥 Equipo

- Antonio Jesús Terrón Moreno — Project Lead & BI
- Guimel Barrientos Ustariz — Data & Frontend Developer
- David Codón Agüera — ML
- Bruno Díaz Riera — Platform
  
## 📁 Estructura
```text
repo/
├── docs/
│   ├── NF1_Presentacion_alineacion_viabilidad.pdf
│   ├── NF2_Documentacion_y_Diseno_Tecnico.pdf
│   ├── fase1_grupo5.docx
│   ├── fase1_grupo5.pdf
│   ├── riesgos.md
│   └── README.md
│
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   ├── frontend/
│   ├── models/
│   │   ├── vgg16.keras
│   │   ├── mobilenetv2.keras
│   │   ├── baseline_efficientnetv2b0.keras
│   │   ├── efficientnetv2b0_CW_y_DA.keras
│   │   ├── 03_efficientnetv2b0_CW_y_DA.keras
│   │   ├── metadata.json
│   │   └── resultados/
│   │       ├── metrics.json
│   │       ├── informe_vgg16_val.csv
│   │       ├── vgg16_metricas_val.csv
│   │       ├── informe_mobilenetv2_val.csv
│   │       ├── mobilenetv2_metricas_val.csv
│   │       ├── informe_da_cw_val.csv
│   │       ├── baseline_history.csv
│   │       ├── efficientnetv2b0_CW_y_DA_history.csv
│   │       ├── comparativa_modelos.csv
│   │       ├── classification_report.csv
│   │       ├── validation_predictions.csv
│   │       ├── validation_manifest.csv
│   │       ├── predicciones_vgg16_val.npz
│   │       ├── predicciones_mobilenetv2_val.npz
│   │       ├── predicciones_da_cw_val.npz
│   │       ├── predicciones_baseline_val.npz
│   │       ├── confusion_matrices.png
│   │       └── comparativa_matrices.png
│   ├── paths.py
│   └── README.md
│
├── data/
│   ├── processed/
│   ├── raw/
│   ├── splits/
│   └── README.md
│
├── notebooks/
│   ├── 01_data.ipynb
│   ├── 02_baseline_efficientnetv2b0.ipynb
│   ├── 02_exp01_dataaugmentation_classweights.ipynb
│   ├── 02_exp02_mobilenetv2.ipynb
│   ├── 02_exp02_vgg16.ipynb
│   ├── comparativa_experimentos.ipynb
│   ├── entrenamiento_imagenes.ipynb
│   ├── Comprobar_gpu.ipymb.ipynb
│   └── README.md
│
├── environment/
│   ├── .gitignore
│   ├── requeriments.txt
│   └── README.md
│
├── tests/
│   ├── test_api.py
│   └── test_rendimiento.py
│
├── README.md
├── .gitignore
├── .gitattributes
└── requirements.txt


Ejecución del backend (API VGG16)

1. Descripción

El backend utiliza FastAPI para ofrecer un servicio de clasificación de imágenes dermatológicas mediante el modelo VGG16 entrenado para seis categorías.

El sistema permite cargar una imagen y obtener la clase predicha, su nivel de confianza y las probabilidades correspondientes a las seis categorías.

2. Requisitos previos

Python 3.9.

Git.

Git LFS para descargar el modelo VGG16.

Entorno virtual de Python.

Dependencias indicadas en requirements.txt. Nota: requirements.txt actual utiliza tensorflow-macos, por lo que esta configuración está orientada a macOS 

3. Instalación

Clonar el repositorio y acceder a la carpeta del proyecto:

git clone -b Bruno-2 https://github.com/grupo5-proyectoBigData/Proyecto.git
cd Proyecto

Descargar los archivos grandes mediante Git LFS:

git lfs install
git lfs pull

Crear y activar el entorno virtual:

python3 -m venv .venv
source .venv/bin/activate

Instalar las dependencias:

python -m pip install -r requirements.txt

4. Arranque del backend

Desde la carpeta raíz del proyecto, ejecutar:

python -m uvicorn src.api.main:app --reload

La API estará disponible en:

API: http://127.0.0.1:8000

Documentación Swagger: http://127.0.0.1:8000/docs

5. Endpoints disponibles

Método

Endpoint

Descripción

GET

/

Información general de la API

GET

/health

Comprobación del estado del servicio y del modelo

POST

/predict

Clasificación de una imagen

El endpoint /predict admite imágenes JPEG, PNG y WEBP.

La respuesta incluye la categoría predicha, la confianza y las probabilidades de las seis clases.

6. Modelo utilizado

Arquitectura: VGG16.

Tamaño de entrada: 224 × 224 píxeles.

Canales: RGB.

Número de categorías: 6.

Archivo: src/models/vgg16.keras.

Preprocesamiento: función preprocess_input específica de VGG16.

El modelo se carga durante el inicio de la aplicación.

7. Ejecución de pruebas automatizadas

Para ejecutar las pruebas:

python -m pytest -v