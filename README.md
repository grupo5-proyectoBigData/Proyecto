
# Ayuda al diagnóstico de enfermedades en la piel
Proyecto de Inteligencia Artificial orientado a la clasificación de lesiones cutáneas mediante técnicas de Deep Learning y visión artificial.

Aplicación destinada a determinar el tipo de lesión de la piel con el objetivo de facilitar su priorización y orientar el tratamiento.

**Importante:** Los resultados obtenidos por el uso de esta aplicación no constituyen un diagnóstico médico. es un proyecto experimental con fines educativos.

## 👥 Equipo

- Antonio Jesús Terrón Moreno — Project Lead & BI
- Guimel Barrientos Ustariz — Data & Frontend Developer
- David Codón Agüera — ML
- Bruno Díaz Riera — Platform
  
## 📁 Estructura
```text
repo/
├── docs/
├── src/
│   ├── api/
│   ├── frontend/
│   └── models/
├── data/
│   ├── processed/
│   ├── raw/
│   └── splits/
├── notebooks/
├── environment/
├── tests/
├── README.md
└── requirements.txt
```text

## Ejecución del backend (API VGG16)
### 1. Descripción

El backend utiliza FastAPI para ofrecer un servicio de clasificación de imágenes dermatológicas mediante el modelo VGG16 entrenado para seis categorías.

El sistema permite cargar una imagen y obtener la clase predicha, su nivel de confianza y las probabilidades correspondientes a las seis categorías.

### 2. Requisitos previos:
- Git.
- Git LFS para descargar el modelo VGG16.

- Separamos tres instalaciones distintas, para que puedan ejecutarse cada fase del proyecto en sistemas independientes, sin tener que descargar todo el proyecto. 

**EDA y Modelado:** 
    Necesitamos el dataset decargado de Kaggle. Los datos procesados se guardan en la carpeta data/processed. 
    - *git clone --filter=blob:none --sparse https://github.com/grupo5-proyectoBigData/Proyecto.git Proyecto-Data*
    - *cd Proyecto-Data*
    - *git sparse-checkout set src/data environment docs data notebooks*
    - *git lfs install*
    - *git lfs pull --include="src/models/**"*

**API** 
    - *git clone --filter=blob:none --sparse https://github.com/grupo5-proyectoBigData/Proyecto.git Proyecto-API*
    - *cd Proyecto-API*
    - *git sparse-checkout set src/frontend environment docs*
    - *git lfs install*
    - *git lfs pull --include="src/models/vgg16.keras"*


**Aplicación web - Frontend**
    - *git clone --filter=blob:none --sparse --branch v1.0.0 https://github.com/grupo5-proyectoBigData/Proyecto.git Proyecto-Frontend*
    - *cd Proyecto-Frontend*
    - *git sparse-checkout set src/frontend environment docs*


### 3. Instalación
- Atendiendo nuevamente a la separación por tareas, Los ficheros *README.md* en cada carpeta *src/* incluyen instrucciones precisas para la configuración de los entornos de cada fase, así como información precisa sobre su uso, si fuera necesario.


### 4. Arranque del backend

Desde la carpeta raíz del proyecto, ejecutar:
*python -m uvicorn src.api.main:app --reload*

- La API estará disponible en la URL: http://127.0.0.1:8000
- Documentación Swagger: http://127.0.0.1:8000/docs

### 5. Endpoints disponibles

| Método | Endpoint | Descripción |
| :--- | :---: | ---: |
| GET | / | Información general de la API |
| GET | /health | Comprobación del estado del servicio y del modelo |
| POST | /predict | Clasificación de una imagen (JPEG, PNG y WEBP) |

La respuesta incluye la categoría predicha, la confianza y las probabilidades de las seis clases.

### 6. Modelo utilizado

Arquitectura: VGG16.

Tamaño de entrada: 224 × 224 píxeles.

Canales: RGB.

Número de categorías: 6.

Archivo: src/models/vgg16.keras.

Preprocesamiento: función preprocess_input específica de VGG16.

El modelo se carga durante el inicio de la aplicación.

### 7. Ejecución de pruebas automatizadas

Para ejecutar las pruebas:

python -m pytest -v

### 8. Lanzar aplicación de usuario (Frontend)
Para lanzar la app, desde la carpeta raíz del repositorio descargado ejecutar:

*python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload*