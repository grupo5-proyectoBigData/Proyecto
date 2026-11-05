
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
│   └── NF1_Presentacion_alineacion_viabilidad.pdf
│   └── NF2_Documentacion_y_Diseno_Tecnico.pdf
│
├── src/
│   └── api/
│   └── data
│   └── frontend
│   └── models  
│
├── data/
│   └── processed/
|       └── train/
|       └── test/
|       └── val/
│   └── raw/
|       └── train/
|       └── test/
|       └── val/
│   └── served/
│
├── notebooks/
|
├── environment/
│
├── README.md
├── .gitignore
├── requirements.txt




## Ejecución del backend (API)

### Requisitos
- Python 3.9
- Modelo entrenado `mobilenetv2.keras`
- Dependencias indicadas en `requirements.txt`

### Instalación

Crear y activar el entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

### Arranque de la API

Desde la raíz del proyecto:

```bash
python -m uvicorn src.api.main:app --reload
```

La API estará disponible en:

- API: http://127.0.0.1:8000
- Documentación interactiva: http://127.0.0.1:8000/docs
- Estado del servicio: http://127.0.0.1:8000/health

### Endpoint de predicción

`POST /predict`

Recibe una imagen JPEG, PNG o WEBP y devuelve:
- Clase predicha.
- Nivel de confianza.
- Probabilidades de las seis categorías.

### Ejecución de pruebas

Desde la raíz del proyecto:

```bash
python -m pytest -v
```
