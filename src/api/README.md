## API de inferencia

*Es necesario tener instalado un entorno Python con las especificaciones indicadas en requirements-api.txt*

- *py -3.10 -m venv entorno_api*
- *entorno_api\Scripts\activate.bat*
- *python -m pip install --upgrade pip*
- *python -m pip install -r requirements_api.txt*

Para lanzar la app, desde la carpeta raíz del repositorio descargado ejecutar:

**python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload**


