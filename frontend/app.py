import streamlit as st
import requests
import pandas as pd

# Dirección de la API
API_URL = "http://127.0.0.1:8000/predict"

# Configuración general de la página
st.set_page_config(
    page_title="DermatologIA | Clasificación dermatológica",
    page_icon="🔬",
    layout="wide"
)

# Cabecera de la aplicación
st.title("DermatologIA")

st.subheader(
    "Sistema de clasificación dermatológica mediante inteligencia artificial"
)

st.caption(
    "Herramienta académica de apoyo para la clasificación "
    "de imágenes dermatológicas."
)

st.divider()

# Sección para cargar la imagen
st.header("Análisis de imagen")

st.write(
    "Sube una imagen dermatológica en formato JPG, JPEG o PNG "
    "para realizar la clasificación."
)

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)

# Mostrar la imagen seleccionada
if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Imagen seleccionada para el análisis",
        use_container_width=True
    )

# Botón para iniciar el análisis
if uploaded_file is not None:
    if st.button(
        "🔬 Analizar imagen",
        type="primary",
        use_container_width=True
    ):
        with st.spinner("Analizando imagen con inteligencia artificial..."):
            try:
                # Enviar la imagen a la API
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(API_URL, files=files)

                # Comprobar la respuesta
                if response.status_code == 200:
                    result = response.json()
                    st.success("¡Análisis completado correctamente!")
                    st.subheader("Resultados de la clasificación")
                    st.write("**Categoría predicha:**")
                    st.info(result["predicted_class"])
                    st.metric(
                        label="Puntuación del modelo",
                        value=f"{result['confidence'] * 100:.2f}%",
                    )
                    # Distribución de puntuaciones por categoría
                    st.subheader("Distribución por categorías")

                    probabilities = result["probabilities"]

                    df_probabilities = pd.DataFrame({
                        "Categoría": list(probabilities.keys()),
                        "Puntuación (%)": [
                        value * 100 for value in probabilities.values()
                        ]
                    })
                    st.dataframe(df_probabilities)
                else:
                    st.error(f"Error en la API: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"No se pudo conectar con la API: {e}")


st.divider()

st.warning(
    "Aviso importante: DermatologIA es un prototipo académico "
    "desarrollado con fines educativos y de investigación. "
    "Sus resultados son experimentales y no constituyen un "
    "diagnóstico médico ni sustituyen la valoración de un "
    "profesional sanitario."
)



