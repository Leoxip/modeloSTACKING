import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predicción de Demanda Universitaria", layout="wide")

# ==========================
# CARGAR PIPELINE
# ==========================
pipeline = joblib.load("pipeline_stacking.joblib")

# columnas esperadas por el modelo
columnas_modelo = list(pipeline.feature_names_in_)

st.title("Sistema Inteligente de Predicción de Demanda Universitaria")
st.markdown("Modelo basado en **Stacking Ensemble Machine Learning**")
st.divider()

# ==========================
# FORMULARIO
# ==========================
st.subheader("Simulador de Postulación")

c1, c2, c3 = st.columns(3)

with c1:
    edad = st.slider("Edad postulante", 16, 60, 18)
    sexo = st.selectbox("Sexo", ["MASCULINO","FEMENINO"])
    gestion = st.selectbox("Tipo gestión universidad", ["PÚBLICO","PRIVADO"])

with c2:
    nivel = st.selectbox("Nivel académico", ["CARRERA PROFESIONAL","MAESTRÍA","DOCTORADO"])
    modalidad = st.selectbox("Modalidad ingreso", ["ORDINARIO","TRASLADO","SEGUNDA PROFESIÓN"])
    departamento = st.text_input("Departamento nacimiento", "LIMA")

with c3:
    anio = st.number_input("Año de proceso", 2018, 2035, 2024)
    proceso = st.text_input("Proceso admisión", "REGULAR")
    programa = st.text_input("Código programa SIU", "001")

# ==========================
# CREAR DATAFRAME EXACTO
# ==========================
# crear dataframe vacío con columnas del modelo
df = pd.DataFrame(columns=columnas_modelo)

# crear fila con NaN (para que imputador funcione)
fila = {col: np.nan for col in columnas_modelo}

# sobrescribir SOLO si la columna existe en el modelo
def asignar(col, val):
    if col in fila:
        fila[col] = val

asignar("POSTULANTE__edad", edad)
asignar("POSTULANTE__sexo", sexo)
asignar("POSTULANTE__tipo_gestion", gestion)
asignar("POSTULANTE__nivel_academico", nivel)
asignar("POSTULANTE__modalidad_ingreso", modalidad)
asignar("POSTULANTE__departamento_nacimiento", departamento)
asignar("POSTULANTE__anio", anio)
asignar("POSTULANTE__proceso_admision", proceso)
asignar("POSTULANTE__codigo_siu_programa_primera_opcion", programa)

df = pd.DataFrame([fila])
df = df[columnas_modelo]

# ==========================
# DEBUG OPCIONAL (IMPORTANTE)
# ==========================
faltantes = set(columnas_modelo) - set(df.columns)

if faltantes:
    st.error("⚠️ El modelo espera columnas que no están en el DataFrame")
    st.write(faltantes)
    st.stop()

# ==========================
# PREDICCION
# ==========================
st.divider()

if st.button("Predecir Demanda Universitaria"):

    try:
        pred = pipeline.predict(df)[0]
        pred = int(round(pred))

        st.success(f"Demanda estimada: {pred} estudiantes")

        # gráfico
        fig, ax = plt.subplots()
        ax.bar(["Demanda estimada"], [pred])
        ax.set_ylabel("Número de estudiantes")
        ax.set_title("Resultado de Predicción")
        st.pyplot(fig)

        # interpretación
        st.subheader("Interpretación")

        if pred > 800:
            nivel_demanda = "ALTA"
            st.info("Alta demanda proyectada")
        elif pred > 400:
            nivel_demanda = "MEDIA"
            st.warning("Demanda media proyectada")
        else:
            nivel_demanda = "BAJA"
            st.error("Demanda baja proyectada")

        # panel analítico
        st.divider()
        st.subheader("Resumen Analítico")

        k1, k2, k3 = st.columns(3)
        k1.metric("Demanda estimada", pred)
        k2.metric("Nivel proyectado", nivel_demanda)
        k3.metric("Año simulado", anio)

        # texto automático tesis
        st.divider()
        st.subheader("Descripción automática para informe")

        st.write(f"""
        El modelo de aprendizaje automático estima una demanda proyectada de 
        **{pred} estudiantes**, clasificada como demanda **{nivel_demanda.lower()}**.
        Este resultado permite optimizar la planificación académica basada en evidencia.
        """)

    except Exception as e:
        st.error("El modelo lanzó un error durante la predicción")
        st.exception(e)
