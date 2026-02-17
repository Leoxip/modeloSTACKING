import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predicción de Demanda Universitaria", layout="wide")

# ==========================
# CARGAR PIPELINE
# ==========================
pipeline = joblib.load("pipeline_stacking.joblib")
columnas_modelo = pipeline.feature_names_in_

st.title("Sistema Inteligente de Predicción de Demanda Universitaria")
st.markdown("Modelo basado en **Machine Learning Stacking Ensemble**")
st.divider()

# ==========================
# FORMULARIO
# ==========================
st.subheader("Simulador de Postulación")

c1, c2, c3 = st.columns(3)

with c1:
    edad = st.slider("Edad postulante", 16, 60, 18)
    sexo = st.selectbox("Sexo", ["MASCULINO","FEMENINO"])
    gestion = st.selectbox("Tipo gestión universidad", ["Público","Privado"])

with c2:
    nivel = st.selectbox("Nivel académico", ["Carrera Profesional","Maestría","Doctorado"])
    modalidad = st.selectbox("Modalidad ingreso", ["Ordinario","Traslado","Segunda profesión"])
    departamento = st.text_input("Departamento nacimiento", "Lima")

with c3:
    anio = st.number_input("Año de proceso", 2018, 2035, 2024)
    proceso = st.text_input("Proceso admisión", "Regular")
    programa = st.text_input("Código programa SIU", "001")

# ==========================
# CREAR FILA CORRECTA
# ==========================
fila = {}

for col in columnas_modelo:

    # columnas numericas detectadas por nombre
    if any(x in col.lower() for x in ["edad","anio","codigo","numero","cantidad"]):
        fila[col] = 0

    else:
        fila[col] = "DESCONOCIDO"

# sobrescribir con valores reales
if "POSTULANTE__edad" in fila:
    fila["POSTULANTE__edad"] = edad

if "POSTULANTE__sexo" in fila:
    fila["POSTULANTE__sexo"] = sexo

if "POSTULANTE__tipo_gestion" in fila:
    fila["POSTULANTE__tipo_gestion"] = gestion

if "POSTULANTE__nivel_academico" in fila:
    fila["POSTULANTE__nivel_academico"] = nivel

if "POSTULANTE__modalidad_ingreso" in fila:
    fila["POSTULANTE__modalidad_ingreso"] = modalidad

if "POSTULANTE__departamento_nacimiento" in fila:
    fila["POSTULANTE__departamento_nacimiento"] = departamento

if "POSTULANTE__anio" in fila:
    fila["POSTULANTE__anio"] = anio

if "POSTULANTE__proceso_admision" in fila:
    fila["POSTULANTE__proceso_admision"] = proceso

if "POSTULANTE__codigo_siu_programa_primera_opcion" in fila:
    fila["POSTULANTE__codigo_siu_programa_primera_opcion"] = programa

# dataframe final compatible con pipeline
df = pd.DataFrame([fila])
df = df[columnas_modelo]

# ==========================
# PREDICCION
# ==========================
st.divider()

if st.button("Predecir Demanda Universitaria"):

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
    st.subheader("Interpretación del Resultado")

    if pred > 800:
        st.info("Alta demanda proyectada")
        nivel_demanda = "ALTA"
    elif pred > 400:
        st.warning("Demanda media proyectada")
        nivel_demanda = "MEDIA"
    else:
        st.error("Demanda baja proyectada")
        nivel_demanda = "BAJA"

    # panel analítico
    st.divider()
    st.subheader("Resumen Analítico")

    k1, k2, k3 = st.columns(3)
    k1.metric("Demanda estimada", pred)
    k2.metric("Nivel proyectado", nivel_demanda)
    k3.metric("Año simulado", anio)

    # texto tesis automático
    st.divider()
    st.subheader("Descripción automática para informe")

    st.write(f"""
    El sistema de predicción basado en aprendizaje automático
    estima que la demanda proyectada será de **{pred} estudiantes**,
    clasificándose como una demanda **{nivel_demanda.lower()}**.
    Este resultado permite planificar recursos académicos
    con base en evidencia predictiva.
    """)
