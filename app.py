# app.py — Streamlit App Final Tesis
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import pandas as pd
import joblib

# 🔑 FUNCIÓN NECESARIA PARA DESERIALIZAR EL PIPELINE
def to_float32(x):
    return x.astype(np.float32)

# ============================
# CONFIG
# ============================

PIPELINE_PATH = "pipeline_stacking.joblib"

st.set_page_config(
    page_title="Predicción Demanda Universitaria",
    layout="centered"
)

st.title("🎓 Predicción de Demanda Universitaria")
st.markdown("""
Metamodelo de **Stacking Ensemble**
(Random Forest + XGBoost + LightGBM + ElasticNet)

Modelo entrenado con datos nacionales reales.
""")

# ============================
# LOAD MODEL
# ============================

@st.cache_resource
def load_model():
    return joblib.load(PIPELINE_PATH)

pipeline = load_model()

# ============================
# FORMULARIO
# ============================

st.subheader("📊 Ingrese datos del postulante")

input_data = {}

input_data["POSTULANTE__edad"] = st.slider(
    "Edad del postulante",
    16, 60, 18
)

input_data["POSTULANTE__sexo"] = st.selectbox(
    "Sexo",
    ["MASCULINO", "FEMENINO"]
)

input_data["POSTULANTE__tipo_gestion"] = st.selectbox(
    "Tipo de gestión",
    ["PÚBLICO", "PRIVADO"]
)

input_data["POSTULANTE__nivel_academico"] = st.selectbox(
    "Nivel académico",
    ["CARRERA PROFESIONAL", "SEGUNDA ESPECIALIDAD", "MAESTRÍA", "DOCTORADO"]
)

input_data["POSTULANTE__modalidad_ingreso"] = st.selectbox(
    "Modalidad de ingreso",
    ["ORDINARIO", "EXTRAORDINARIO", "TRASLADO", "SEGUNDA PROFESIÓN"]
)

input_data["POSTULANTE__departamento_nacimiento"] = st.text_input(
    "Departamento de nacimiento",
    value="LIMA"
)

input_data["POSTULANTE__codigo_siu_programa_primera_opcion"] = st.text_input(
    "Código SIU del programa",
    value="001234"
)

# ============================
# DATAFRAME
# ============================

df_input = pd.DataFrame([input_data])

# Ajustar columnas exactamente igual al entrenamiento
expected_cols = pipeline.named_steps["preprocess"].feature_names_in_

for col in expected_cols:
    if col not in df_input.columns:
        df_input[col] = None

df_input = df_input[expected_cols]

# ============================
# PREDICCIÓN
# ============================

if st.button("🔮 Predecir demanda"):

    try:
        pred = pipeline.predict(df_input)[0]
        pred = int(round(pred))

        st.success(
            f"📈 Demanda estimada para este perfil: **{pred} estudiantes**"
        )

        # Indicador visual adicional (para tesis)
        st.metric(
            label="Demanda proyectada",
            value=pred
        )

        # Gráfico simple
        chart_data = pd.DataFrame({
            "Tipo": ["Demanda estimada"],
            "Estudiantes": [pred]
        })

        st.bar_chart(chart_data.set_index("Tipo"))

    except Exception as e:
        st.error("❌ Error al generar predicción")
        st.exception(e)
