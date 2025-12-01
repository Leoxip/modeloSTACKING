# app.py — Streamlit App para Predicción de Demanda Universitaria
# -*- coding: utf-8 -*-

import numpy as np

# 🔑 FUNCIÓN NECESARIA PARA DESERIALIZAR EL PIPELINE
def to_float32(x):
    return x.astype(np.float32)

import streamlit as st
import pandas as pd
import joblib

# ============================
# CONFIG
# ============================
PIPELINE_PATH = "pipeline_stacking.joblib"

# ============================
# LOAD MODEL
# ============================
st.set_page_config(page_title="Predicción Demanda Universitaria", layout="centered")

st.title("🎓 Predicción de Demanda de Carreras Universitarias")
st.write("""
Metamodelo Stacking (RF + XGB + LGBM + SVR + ElasticNet) entrenado con datos nacionales.
""")

@st.cache_resource
def load_model():
    return joblib.load(PIPELINE_PATH)

pipeline = load_model()

# ============================
# FEATURES PRINCIPALES
# ============================

st.subheader("🔎 Ingrese los datos")

input_data = {}

# Variables "human-friendly"
input_data["POSTULANTE__edad"] = st.slider(
    "Edad del postulante",
    min_value=16,
    max_value=60,
    value=18
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
# CONSTRUIR DATAFRAME
# ============================
df_input = pd.DataFrame([input_data])

# ============================
# AJUSTAR COLUMNAS DEL PIPELINE
# ============================

expected_cols = pipeline.named_steps["preprocess"].feature_names_in_

for col in expected_cols:
    if col not in df_input.columns:
        df_input[col] = None

df_input = df_input[expected_cols]

# ============================
# PREDICCIÓN
# ============================

if st.button("📊 Predecir demanda"):
    try:
        pred = pipeline.predict(df_input)[0]
        st.success(
            f"👉 Demanda estimada de matrícula: **{int(round(pred))} estudiantes**"
        )
    except Exception as e:
        st.error("❌ Error al generar la predicción")
        st.exception(e)
