import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predicción de Demanda Universitaria", layout="wide")

# ==========================
# CARGAR PIPELINE
# ==========================
pipeline = joblib.load("pipeline_stacking.joblib")

# columnas esperadas por el modelo
columnas_modelo = pipeline.feature_names_in_

# ==========================
# TITULO
# ==========================
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
# CREAR FILA COMPLETA
# ==========================
fila = {}

# rellenar columnas faltantes automáticamente
for col in columnas_modelo:
    fila[col] = "DESCONOCIDO"

# sobrescribir con inputs reales
fila["POSTULANTE__edad"] = edad
fila["POSTULANTE__sexo"] = sexo
fila["POSTULANTE__tipo_gestion"] = gestion
fila["POSTULANTE__nivel_academico"] = nivel
fila["POSTULANTE__modalidad_ingreso"] = modalidad
fila["POSTULANTE__departamento_nacimiento"] = departamento
fila["POSTULANTE__anio"] = anio
fila["POSTULANTE__proceso_admision"] = proceso
fila["POSTULANTE__codigo_siu_programa_primera_opcion"] = programa

df = pd.DataFrame([fila])
df = df[columnas_modelo]  # mantener orden correcto

# ==========================
# BOTON PREDICCION
# ==========================
st.divider()

if st.button("Predecir Demanda Universitaria"):

    pred = pipeline.predict(df)[0]
    pred = int(round(pred))

    # ======================
    # RESULTADO PRINCIPAL
    # ======================
    st.success(f"Demanda estimada: {pred} estudiantes")

    # ======================
    # GRAFICO
    # ======================
    fig, ax = plt.subplots()
    ax.bar(["Demanda estimada"], [pred])
    ax.set_ylabel("Número de estudiantes")
    ax.set_title("Resultado de Predicción")
    st.pyplot(fig)

    # ======================
    # INTERPRETACION
    # ======================
    st.subheader("Interpretación del Resultado")

    if pred > 800:
        st.info("Alta demanda proyectada. Se recomienda ampliar vacantes y recursos académicos.")
        nivel_demanda = "ALTA"
    elif pred > 400:
        st.warning("Demanda media proyectada. Se recomienda monitoreo y planificación moderada.")
        nivel_demanda = "MEDIA"
    else:
        st.error("Demanda baja proyectada. Se recomienda evaluar promoción o rediseño del programa.")
        nivel_demanda = "BAJA"

    # ======================
    # PANEL ANALITICO
    # ======================
    st.divider()
    st.subheader("Resumen Analítico")

    k1, k2, k3 = st.columns(3)
    k1.metric("Demanda estimada", pred)
    k2.metric("Nivel proyectado", nivel_demanda)
    k3.metric("Año simulado", anio)

    # ======================
    # TEXTO AUTOMATICO TESIS
    # ======================
    st.divider()
    st.subheader("Descripción automática para informe")

    st.write(f"""
    El sistema de predicción basado en técnicas de aprendizaje automático
    estima que la demanda para el escenario simulado será de **{pred} estudiantes**,
    clasificándose como una demanda **{nivel_demanda.lower()}**.
    Este resultado permite a las autoridades universitarias planificar
    la asignación de vacantes, recursos docentes y logística académica
    con base en evidencia predictiva.
    """)

