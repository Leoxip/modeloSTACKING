import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# =====================
# CARGAR PIPELINE REAL
# =====================
pipeline = joblib.load("pipeline_stacking.joblib")

st.title("Predicción de Demanda Universitaria")
st.markdown("Sistema de apoyo a la planificación universitaria basado en Machine Learning")

st.divider()

# =====================
# FORMULARIO PROFESIONAL
# =====================
st.subheader("Simulación de demanda")

col1, col2, col3 = st.columns(3)

with col1:
    edad = st.slider("Edad postulante", 16, 60, 18)
    sexo = st.selectbox("Sexo", ["MASCULINO","FEMENINO"])
    gestion = st.selectbox("Tipo gestión universidad", ["PÚBLICO","PRIVADO"])

with col2:
    nivel = st.selectbox("Nivel académico", ["CARRERA PROFESIONAL","MAESTRÍA","DOCTORADO"])
    modalidad = st.selectbox("Modalidad ingreso", ["ORDINARIO","TRASLADO","SEGUNDA PROFESIÓN"])
    departamento = st.text_input("Departamento nacimiento", "LIMA")

with col3:
    anio = st.number_input("Año", 2018, 2030, 2024)
    proceso = st.text_input("Proceso admisión", "REGULAR")
    programa = st.text_input("Código programa SIU", "001")

# =====================
# DATAFRAME PARA MODELO
# =====================
input_dict = {
    "POSTULANTE__edad": edad,
    "POSTULANTE__sexo": sexo,
    "POSTULANTE__tipo_gestion": gestion,
    "POSTULANTE__nivel_academico": nivel,
    "POSTULANTE__modalidad_ingreso": modalidad,
    "POSTULANTE__departamento_nacimiento": departamento,
    "POSTULANTE__anio": anio,
    "POSTULANTE__proceso_admision": proceso,
    "POSTULANTE__codigo_siu_programa_primera_opcion": programa
}

df = pd.DataFrame([input_dict])

# =====================
# PREDICCION
# =====================
if st.button("Predecir demanda"):

    pred = pipeline.predict(df)[0]
    pred = int(round(pred))

    st.success(f"Demanda estimada: {pred} estudiantes")

    # gráfico
    fig, ax = plt.subplots()
    ax.bar(["Demanda estimada"], [pred])
    ax.set_ylabel("Número de estudiantes")
    st.pyplot(fig)

    # interpretación académica
    st.subheader("Interpretación")

    if pred > 800:
        st.info("Alta demanda proyectada. Se recomienda ampliar vacantes.")
    elif pred > 400:
        st.warning("Demanda media. Evaluar capacidad instalada.")
    else:
        st.error("Demanda baja. Evaluar sostenibilidad del programa.")
