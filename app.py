import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# =====================
# CARGAR PIPELINE
# =====================
pipeline = joblib.load("pipeline_stacking.joblib")

st.title("🎓 Predicción de demanda universitaria")
st.markdown("Sistema basado en Stacking Ensemble de Machine Learning")

# =====================
# FORMULARIO REALISTA
# =====================
st.subheader("Simulador de escenario")

data = {}

# ---- VARIABLES NUMERICAS ----
data["INGRESANTE__edad"] = st.slider("Edad ingresante", 16, 60, 18)
data["DOCENTE__edad"] = st.slider("Edad docente", 25, 80, 45)
data["POSTULANTE__edad"] = st.slider("Edad postulante", 16, 60, 18)

# ---- VARIABLES CATEGORICAS ----
data["INGRESANTE__sexo"] = st.selectbox(
    "Sexo ingresante", ["MASCULINO", "FEMENINO"]
)

data["INGRESANTE__tipo_gestion"] = st.selectbox(
    "Tipo gestión universidad", ["Público", "Privado"]
)

data["INGRESANTE__nivel_academico"] = st.selectbox(
    "Nivel académico",
    ["Carrera Profesional", "Maestría", "Doctorado"]
)

data["POSTULANTE__modalidad_ingreso_grupo"] = st.selectbox(
    "Modalidad ingreso",
    ["Regular", "Extraordinario"]
)

data["INGRESANTE__departamento_filial"] = st.selectbox(
    "Departamento filial",
    ["Lima","Arequipa","Cajamarca","Ayacucho","Loreto"]
)

# =====================
# CONSTRUIR DATAFRAME
# =====================
df = pd.DataFrame([data])

# =====================
# PREDICCION
# =====================
if st.button("Predecir demanda"):

    pred = pipeline.predict(df)[0]

    st.success(f"Demanda estimada: {int(pred)} estudiantes")

    # gráfico
    fig, ax = plt.subplots()
    ax.bar(["Demanda estimada"], [pred])
    ax.set_ylabel("Estudiantes")
    st.pyplot(fig)

    # interpretación automática
    st.subheader("Interpretación")

    if pred > 800:
        st.info("Alta demanda proyectada")
    elif pred > 400:
        st.warning("Demanda media proyectada")
    else:
        st.error("Demanda baja proyectada")
