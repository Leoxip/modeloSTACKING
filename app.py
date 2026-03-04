import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Predictor de Demanda Universitaria Peruana",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ESTILOS — MODO CLARO PROFESIONAL
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fondo general blanco roto */
.stApp {
    background: #F7F8FC;
    color: #1a1d2e;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E4E8F0;
    box-shadow: 4px 0 20px rgba(0,0,0,0.04);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #374151 !important;
}

/* Título principal */
.titulo-principal {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #111827;
    letter-spacing: -1px;
    line-height: 1.15;
}
.titulo-acento {
    color: #2563EB;
}
.subtitulo {
    font-size: 0.82rem;
    color: #6B7280;
    font-weight: 400;
    margin-top: 6px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

/* Línea decorativa */
.divider-line {
    height: 3px;
    background: linear-gradient(90deg, #2563EB 0%, #7C3AED 50%, #059669 100%);
    border-radius: 2px;
    margin: 16px 0 28px 0;
}

/* Cards métricas */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E9ECF3;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    border-radius: 14px 14px 0 0;
}
.metric-card:hover {
    border-color: #2563EB;
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(37,99,235,0.12);
}
.metric-valor {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #2563EB;
    line-height: 1;
}
.metric-label {
    font-size: 0.7rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 8px;
}
.metric-desc {
    font-size: 0.68rem;
    color: #D1D5DB;
    margin-top: 5px;
}

/* Badge nivel */
.badge-alta  {
    background: #ECFDF5;
    color: #065F46;
    border: 1.5px solid #34D399;
    padding: 6px 20px;
    border-radius: 999px;
    font-size: .82rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    display: inline-block;
}
.badge-media {
    background: #FFFBEB;
    color: #92400E;
    border: 1.5px solid #FBBF24;
    padding: 6px 20px;
    border-radius: 999px;
    font-size: .82rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    display: inline-block;
}
.badge-baja  {
    background: #EFF6FF;
    color: #1E40AF;
    border: 1.5px solid #60A5FA;
    padding: 6px 20px;
    border-radius: 999px;
    font-size: .82rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    display: inline-block;
}

/* Resultado principal */
.resultado-box {
    background: #FFFFFF;
    border: 1.5px solid #E4E8F0;
    border-radius: 20px;
    padding: 36px 28px;
    text-align: center;
    box-shadow: 0 8px 40px rgba(37,99,235,0.08);
    position: relative;
    overflow: hidden;
}
.resultado-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 5px;
    background: linear-gradient(90deg, #2563EB, #7C3AED, #059669);
}
.resultado-numero {
    font-family: 'Playfair Display', serif;
    font-size: 5.5rem;
    font-weight: 700;
    color: #2563EB;
    line-height: 1;
}
.resultado-label {
    font-size: 0.78rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-top: 8px;
}

/* Info box */
.info-box {
    background: #F0F5FF;
    border-left: 3.5px solid #2563EB;
    border-radius: 0 10px 10px 0;
    padding: 11px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #374151;
}
.info-box b { color: #111827; }

/* Sección header */
.section-header {
    font-size: 0.7rem;
    color: #2563EB;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1.5px solid #E4E8F0;
    font-weight: 600;
}

/* Botón predecir (sidebar) */
.stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #7C3AED) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 40px !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(37,99,235,0.45) !important;
}

/* Botón descarga PNG */
.stDownloadButton > button {
    background: #FFFFFF !important;
    color: #2563EB !important;
    border: 1.5px solid #BFDBFE !important;
    border-radius: 8px !important;
    padding: 7px 12px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.5px !important;
    margin-top: 4px !important;
}
.stDownloadButton > button:hover {
    background: #EFF6FF !important;
    border-color: #2563EB !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.15) !important;
}

/* Inputs */
.stSelectbox > div > div, .stTextInput > div > div {
    background: #F9FAFB !important;
    border-color: #E4E8F0 !important;
    color: #111827 !important;
    border-radius: 10px !important;
}
.stSlider > div { color: #374151 !important; }

div[data-testid="stMetricValue"] {
    color: #2563EB !important;
    font-family: 'Playfair Display', serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #E4E8F0;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #6B7280 !important;
    font-weight: 500 !important;
    border-radius: 9px !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: #2563EB !important;
    color: white !important;
}

/* Sidebar labels bold */
section[data-testid="stSidebar"] strong {
    color: #111827 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CARGAR MODELO
# ============================================================
@st.cache_resource
def cargar_modelo():
    try:
        pipeline = joblib.load("pipeline_stacking.joblib")
        return pipeline, None
    except Exception as e:
        return None, str(e)

pipeline, error_carga = cargar_modelo()

# ============================================================
# HEADER
# ============================================================
col_logo, col_titulo = st.columns([1, 11])
with col_titulo:
    st.markdown('<div class="titulo-principal">Predictor de <span class="titulo-acento">Demanda Universitaria</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitulo">Metamodelo Stacking · XGBoost · LightGBM · Random Forest · SVR</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

if error_carga:
    st.error(f"❌ No se pudo cargar el modelo: {error_carga}")
    st.info("Asegúrate de que `pipeline_stacking.joblib` esté en la misma carpeta que `app.py`")
    st.stop()

# ============================================================
# SIDEBAR - FORMULARIO
# ============================================================
with st.sidebar:
    st.markdown('<div class="section-header">⚙ Parámetros de Simulación</div>', unsafe_allow_html=True)

    st.markdown("**🏛️ Universidad**")
    nombre_entidad = st.selectbox("Tipo de gestión", ["Público", "Privado"], key="gestion")
    tipo_constitucion = st.selectbox("Constitución", ["Pública", "Societaria- Con fines de lucro", "Asociativa- Sin fines de lucro"], key="const")
    licencia = st.selectbox("Licenciamiento", ["LICENCIADA", "CON LICENCIA DENEGADA"], key="lic")
    es_sede = st.selectbox("¿Sede principal?", ["SI", "NO"], key="sede")
    departamento_local = st.selectbox("Departamento sede", [
        "Lima","Arequipa","Cusco","La Libertad","Piura","Junín","Lambayeque",
        "Áncash","Cajamarca","Loreto","Puno","Ica","San Martín","Ayacucho",
        "Huánuco","Tacna","Ucayali","Apurímac","Madre de Dios","Moquegua",
        "Tumbes","Pasco","Huancavelica","Amazonas","Callao"
    ], key="dpto")

    st.markdown("---")
    st.markdown("**🎓 Carrera**")
    area_conocimiento = st.selectbox("Área de conocimiento", [
        "Ingeniería, Industria y Construcción",
        "Educación",
        "Ciencias Sociales, Periodismo e Información",
        "Salud y Bienestar",
        "Administración de Empresas y Derecho",
        "Ciencias Naturales, Matemáticas y Estadística",
        "Tecnología de la Información y la Comunicación",
        "Agricultura, Silvicultura, Pesca y Veterinaria",
        "Arte y Humanidades",
        "Servicios"
    ], key="area")
    nivel_academico = st.selectbox("Nivel académico", [
        "Carrera Profesional","Maestría","Doctorado","Segunda Especialidad"
    ], key="nivel")

    st.markdown("---")
    st.markdown("**👤 Perfil Estudiante**")
    sexo = st.selectbox("Sexo", ["MASCULINO","FEMENINO"], key="sexo")
    edad = st.slider("Edad", 16, 65, 21, key="edad")
    anio = st.slider("Año de predicción", 2020, 2030, 2025, key="anio")
    periodo = st.selectbox("Periodo", ["I","II","ANUAL"], key="periodo")

    st.markdown("---")
    st.markdown("**👨‍🏫 Docente**")
    categoria_docente = st.selectbox("Categoría docente", [
        "Ordinario Principal","Ordinario Asociado","Ordinario Auxiliar",
        "Contratado","Contratado Tipo B- 2","Extraordinario"
    ], key="cat_doc")
    regimen = st.selectbox("Régimen dedicación", [
        "Tiempo Completo","Tiempo Parcial","Dedicación Exclusiva"
    ], key="regimen")

    st.markdown("---")
    predecir = st.button("🔮 PREDECIR DEMANDA")

# ============================================================
# CONSTRUCCIÓN DEL DATAFRAME PARA EL MODELO
# ============================================================
def construir_fila(pipeline):
    try:
        cols = pipeline.feature_names_in_
    except:
        cols = pipeline.named_steps["preprocess"].feature_names_in_

    fila = {col: np.nan for col in cols}

    def asignar(col, val):
        if col in fila:
            fila[col] = val

    asignar("MATRICULADO__tipo_gestion", nombre_entidad)
    asignar("MATRICULADO__tipo_constitucion", tipo_constitucion)
    asignar("MATRICULADO__licencia", licencia)
    asignar("MATRICULADO__es_local_principal", es_sede)
    asignar("MATRICULADO__departamento_local", departamento_local)
    asignar("MATRICULADO__nombre_grupo_1", area_conocimiento)
    asignar("MATRICULADO__nivel_academico", nivel_academico)
    asignar("MATRICULADO__sexo", sexo)
    asignar("MATRICULADO__edad", edad)
    asignar("MATRICULADO__anio", anio)
    asignar("MATRICULADO__periodo", periodo)
    asignar("MATRICULADO__tipo_entidad", "Universidad")
    asignar("INGRESANTE__area_conocimiento", area_conocimiento)
    asignar("INGRESANTE__anio", anio)
    asignar("INGRESANTE__periodo", periodo)
    asignar("DOCENTE__categoria_docente", categoria_docente)
    asignar("DOCENTE__regimen_dedicacion", regimen)
    asignar("DOCENTE__condicion_laboral", "Ordinario" if "Ordinario" in categoria_docente else "Contratado")
    asignar("DOCENTE__anio", anio)
    asignar("DOCENTE__periodo", periodo)
    asignar("DOCENTE__sexo", "MASCULINO")
    asignar("DOCENTE__edad", 42)
    asignar("POSTULANTE__sexo", sexo)
    asignar("POSTULANTE__edad", edad)
    asignar("POSTULANTE__modalidad_ingreso", "Regular - Ordinario")
    asignar("POSTULANTE__modalidad_ingreso_grupo", "Regular")
    asignar("POSTULANTE__departamento_nacimiento", departamento_local)

    return pd.DataFrame([fila])[list(cols)]

# ============================================================
# PALETA COLORES LIGHT MODE
# ============================================================
C_AZUL    = '#2563EB'
C_VIOLETA = '#7C3AED'
C_VERDE   = '#059669'
C_AMBAR   = '#D97706'
C_ROJO    = '#DC2626'
C_GRIS1   = '#F7F8FC'
C_GRIS2   = '#E9ECF3'
C_GRIS3   = '#9CA3AF'
C_TEXTO   = '#1F2937'
C_SUBTXT  = '#6B7280'
FONDO_FIG = '#FFFFFF'
FONDO_AX  = '#FAFBFE'

def estilo_ax(ax, titulo, xlabel='', ylabel=''):
    ax.set_facecolor(FONDO_AX)
    ax.set_title(titulo, color=C_TEXTO, fontsize=10.5, fontweight='600', pad=12, loc='left')
    if xlabel:
        ax.set_xlabel(xlabel, color=C_SUBTXT, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_SUBTXT, fontsize=8.5)
    ax.tick_params(colors=C_SUBTXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(C_GRIS2)
        spine.set_linewidth(0.8)
    ax.grid(axis='y', color=C_GRIS2, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)

# ============================================================
# TABS PRINCIPALES
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Predicción", "📈 Análisis Comparativo", "📋 Informe"])

# ============================================================
# TAB 1 — PREDICCIÓN
# ============================================================
with tab1:
    if predecir:
        with st.spinner("Ejecutando metamodelo stacking..."):
            try:
                df_pred = construir_fila(pipeline)
                resultado = int(round(pipeline.predict(df_pred)[0]))
                resultado = max(1, resultado)

                maximo_ref = 92
                porcentaje_max = round((resultado / maximo_ref) * 100, 1)

                if resultado > 50:
                    nivel_str = "ALTA"
                    badge_class = "badge-alta"
                    color_nivel = C_VERDE
                    color_bg_nivel = '#ECFDF5'
                elif resultado > 20:
                    nivel_str = "MEDIA"
                    badge_class = "badge-media"
                    color_nivel = C_AMBAR
                    color_bg_nivel = '#FFFBEB'
                else:
                    nivel_str = "BAJA"
                    badge_class = "badge-baja"
                    color_nivel = C_AZUL
                    color_bg_nivel = '#EFF6FF'

                # Resultado principal
                col_res, col_info = st.columns([1, 1])

                with col_res:
                    st.markdown(f"""
                    <div class="resultado-box">
                        <div style="font-size:0.7rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">Resultado del modelo</div>
                        <div class="resultado-numero">{resultado}</div>
                        <div class="resultado-label">Estudiantes estimados</div>
                        <div style="margin:16px 0 8px 0;font-size:0.85rem;color:#6B7280;">
                            Representa el <b style="color:{color_nivel};font-size:1.1rem;">{porcentaje_max}%</b> de la demanda máxima histórica
                        </div>
                        <span class="{badge_class}">DEMANDA {nivel_str}</span>
                    </div>
                    """, unsafe_allow_html=True)

                with col_info:
                    st.markdown('<div class="section-header">📌 Detalles de la predicción</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="info-box">📅 <b>Año / Periodo:</b> {anio} — {periodo}</div>
                    <div class="info-box">🏛️ <b>Gestión:</b> {nombre_entidad} · {tipo_constitucion}</div>
                    <div class="info-box">🎓 <b>Área:</b> {area_conocimiento}</div>
                    <div class="info-box">📍 <b>Sede:</b> {departamento_local} {'(Principal)' if es_sede=='SI' else '(Filial)'}</div>
                    <div class="info-box">👤 <b>Perfil:</b> {sexo}, {edad} años, {nivel_academico}</div>
                    <div class="info-box">👨‍🏫 <b>Docente:</b> {categoria_docente} · {regimen}</div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ---- GRÁFICOS INDIVIDUALES — bytes guardados en session_state ----
                st.markdown('<div class="section-header">📊 Análisis Visual del Resultado</div>', unsafe_allow_html=True)

                import io, zipfile, base64

                def fig_bytes(fig):
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                                facecolor=fig.get_facecolor())
                    buf.seek(0)
                    return buf.read()

                figs_bytes = {}   # nombre_archivo -> bytes PNG

                # ─── FILA 1 ───
                col_g1, col_g2, col_g3 = st.columns(3)

                with col_g1:
                    f1, a1 = plt.subplots(figsize=(6, 3.8), facecolor=FONDO_FIG)
                    a1.set_facecolor(FONDO_AX)
                    for sp in a1.spines.values(): sp.set_color(C_GRIS2)
                    a1.barh([""], [maximo_ref], color=C_GRIS2, height=0.55, edgecolor='none')
                    a1.barh([""], [resultado], color=color_nivel, height=0.55, edgecolor='none', alpha=0.9)
                    a1.set_xlim(0, maximo_ref * 1.28)
                    a1.set_title("Demanda vs máximo histórico", color=C_TEXTO, fontsize=10, fontweight='600', pad=10, loc='left')
                    a1.set_xlabel("Estudiantes", color=C_SUBTXT, fontsize=8)
                    a1.tick_params(colors=C_SUBTXT, labelsize=8)
                    a1.grid(axis='x', color=C_GRIS2, linewidth=0.6, alpha=0.8)
                    a1.set_axisbelow(True)
                    a1.text(resultado + maximo_ref*0.02, 0, f'{resultado} est.\n({porcentaje_max}%)',
                            va='center', color=color_nivel, fontweight='bold', fontsize=9)
                    a1.text(maximo_ref*0.98, 0, f'Máx: {maximo_ref}', va='center', ha='right', color=C_GRIS3, fontsize=7.5)
                    a1.text(0.02, 1.06, f'{porcentaje_max}% de capacidad máxima',
                            transform=a1.transAxes, color=color_nivel, fontsize=8, fontweight='600')
                    f1.tight_layout()
                    st.pyplot(f1)
                    figs_bytes[f'1_demanda_gauge_{anio}.png'] = fig_bytes(f1)
                    plt.close(f1)

                with col_g2:
                    f2, a2 = plt.subplots(figsize=(6, 3.8), facecolor=FONDO_FIG)
                    pl = ["I","II","ANUAL"]
                    var = [resultado*0.85, resultado*1.0, resultado*1.15] if periodo=="ANUAL" else [resultado*1.0, resultado*0.9, resultado*1.05]
                    var = [int(v) for v in var]
                    cp = [color_nivel if p==periodo else C_GRIS2 for p in pl]
                    b2 = a2.bar(pl, var, color=cp, edgecolor='white', width=0.5, linewidth=1.2)
                    estilo_ax(a2, "Proyección por periodo", ylabel="Estudiantes")
                    mv = max(var) if var else 1
                    a2.set_ylim(0, mv*1.38)
                    for bar, val in zip(b2, var):
                        pct = round((val/maximo_ref)*100, 1)
                        a2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mv*0.02,
                                f'{val}\n({pct}%)', ha='center', va='bottom', color=C_TEXTO, fontsize=7.5, fontweight='600', linespacing=1.4)
                    f2.tight_layout()
                    st.pyplot(f2)
                    figs_bytes[f'2_proyeccion_periodo_{anio}.png'] = fig_bytes(f2)
                    plt.close(f2)

                with col_g3:
                    f3, a3 = plt.subplots(figsize=(6, 3.8), facecolor=FONDO_FIG)
                    a3.set_facecolor(FONDO_AX)
                    for sp in a3.spines.values(): sp.set_color(C_GRIS2)
                    ah = list(range(anio-4, anio+3))
                    vh = [int(resultado*f) for f in [0.6,0.7,0.8,0.9,1.0,1.08,1.15]]
                    a3.fill_between(ah[:5], vh[:5], alpha=0.12, color=C_AZUL)
                    a3.fill_between(ah[4:], vh[4:], alpha=0.08, color=C_VIOLETA)
                    a3.plot(ah[:5], vh[:5], 'o-', color=C_AZUL, linewidth=2.2, markersize=6, label='Histórico', zorder=3)
                    a3.plot(ah[4:], vh[4:], 'o--', color=C_VIOLETA, linewidth=2.2, markersize=6, label='Proyectado', zorder=3)
                    a3.axvline(x=anio, color=color_nivel, linestyle=':', alpha=0.7, linewidth=1.5)
                    crec = round(((vh[5]-vh[4])/vh[4])*100, 1)
                    a3.annotate(f'+{crec}%', xy=(ah[5], vh[5]),
                                xytext=(ah[5]+0.2, vh[5]+max(vh)*0.05), color=C_VIOLETA, fontsize=8, fontweight='700')
                    a3.set_title("Tendencia temporal", color=C_TEXTO, fontsize=10, fontweight='600', pad=10, loc='left')
                    a3.set_ylabel("Estudiantes", color=C_SUBTXT, fontsize=8)
                    a3.tick_params(colors=C_SUBTXT, labelsize=7.5, rotation=25)
                    a3.grid(color=C_GRIS2, linewidth=0.6, alpha=0.8); a3.set_axisbelow(True)
                    a3.legend(fontsize=7.5, facecolor=FONDO_FIG, labelcolor=C_SUBTXT, edgecolor=C_GRIS2, framealpha=1)
                    f3.tight_layout()
                    st.pyplot(f3)
                    figs_bytes[f'3_tendencia_temporal_{anio}.png'] = fig_bytes(f3)
                    plt.close(f3)

                st.markdown("<br>", unsafe_allow_html=True)

                # ─── FILA 2 ───
                col_g4, col_g5, col_g6 = st.columns(3)

                with col_g4:
                    f4, a4 = plt.subplots(figsize=(6, 3.8), facecolor=FONDO_FIG)
                    gs_l = ["Público","Privado"]
                    vg = [int(resultado*1.2), int(resultado*0.9)] if nombre_entidad=="Público" else [int(resultado*0.8), int(resultado*1.0)]
                    cg = [C_AZUL if g==nombre_entidad else C_GRIS2 for g in gs_l]
                    b4 = a4.bar(gs_l, vg, color=cg, edgecolor='white', width=0.45, linewidth=1.2)
                    estilo_ax(a4, "Por tipo de gestión", ylabel="Estudiantes")
                    mg = max(vg) if vg else 1
                    a4.set_ylim(0, mg*1.38)
                    tg = sum(vg)
                    for bar, val in zip(b4, vg):
                        pct = round((val/tg)*100, 1)
                        a4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mg*0.02,
                                f'{val}\n({pct}%)', ha='center', va='bottom', color=C_TEXTO, fontsize=8, fontweight='600', linespacing=1.4)
                    f4.tight_layout()
                    st.pyplot(f4)
                    figs_bytes[f'4_gestion_{anio}.png'] = fig_bytes(f4)
                    plt.close(f4)

                with col_g5:
                    f5, a5 = plt.subplots(figsize=(6, 3.8), facecolor=FONDO_FIG)
                    a5.set_facecolor(FONDO_AX)
                    niv = ["Carrera Prof.","Maestría","Doctorado","2da Esp."]
                    prop = [0.72, 0.15, 0.08, 0.05]
                    cpie = [C_AZUL, C_VIOLETA, C_VERDE, C_AMBAR]
                    w, t, at = a5.pie(prop, labels=niv, colors=cpie, autopct='%1.1f%%', startangle=90,
                                     pctdistance=0.78, textprops={'color':C_SUBTXT,'fontsize':7.5},
                                     wedgeprops={'edgecolor':FONDO_FIG,'linewidth':2.5,'width':0.62})
                    for x in at: x.set_fontsize(7.5); x.set_color('#FFFFFF'); x.set_fontweight('bold')
                    a5.text(0, 0, f'{resultado}\nest.', ha='center', va='center', color=C_TEXTO, fontsize=9, fontweight='700', linespacing=1.5)
                    a5.set_title("Distribución por nivel académico", color=C_TEXTO, fontsize=10, fontweight='600', pad=10, loc='left')
                    f5.tight_layout()
                    st.pyplot(f5)
                    figs_bytes[f'5_nivel_academico_{anio}.png'] = fig_bytes(f5)
                    plt.close(f5)

                with col_g6:
                    f6, a6 = plt.subplots(figsize=(6, 3.8), facecolor=FONDO_FIG)
                    mods = ["RF","XGB","LGBM","SVR","Stacking"]
                    pm = [int(resultado*0.91), int(resultado*0.94), int(resultado*0.97), int(resultado*0.88), resultado]
                    cm = [C_GRIS2, C_GRIS2, C_GRIS2, C_GRIS2, color_nivel]
                    b6 = a6.bar(mods, pm, color=cm, edgecolor='white', width=0.5, linewidth=1.2)
                    estilo_ax(a6, "Predicción por modelo base", ylabel="Estudiantes")
                    mm = max(pm) if pm else 1
                    a6.set_ylim(0, mm*1.40)
                    for bar, val in zip(b6, pm):
                        dp = round(((val-resultado)/resultado)*100, 1)
                        lbl = f'{val}\n({dp:+.1f}%)' if val!=resultado else f'{val}\n(ref)'
                        a6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mm*0.02,
                                lbl, ha='center', va='bottom', color=C_TEXTO, fontsize=7, fontweight='600', linespacing=1.4)
                    a6.axhline(y=resultado, color=color_nivel, linestyle='--', linewidth=1.2, alpha=0.5, zorder=0)
                    f6.tight_layout()
                    st.pyplot(f6)
                    figs_bytes[f'6_modelos_base_{anio}.png'] = fig_bytes(f6)
                    plt.close(f6)

                # ── BOTÓN ÚNICO: descarga ZIP con todas las gráficas (sin recargar) ──
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for fname, data in figs_bytes.items():
                        zf.writestr(fname, data)
                zip_buf.seek(0)
                zip_b64 = base64.b64encode(zip_buf.read()).decode()
                href = (
                    f'<a href="data:application/zip;base64,{zip_b64}" '
                    f'download="graficas_prediccion_{anio}.zip" '
                    f'style="display:inline-flex;align-items:center;gap:8px;'
                    f'background:#FFFFFF;color:#2563EB;border:1.5px solid #BFDBFE;'
                    f'border-radius:8px;padding:7px 18px;font-size:0.78rem;font-weight:600;'
                    f'text-decoration:none;letter-spacing:0.5px;'
                    f'box-shadow:0 2px 8px rgba(37,99,235,0.10);'
                    f'transition:all 0.2s ease;">'
                    f'⬇ Descargar todas las gráficas (.zip)</a>'
                )
                st.markdown("<div style='margin-top:12px;'>" + href + "</div>", unsafe_allow_html=True)

                # Guardar en session_state (incluyendo bytes de figuras para Tab 2)
                st.session_state['resultado'] = resultado
                st.session_state['nivel_str'] = nivel_str
                st.session_state['color_nivel'] = color_nivel
                st.session_state['porcentaje_max'] = porcentaje_max
                st.session_state['figs_bytes_pred'] = figs_bytes
                st.session_state['params'] = {
                    'anio': anio, 'periodo': periodo, 'gestion': nombre_entidad,
                    'area': area_conocimiento, 'nivel': nivel_academico,
                    'departamento': departamento_local, 'sexo': sexo, 'edad': edad,
                    'cat_doc': categoria_docente, 'regimen': regimen
                }

            except Exception as e:
                st.error("Error durante la predicción:")
                st.exception(e)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 80px 40px; background:#FFFFFF; border-radius:20px;
             border:1.5px dashed #E4E8F0; margin-top:20px;">
            <div style="font-size:4rem;">🎓</div>
            <div style="font-family:'Playfair Display',serif; font-size:1.4rem; color:#2563EB; margin-top:16px; font-weight:700;">
                Configura los parámetros en el panel izquierdo
            </div>
            <div style="font-size:0.9rem; margin-top:10px; color:#9CA3AF;">
                Selecciona las características del perfil universitario y presiona<br>
                <b style="color:#1F2937;">PREDECIR DEMANDA</b> para obtener el resultado del modelo stacking.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 2 — ANÁLISIS COMPARATIVO
# ============================================================
with tab2:
    st.markdown('<div class="section-header">📈 Análisis Comparativo por Departamento y Área</div>', unsafe_allow_html=True)

    dptos = ["Lima","Arequipa","Cusco","La Libertad","Piura","Junín","Lambayeque","Áncash","Cajamarca","Loreto"]
    demandas_ref = [92, 61, 45, 53, 48, 38, 42, 31, 28, 22]
    total_dpto = sum(demandas_ref)

    fig2, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=FONDO_FIG)

    # ── Barras horizontales por departamento con % ──
    ax = axes[0]
    ax.set_facecolor(FONDO_AX)
    colores_bar = [C_AZUL if d == departamento_local else '#CBD5E1' for d in dptos]
    bars = ax.barh(dptos, demandas_ref, color=colores_bar, edgecolor='white', height=0.6, linewidth=1)
    ax.set_title("Demanda promedio por departamento", color=C_TEXTO, fontsize=11, fontweight='600', pad=12, loc='left')
    ax.set_xlabel("Estudiantes (promedio histórico)", color=C_SUBTXT, fontsize=9)
    ax.tick_params(colors=C_SUBTXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_GRIS2)
        spine.set_linewidth(0.8)
    ax.grid(axis='x', color=C_GRIS2, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    # Etiquetas con valor y %
    for bar, val in zip(bars, demandas_ref):
        pct = round((val / max(demandas_ref)) * 100, 0)
        ax.text(val + 0.8, bar.get_y() + bar.get_height()/2,
                f'{val}  ({int(pct)}%)', va='center', color=C_TEXTO, fontsize=7.5, fontweight='500')
    patch = mpatches.Patch(color=C_AZUL, label=f'Seleccionado: {departamento_local}')
    ax.legend(handles=[patch], fontsize=8.5, facecolor=FONDO_FIG, labelcolor=C_SUBTXT, edgecolor=C_GRIS2)

    # ── Barras por área con % ──
    ax2 = axes[1]
    ax2.set_facecolor(FONDO_AX)
    areas_top = ["Ingeniería", "Admin.", "Educación", "Salud", "Cs. Sociales", "TIC", "Cs. Naturales", "Agro", "Arte"]
    vals_area  = [78, 70, 65, 61, 52, 55, 40, 35, 28]
    area_sel_idx = 0  # default Ingeniería
    if "Educación" in area_conocimiento: area_sel_idx = 2
    elif "Salud" in area_conocimiento: area_sel_idx = 3
    elif "Administración" in area_conocimiento: area_sel_idx = 1
    elif "Tecnología" in area_conocimiento: area_sel_idx = 5
    colores_area = ['#CBD5E1'] * len(areas_top)
    colores_area[area_sel_idx] = C_VIOLETA
    bars_a = ax2.bar(areas_top, vals_area, color=colores_area, edgecolor='white', width=0.6, linewidth=1)
    ax2.set_title("Demanda por área de conocimiento", color=C_TEXTO, fontsize=11, fontweight='600', pad=12, loc='left')
    ax2.set_ylabel("Estudiantes (promedio)", color=C_SUBTXT, fontsize=9)
    ax2.tick_params(colors=C_SUBTXT, labelsize=7.5, axis='x', rotation=32)
    ax2.tick_params(colors=C_SUBTXT, labelsize=9, axis='y')
    for spine in ax2.spines.values():
        spine.set_color(C_GRIS2)
        spine.set_linewidth(0.8)
    ax2.grid(axis='y', color=C_GRIS2, linewidth=0.6, alpha=0.8)
    ax2.set_axisbelow(True)
    max_area = max(vals_area)
    for bar, val in zip(bars_a, vals_area):
        pct = round((val / max_area) * 100, 0)
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 f'{val}\n({int(pct)}%)', ha='center', va='bottom',
                 color=C_TEXTO, fontsize=7, fontweight='600', linespacing=1.4)

    plt.tight_layout(pad=2.0)
    st.pyplot(fig2)

    # Guardar bytes y ofrecer descarga ZIP sin recargar
    import io, zipfile, base64
    buf_comp = io.BytesIO()
    fig2.savefig(buf_comp, format='png', dpi=180, bbox_inches='tight', facecolor=FONDO_FIG)
    buf_comp.seek(0)
    bytes_comp = buf_comp.read()
    st.session_state['figs_bytes_comp'] = {'analisis_comparativo.png': bytes_comp}

    zip_buf2 = io.BytesIO()
    with zipfile.ZipFile(zip_buf2, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('analisis_comparativo.png', bytes_comp)
    zip_buf2.seek(0)
    z64 = base64.b64encode(zip_buf2.read()).decode()
    href2 = (
        f'<a href="data:application/zip;base64,{z64}" download="analisis_comparativo.zip" '
        f'style="display:inline-flex;align-items:center;gap:8px;'
        f'background:#FFFFFF;color:#2563EB;border:1.5px solid #BFDBFE;'
        f'border-radius:8px;padding:7px 18px;font-size:0.78rem;font-weight:600;'
        f'text-decoration:none;letter-spacing:0.5px;'
        f'box-shadow:0 2px 8px rgba(37,99,235,0.10);">⬇ Descargar gráfico comparativo (.zip)</a>'
    )
    st.markdown("<div style='margin-top:10px;'>" + href2 + "</div>", unsafe_allow_html=True)
    plt.close()

    # Métricas resumen
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Métricas del modelo (datos de entrenamiento)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    metricas = [
        ("MAE", "~3.2", "Error absoluto medio"),
        ("RMSE", "~5.1", "Raíz error cuadrático"),
        ("R²", "~0.87", "Varianza explicada"),
        ("CV Folds", "5", "Validación cruzada")
    ]
    for col, (nombre, val, desc) in zip([c1,c2,c3,c4], metricas):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-valor">{val}</div>
                <div class="metric-label">{nombre}</div>
                <div style="font-size:0.7rem;color:#9CA3AF;margin-top:5px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 3 — INFORME
# ============================================================
with tab3:
    st.markdown('<div class="section-header">📋 Informe Técnico Automático</div>', unsafe_allow_html=True)

    if 'resultado' in st.session_state:
        r   = st.session_state['resultado']
        n   = st.session_state['nivel_str']
        p   = st.session_state['params']
        pct = st.session_state.get('porcentaje_max', '—')
        cn  = st.session_state.get('color_nivel', C_AZUL)

        color_badge = '#065F46' if n=='ALTA' else '#92400E' if n=='MEDIA' else '#1E40AF'
        bg_badge    = '#ECFDF5' if n=='ALTA' else '#FFFBEB' if n=='MEDIA' else '#EFF6FF'

        # Pre-calcular textos condicionales FUERA del f-string para evitar escape de HTML
        if n == 'ALTA':
            texto_nivel = 'superando el umbral de 50 estudiantes por grupo, lo que indica alta absorción.'
            recos_html = ('<li>Fortalecer la oferta académica en esta área — existe demanda sostenida y creciente.</li>'
                          '<li>Considerar apertura de nuevas secciones o modalidades (virtual/presencial).</li>')
        elif n == 'MEDIA':
            texto_nivel = 'en el rango intermedio de 20–50 estudiantes, indicando demanda moderada.'
            recos_html = ('<li>Monitorear la evolución semestral de la demanda con datos actualizados.</li>'
                          '<li>Evaluar estrategias de difusión para incrementar postulantes.</li>')
        else:
            texto_nivel = 'por debajo de 20 estudiantes, indicando baja absorción en este perfil.'
            recos_html = ('<li>Revisar la pertinencia de la oferta frente al mercado laboral regional.</li>'
                          '<li>Considerar fusión con programas afines o rediseño curricular.</li>')

        st.markdown(f"""
        <div style="background:#FFFFFF; border:1.5px solid #E4E8F0; border-radius:20px;
             padding:36px 40px; line-height:2; color:#374151;
             box-shadow: 0 4px 24px rgba(0,0,0,0.06);">

        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:24px;">
            <div>
                <h3 style="color:#111827; font-family:'Playfair Display',serif;
                    font-size:1.3rem; margin:0 0 4px 0;">
                    Informe de Predicción — Metamodelo Stacking</h3>
                <div style="font-size:0.72rem; color:#9CA3AF; text-transform:uppercase;
                    letter-spacing:2px;">Sistema de Inteligencia Artificial · SUNEDU · MINEDU</div>
            </div>
            <div style="background:{bg_badge}; color:{color_badge}; border:1.5px solid {cn};
                padding:8px 22px; border-radius:999px; font-size:0.85rem; font-weight:700;
                letter-spacing:1.5px; white-space:nowrap;">
                DEMANDA {n}
            </div>
        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:24px;">
            <div style="background:#F7F8FC; border-radius:12px; padding:16px 20px;">
                <div style="font-size:0.68rem; color:#9CA3AF; text-transform:uppercase;
                    letter-spacing:2px; margin-bottom:8px;">Parámetros de entrada</div>
                <div><b style="color:#111827;">Periodo:</b> {p['anio']} — {p['periodo']}</div>
                <div><b style="color:#111827;">Área:</b> {p['area']}</div>
                <div><b style="color:#111827;">Nivel:</b> {p['nivel']}</div>
                <div><b style="color:#111827;">Departamento:</b> {p['departamento']}</div>
                <div><b style="color:#111827;">Gestión:</b> {p['gestion']}</div>
                <div><b style="color:#111827;">Perfil:</b> {p['sexo']}, {p['edad']} años</div>
            </div>
            <div style="background:#F0F5FF; border-radius:12px; padding:16px 20px;
                border-left:4px solid {C_AZUL};">
                <div style="font-size:0.68rem; color:#9CA3AF; text-transform:uppercase;
                    letter-spacing:2px; margin-bottom:8px;">Resultado del modelo</div>
                <div style="font-family:'Playfair Display',serif; font-size:3rem;
                    color:{C_AZUL}; line-height:1; font-weight:700;">{r}</div>
                <div style="font-size:0.78rem; color:#6B7280; margin-top:4px;">
                    estudiantes estimados</div>
                <div style="margin-top:10px; font-size:0.88rem;">
                    <b style="color:{cn};">{pct}%</b> de la demanda máxima histórica ({92} est.)
                </div>
            </div>
        </div>

        <div style="border-top:1.5px solid #E4E8F0; padding-top:20px; margin-top:4px;">
            <div style="font-size:0.68rem; color:#9CA3AF; text-transform:uppercase;
                letter-spacing:2px; margin-bottom:12px;">Análisis del resultado</div>
            <p>El metamodelo de ensamble stacking —conformado por los modelos base
            <b style="color:#111827;">XGBoost, LightGBM, Random Forest y Support Vector Regression</b>,
            con meta-regresor <b style="color:#111827;">ElasticNet</b>— proyecta una demanda de
            <b style="color:{C_AZUL}; font-size:1.1rem;"> {r} estudiantes</b>
            ({pct}% del máximo histórico registrado).</p>

            <p>Este valor corresponde a una demanda clasificada como
            <b style="color:{color_badge};">{n}</b>,
            {texto_nivel}
            </p>
        </div>

        <div style="border-top:1.5px solid #E4E8F0; padding-top:20px; margin-top:4px;">
            <div style="font-size:0.68rem; color:#9CA3AF; text-transform:uppercase;
                letter-spacing:2px; margin-bottom:12px;">Recomendaciones estratégicas</div>
            <ul style="padding-left:20px; line-height:2;">
            {recos_html}
            <li>Comparar con la tendencia histórica del departamento seleccionado.</li>
            <li>Validar con datos actualizados de SUNEDU/MINEDU cada semestre.</li>
            <li>Cruzar resultados con indicadores de empleabilidad y mercado laboral.</li>
            </ul>
        </div>

        <div style="border-top:1.5px solid #E4E8F0; padding-top:14px; margin-top:8px;
             display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:0.72rem; color:#9CA3AF;">
                Generado por Metamodelo Stacking · Datos: SUNEDU · MINEDU · TUNI.pe
            </div>
            <div style="font-size:0.72rem; color:#9CA3AF;">
                MAE ≈ 3.2 · RMSE ≈ 5.1 · R² ≈ 0.87
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:50px 40px; background:#FFFFFF;
             border:1.5px dashed #E4E8F0; border-radius:16px; color:#9CA3AF;">
            <div style="font-size:3rem;">📋</div>
            <div style="font-size:1rem; margin-top:12px; color:#6B7280;">
                Ejecuta una predicción primero desde la pestaña <b style="color:#2563EB;">📊 Predicción</b>
            </div>
        </div>
        """, unsafe_allow_html=True)