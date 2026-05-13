"""
SOLEMNE 2 — Magíster en Data Science (USS)
Dashboard interactivo: Clasificador de Niveles de Obesidad
Random Forest entrenado solo con variables conductuales (sin Peso ni Altura)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador Obesidad — USS MDSS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    /* Métricas con alto contraste */
    [data-testid="stMetric"] {
        background: #1565c0;
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid #1e88e5;
    }
    [data-testid="stMetricLabel"] p {
        color: #bbdefb !important;
        font-size: 0.85em !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8em !important;
        font-weight: 700 !important;
    }
    h1 { color: #4fc3f7; }
    h2 { color: #81d4fa; }
    .pred-box {
        background: linear-gradient(135deg, #1a237e, #283593);
        border-left: 5px solid #42a5f5;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .alert-ok {
        background: #1b5e20;
        border-left: 5px solid #66bb6a;
        border-radius: 8px;
        padding: 12px;
    }
    .alert-warn {
        background: #4a148c;
        border-left: 5px solid #ce93d8;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CARGA Y PREPROCESAMIENTO DE DATOS
# ─────────────────────────────────────────────
LABEL_MAP = {
    "Insufficient_Weight": "Peso Insuficiente",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso I",
    "Overweight_Level_II": "Sobrepeso II",
    "Obesity_Type_I": "Obesidad I",
    "Obesity_Type_II": "Obesidad II",
    "Obesity_Type_III": "Obesidad III",
}
CLASS_ORDER = [
    "Peso Insuficiente", "Peso Normal", "Sobrepeso I", "Sobrepeso II",
    "Obesidad I", "Obesidad II", "Obesidad III"
]
CLASS_COLORS = {
    "Peso Insuficiente": "#42a5f5",
    "Peso Normal":       "#66bb6a",
    "Sobrepeso I":       "#ffca28",
    "Sobrepeso II":      "#ffa726",
    "Obesidad I":        "#ef5350",
    "Obesidad II":       "#c62828",
    "Obesidad III":      "#7b1fa2",
}

# Variables conductuales (sin Peso ni Altura — igual que Solemne 1)
BEHAVIORAL_COLS = [
    "Gender", "Age", "family_history_with_overweight",
    "FAVC", "FCVC", "NCP", "CAEC", "SMOKE",
    "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"
]
TARGET_COL = "NObeyesdad"

@st.cache_data
def load_and_train():
    """Carga CSV, entrena RF sin peso/altura, retorna todo lo necesario."""
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dsrscientist/"
        "dataset1/master/obesity.csv"
        if False else "ObesityDataSet_raw_and_data_sinthetic.csv"
    )

    # Traducir etiquetas
    df["clase_es"] = df[TARGET_COL].map(LABEL_MAP)

    # Codificar variables categóricas
    df_enc = df[BEHAVIORAL_COLS + [TARGET_COL]].copy()
    encoders = {}
    for col in df_enc.select_dtypes(include=["object", "str"]).columns:
        if col == TARGET_COL:
            continue
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    le_target = LabelEncoder()
    df_enc[TARGET_COL] = le_target.fit_transform(df_enc[TARGET_COL])
    encoders[TARGET_COL] = le_target

    X = df_enc[BEHAVIORAL_COLS]
    y = df_enc[TARGET_COL]

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modelo
    model = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Métricas
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # CV 5-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # Reporte por clase
    target_names_orig = le_target.classes_
    report = classification_report(
        y_test, y_pred,
        target_names=target_names_orig,
        output_dict=True
    )
    report_df = pd.DataFrame(report).T.iloc[:-3]
    report_df.index = [LABEL_MAP.get(i, i) for i in report_df.index]

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    class_names_es = [LABEL_MAP.get(c, c) for c in target_names_orig]

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=BEHAVIORAL_COLS).sort_values(ascending=False)

    return (
        df, model, encoders, le_target,
        acc, cv_scores, report_df, cm, class_names_es,
        fi, X_test, y_test, y_pred
    )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:2.2em;'>🧬</div>
        <div style='font-size:1.3em; font-weight:700; color:#4fc3f7; margin-top:6px;'>
            Solemne 2 — MDSS
        </div>
        <div style='font-size:0.85em; color:#90caf9; margin-top:4px;'>
            Magíster en Ciencia de Datos<br>Universidad San Sebastián
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "Navegar",
        ["📊 Dashboard del Modelo", "🔬 Predictor Interactivo"],
        index=0,
    )
    st.divider()
    st.caption("Dataset: Obesidad — Palechor & De la Hoz (2019)")
    st.caption("Modelo: Random Forest (sin Peso/Altura)")

# ─────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────
with st.spinner("⏳ Entrenando modelo y preparando dashboard..."):
    (
        df, model, encoders, le_target,
        acc, cv_scores, report_df, cm, class_names_es,
        fi, X_test, y_test, y_pred
    ) = load_and_train()

# ═══════════════════════════════════════════════
#  PÁGINA 1 — DASHBOARD DEL MODELO
# ═══════════════════════════════════════════════
if page == "📊 Dashboard del Modelo":

    st.title("🧬 Clasificador de Niveles de Obesidad")
    st.markdown(
        "**Random Forest entrenado con variables conductuales** — "
        "sin Peso ni Altura | Dataset: Palechor & De la Hoz (2019)"
    )

    # ── KPIs ──────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy (test)", f"{acc*100:.1f}%")
    k2.metric("CV 5-fold (media)", f"{cv_scores.mean()*100:.1f}%")
    k3.metric("CV std (±)", f"{cv_scores.std()*100:.2f}%")
    k4.metric("Clases objetivo", "7")

    st.divider()

    # ── Fila 1: Distribución + CV ──────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📈 Distribución de clases")
        conteo = df["clase_es"].value_counts().reindex(CLASS_ORDER).dropna()
        fig_dist = px.bar(
            x=conteo.index, y=conteo.values,
            labels={"x": "Clase", "y": "Registros"},
            color=conteo.index,
            color_discrete_map=CLASS_COLORS,
            template="plotly_dark",
        )
        fig_dist.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_b:
        st.subheader("📉 Accuracy por fold (CV k=5)")
        fig_cv = go.Figure()
        fig_cv.add_bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=cv_scores * 100,
            marker_color=["#42a5f5" if v == cv_scores.max() else "#78909c" for v in cv_scores],
            text=[f"{v*100:.1f}%" for v in cv_scores],
            textposition="outside",
        )
        fig_cv.add_hline(y=cv_scores.mean()*100, line_dash="dash",
                         line_color="#ffca28",
                         annotation_text=f"Media {cv_scores.mean()*100:.1f}%")
        fig_cv.update_layout(
            template="plotly_dark", yaxis_range=[0, 100],
            height=350, yaxis_title="Accuracy (%)"
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    # ── Fila 2: Matriz confusión + Feature Importance ──
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("🗺️ Matriz de Confusión (test set)")
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        fig_cm = px.imshow(
            cm_pct,
            x=class_names_es, y=class_names_es,
            color_continuous_scale="Blues",
            text_auto=".0f",
            labels={"color": "%"},
            template="plotly_dark",
        )
        fig_cm.update_layout(height=420)
        fig_cm.update_xaxes(tickangle=30)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_d:
        st.subheader("🏆 Importancia de Variables")
        fi_df = fi.reset_index()
        fi_df.columns = ["Variable", "Importancia"]
        fig_fi = px.bar(
            fi_df, x="Importancia", y="Variable",
            orientation="h",
            color="Importancia",
            color_continuous_scale="Teal",
            template="plotly_dark",
            text=fi_df["Importancia"].apply(lambda v: f"{v*100:.1f}%"),
        )
        fig_fi.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
        fig_fi.update_traces(textposition="outside")
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Fila 3: Métricas por clase ─────────────
    st.subheader("📋 Métricas por Clase — Random Forest (sin Peso/Altura)")
    cols_show = ["precision", "recall", "f1-score", "support"]
    rd = report_df[cols_show].copy()
    rd["precision"] = rd["precision"].apply(lambda v: f"{v*100:.1f}%")
    rd["recall"] = rd["recall"].apply(lambda v: f"{v*100:.1f}%")
    rd["f1-score"] = rd["f1-score"].apply(lambda v: f"{v*100:.1f}%")
    rd["support"] = rd["support"].astype(int)
    rd.columns = ["Precisión", "Recall", "F1-Score", "Soporte"]

    def color_f1(val):
        v = float(val.replace("%", "")) / 100
        if v >= 0.90:
            return "background-color: #1b5e20; color: white"
        elif v >= 0.75:
            return "background-color: #33691e; color: white"
        elif v >= 0.60:
            return "background-color: #f57f17; color: black"
        else:
            return "background-color: #b71c1c; color: white"

    st.dataframe(
        rd.style.map(color_f1, subset=["F1-Score"]),
        use_container_width=True
    )

    # ── Notas analíticas ──────────────────────
    st.divider()
    st.subheader("💡 Hallazgos clave del modelo")
    ia, ib, ic = st.columns(3)
    ia.info("**Peso Normal** es la clase más difícil (F1≈68%): sus hábitos se solapan con Sobrepeso I.")
    ib.success("**Obesidad III** es casi perfecta (F1≈99%): hábitos tan marcados que el modelo los detecta sin el peso.")
    ic.warning("Sin Peso/Altura el accuracy baja ~10%, confirmando que el peso 'robaba' toda la varianza (trampa circular).")


# ═══════════════════════════════════════════════
#  PÁGINA 2 — PREDICTOR INTERACTIVO
# ═══════════════════════════════════════════════
else:
    st.title("🔬 Predictor de Nivel de Obesidad")
    st.markdown(
        "Ingresa los datos conductuales de un individuo y el modelo clasificará "
        "su nivel de obesidad. **No se utiliza el peso ni la altura.**"
    )
    st.divider()

    # ── Formulario de entrada ──────────────────
    with st.form("prediction_form"):
        st.subheader("👤 Datos del individuo")

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            gender = st.selectbox("Género", ["Female", "Male"])
            age = st.slider("Edad (años)", 10, 80, 25)
            family_hist = st.selectbox(
                "Historial familiar sobrepeso",
                ["yes", "no"],
                help="¿Algún familiar con sobrepeso?"
            )
        with r1c2:
            favc = st.selectbox(
                "Consume alimentos hipercalóricos (FAVC)",
                ["yes", "no"],
                help="Frecuente consumo de comida con alta calorías"
            )
            fcvc = st.slider(
                "Frecuencia consumo verduras (FCVC)",
                1.0, 3.0, 2.0, 0.5,
                help="1=Nunca, 2=A veces, 3=Siempre"
            )
            ncp = st.slider(
                "N° comidas principales al día (NCP)",
                1.0, 4.0, 3.0, 0.5,
            )
        with r1c3:
            caec = st.selectbox(
                "Come entre comidas (CAEC)",
                ["no", "Sometimes", "Frequently", "Always"],
                index=1,
            )
            smoke = st.selectbox("Fuma (SMOKE)", ["no", "yes"])
            ch2o = st.slider(
                "Agua diaria (litros) (CH2O)",
                1.0, 3.0, 2.0, 0.5,
            )

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            scc = st.selectbox(
                "Monitorea calorías (SCC)",
                ["no", "yes"],
            )
            faf = st.slider(
                "Días act. física/semana (FAF)",
                0.0, 3.0, 1.0, 0.5,
                help="0=Sedentario … 3=Alta actividad"
            )
        with r2c2:
            tue = st.slider(
                "Horas dispositivos tecnológicos/día (TUE)",
                0.0, 2.0, 1.0, 0.5,
            )
            calc = st.selectbox(
                "Consumo de alcohol (CALC)",
                ["no", "Sometimes", "Frequently", "Always"],
                index=1,
            )
        with r2c3:
            mtrans = st.selectbox(
                "Transporte habitual (MTRANS)",
                [
                    "Public_Transportation", "Walking",
                    "Automobile", "Motorbike", "Bike"
                ],
                index=0,
            )

        submitted = st.form_submit_button("🚀 Clasificar individuo", use_container_width=True)

    # ── Resultado ─────────────────────────────
    if submitted:
        # Construir fila de input
        input_dict = {
            "Gender": gender,
            "Age": age,
            "family_history_with_overweight": family_hist,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans,
        }
        input_df = pd.DataFrame([input_dict])

        # Codificar igual que en entrenamiento
        for col, le in encoders.items():
            if col == TARGET_COL:
                continue
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col])
                except ValueError:
                    input_df[col] = le.transform([le.classes_[0]])[0]

        # Predicción
        pred_idx = model.predict(input_df[BEHAVIORAL_COLS])[0]
        pred_proba = model.predict_proba(input_df[BEHAVIORAL_COLS])[0]
        pred_class_orig = le_target.inverse_transform([pred_idx])[0]
        pred_class_es = LABEL_MAP.get(pred_class_orig, pred_class_orig)
        confianza = pred_proba[pred_idx] * 100

        st.divider()
        st.subheader("🎯 Resultado de la clasificación")

        rc1, rc2 = st.columns([1, 2])
        with rc1:
            color = CLASS_COLORS.get(pred_class_es, "#42a5f5")
            st.markdown(
                f"""
                <div style='
                    background: {color}22;
                    border-left: 6px solid {color};
                    border-radius: 12px;
                    padding: 24px;
                    text-align: center;
                '>
                    <h2 style='color:{color}; margin:0'>🏷️ {pred_class_es}</h2>
                    <p style='color:white; font-size:1.2em; margin:12px 0 0 0'>
                        Confianza: <b>{confianza:.1f}%</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with rc2:
            # Gráfico probabilidades por clase
            all_classes_es = [LABEL_MAP.get(c, c) for c in le_target.classes_]
            fig_prob = go.Figure(go.Bar(
                x=[p * 100 for p in pred_proba],
                y=all_classes_es,
                orientation="h",
                marker_color=[
                    CLASS_COLORS.get(c, "#78909c") for c in all_classes_es
                ],
                text=[f"{p*100:.1f}%" for p in pred_proba],
                textposition="outside",
            ))
            fig_prob.add_vline(x=50, line_dash="dot", line_color="white", opacity=0.3)
            fig_prob.update_layout(
                template="plotly_dark",
                title="Probabilidad por clase",
                xaxis_title="Probabilidad (%)",
                xaxis_range=[0, 110],
                height=320,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # Interpretación
        st.subheader("📌 Interpretación del resultado")
        interpretaciones = {
            "Peso Insuficiente": (
                "El modelo detecta patrones de **bajo consumo calórico o actividad extrema**. "
                "Revisar ingesta nutricional y frecuencia de comidas."
            ),
            "Peso Normal": (
                "Los hábitos del individuo corresponden al **rango saludable**. "
                "Mantener la frecuencia de actividad física y el consumo de verduras."
            ),
            "Sobrepeso I": (
                "Hay señales de inicio de sobrepeso: posible **aumento en comidas hipercalóricas** "
                "o reducción de actividad física. Intervención preventiva recomendada."
            ),
            "Sobrepeso II": (
                "El patrón conductual sugiere **sobrepeso consolidado**. "
                "Se recomienda revisión de hábitos alimenticios y rutina de ejercicio."
            ),
            "Obesidad I": (
                "El modelo identifica hábitos de **riesgo de obesidad moderada**: "
                "sedentarismo, alta ingesta calórica y bajo consumo de agua."
            ),
            "Obesidad II": (
                "Combinación de **inactividad + historial familiar + comida hipercalórica** "
                "caracterizan este nivel. Requiere atención médica."
            ),
            "Obesidad III": (
                "Nivel más severo: hábitos muy marcados. El modelo lo identifica con "
                "**alta confianza incluso sin usar el peso**. Atención médica urgente."
            ),
        }
        msg = interpretaciones.get(pred_class_es, "")
        if "Normal" in pred_class_es or "Insuficiente" in pred_class_es:
            st.success(f"✅ {msg}")
        elif "Sobrepeso" in pred_class_es:
            st.warning(f"⚠️ {msg}")
        else:
            st.error(f"🚨 {msg}")

        # Tabla de inputs ingresados
        with st.expander("🔎 Ver datos ingresados al modelo"):
            display_input = pd.DataFrame([{
                "Variable": k, "Valor ingresado": v
            } for k, v in input_dict.items()])
            st.dataframe(display_input, use_container_width=True)