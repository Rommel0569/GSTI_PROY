"""
Frontend Streamlit (Componente G / Capa 6 de Presentación).

Ejecutar con:  streamlit run app/main_dashboard.py

Funciona de forma autónoma (sin requerir que app/api.py esté corriendo):
importa directamente los módulos de src/ para inferencia y persistencia,
tal como exige el setup de un solo comando (Sección 4 de la directriz).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import database, decision_engine, notification_service, train  # noqa: E402

st.set_page_config(
    page_title="Sistema Predictivo de No-Show Médico",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# Sistema de diseño: paleta validada (dataviz), tipografía de sistema,
# componentes de estado. Sin emojis, sin degradados, sin bloques saturados.
# ==========================================================================

FONT_STACK = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif'

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
BORDER = "rgba(11,11,11,0.10)"
BRAND = "#2a78d6"

# Paleta de estado (fija, nunca se reasigna a otra cosa): riesgo bajo/medio/alto
STATUS_COLORS = {"BAJO": "#0ca30c", "MEDIO": "#fab219", "ALTO": "#d03b3b"}
STATUS_TINTS = {"BAJO": "#e8f6e8", "MEDIO": "#fef3d9", "ALTO": "#fbe4e4"}

# Paleta divergente (SHAP): azul = reduce el riesgo, rojo = lo aumenta
DIVERGING_ASISTENCIA = "#2a78d6"
DIVERGING_INASISTENCIA = "#e34948"

# Paleta categórica (canal de contacto, 3 categorías, slots 1-3)
CHANNEL_COLORS = {"WHATSAPP": "#2a78d6", "SMS_ONLY": "#eb6834", "NO_PHONE": "#1baf7a"}

CHANNEL_LABELS = {"WHATSAPP": "WhatsApp", "SMS_ONLY": "SMS", "NO_PHONE": "Sin contacto telefónico"}
ACTION_LABELS = {
    "RECORDATORIO_AUTOMATICO": "Recordatorio automático (SMS / WhatsApp)",
    "LLAMADA_DIRIGIDA": "Llamada telefónica dirigida",
    "VISITA_DOMICILIARIA": "Visita domiciliaria",
    "SIN_INTERVENCION": "Sin intervención (no costo-eficiente)",
}
CANAL_USO_LABELS = {
    "WHATSAPP": "WhatsApp", "SMS_ONLY": "SMS", "LLAMADA_TELEFONICA": "Llamada telefónica",
    "VISITA_DOMICILIARIA": "Visita domiciliaria", "NINGUNO": "Ninguno",
}

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{ font-family: {FONT_STACK}; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    .block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1180px; }}

    .inst-topbar {{
        height: 4px; background: {BRAND}; margin: -1.2rem -1rem 1.4rem -1rem;
    }}
    .inst-header {{
        display: flex; justify-content: space-between; align-items: flex-end;
        padding-bottom: 0.7rem; margin-bottom: 1.3rem; border-bottom: 1px solid {BORDER};
    }}
    .inst-header .inst-title {{ font-size: 1.35rem; font-weight: 700; color: {INK_PRIMARY};
        margin: 0; letter-spacing: -0.01em; }}
    .inst-header .inst-subtitle {{ font-size: 0.82rem; color: {INK_SECONDARY}; margin: 2px 0 0 0; }}
    .inst-header .inst-tag {{ font-size: 0.72rem; color: {INK_MUTED}; text-align: right;
        text-transform: uppercase; letter-spacing: 0.04em; }}

    .section-header {{ border-left: 3px solid {BRAND}; padding-left: 10px; margin: 0.4rem 0 0.9rem 0; }}
    .section-header .sh-title {{ margin: 0; font-size: 1.0rem; font-weight: 600; color: {INK_PRIMARY}; }}
    .section-header .sh-subtitle {{ margin: 2px 0 0 0; font-size: 0.8rem; color: {INK_MUTED}; }}

    .status-chip {{
        display: inline-flex; align-items: center; gap: 7px; padding: 4px 11px;
        border-radius: 4px; border: 1px solid {GRIDLINE}; background: {SURFACE};
        font-size: 0.78rem; font-weight: 600; color: {INK_PRIMARY};
        text-transform: uppercase; letter-spacing: 0.03em; line-height: 1.6;
    }}
    .status-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}

    .field-label {{ font-size: 0.72rem; color: {INK_MUTED}; text-transform: uppercase;
        letter-spacing: 0.04em; margin-bottom: 2px; }}
    .field-value {{ font-size: 0.92rem; color: {INK_PRIMARY}; margin-bottom: 10px; }}

    .stButton>button {{ border-radius: 4px; font-weight: 600; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}

    .empty-state {{
        border: 1px dashed {GRIDLINE}; border-radius: 6px; padding: 2.4rem 1rem;
        text-align: center; color: {INK_MUTED}; font-size: 0.88rem; background: {SURFACE};
    }}
    </style>
    <div class="inst-topbar"></div>
    """,
    unsafe_allow_html=True,
)


def dot_label(color: str, text: str) -> str:
    return (f'<span class="status-chip"><span class="status-dot" '
            f'style="background:{color};"></span>{text}</span>')


def status_chip(risk_level: str) -> str:
    color = STATUS_COLORS.get(risk_level, INK_MUTED)
    return dot_label(color, f"Riesgo {risk_level.lower()}")


def section_header(title: str, subtitle: str = "") -> None:
    sub_html = f'<p class="sh-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="section-header"><p class="sh-title">{title}</p>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def apply_chart_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        plot_bgcolor=SURFACE,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_STACK, "color": INK_PRIMARY, "size": 12},
        margin=dict(l=10, r=40, t=10, b=30),
        showlegend=False,
    )
    return fig


def _model_ready() -> bool:
    return train.artifacts_available()


with database.connect() as _conn:
    database.init_db(_conn)

# ==========================================================================
# Encabezado institucional
# ==========================================================================
model_ok = _model_ready()
st.markdown(
    f"""
    <div class="inst-header">
        <div>
            <p class="inst-title">Sistema Predictivo de Inasistencia a Citas Médicas</p>
            <p class="inst-subtitle">Motor de segmentación por canal y costo · reasignación
            automática de cupos · consulta externa</p>
        </div>
        <div class="inst-tag">UNSA — Ingeniería de Sistemas<br/>Gestión de Sistemas y TI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# Barra lateral: estado del sistema
# ==========================================================================
with st.sidebar:
    st.markdown("**Estado del sistema**")
    st.markdown(
        dot_label(STATUS_COLORS["BAJO"] if model_ok else STATUS_COLORS["ALTO"],
                  "Modelo entrenado" if model_ok else "Modelo no entrenado"),
        unsafe_allow_html=True,
    )
    st.markdown(
        dot_label(STATUS_COLORS["MEDIO"] if notification_service.MOCK_NOTIFICATIONS else STATUS_COLORS["BAJO"],
                  "Notificaciones: simulación" if notification_service.MOCK_NOTIFICATIONS
                  else "Notificaciones: producción (Twilio)"),
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Resumen del histórico**")
    with database.connect() as conn:
        total_hist = database.count_table(conn, "appointments", "is_demo = 0")
        no_show_hist = database.count_table(conn, "appointments", "is_demo = 0 AND no_show_actual = 1")
    tasa_hist = (no_show_hist / total_hist) if total_hist else 0.0
    st.metric("Citas históricas", f"{total_hist:,}")
    st.metric("Tasa de inasistencia observada", f"{tasa_hist:.1%}")

    st.divider()
    st.markdown("**Niveles de riesgo**")
    for lvl in ("BAJO", "MEDIO", "ALTO"):
        st.markdown(status_chip(lvl), unsafe_allow_html=True)

    st.divider()
    st.caption(
        "Proyecto de investigación aplicada — Escuela Profesional de Ingeniería de "
        "Sistemas, Universidad Nacional de San Agustín de Arequipa."
    )

if not model_ok:
    st.warning(
        "El modelo aún no está entrenado. Ejecute en la terminal: `python src/train.py` "
        "(y antes, `python src/data_pipeline.py` si es la primera vez) y recargue esta página."
    )

tab1, tab2, tab3 = st.tabs([
    "Admisión y evaluación individual",
    "Consola de gestión y auditoría",
    "Panel gerencial de indicadores",
])


# ==========================================================================
# PESTAÑA 1 — Admisión & Evaluación Individual (RF-09, RNF-03)
# ==========================================================================
with tab1:
    col_form, col_result = st.columns([1, 1.25], gap="large")

    with col_form:
        section_header("Datos del paciente y la cita", "Complete el formulario para estimar el riesgo")
        with st.container(border=True):
            with st.form("form_evaluacion"):
                c1, c2 = st.columns(2)
                with c1:
                    age = st.number_input("Edad", min_value=0, max_value=110, value=45)
                    gender = st.selectbox("Sexo", ["F", "M"])
                    scholarship = st.checkbox("Beneficiario de programa social")
                    sms_received = st.checkbox("Ya recibió un SMS previo", value=True)
                with c2:
                    hypertension = st.checkbox("Hipertensión")
                    diabetes = st.checkbox("Diabetes")
                    alcoholism = st.checkbox("Alcoholismo")
                    handicap = st.number_input("Grado de discapacidad (0-4)", 0, 4, 0)

                st.markdown('<p class="field-label">Fechas de la cita</p>', unsafe_allow_html=True)
                c3, c4 = st.columns(2)
                with c3:
                    scheduled_date = st.date_input("Fecha de registro", value=datetime.now().date())
                with c4:
                    appointment_date = st.date_input(
                        "Fecha de la cita", value=datetime.now().date() + timedelta(days=5)
                    )

                contact_channel_available = st.selectbox(
                    "Canal de contacto disponible",
                    ["WHATSAPP", "SMS_ONLY", "NO_PHONE"],
                    format_func=lambda c: CHANNEL_LABELS[c],
                )
                registrar_en_bd = st.checkbox("Registrar esta cita en el sistema", value=True)

                submitted = st.form_submit_button(
                    "Evaluar riesgo y recomendar intervención", type="primary", width="stretch"
                )

    result = None
    recommendation = None
    appointment_id = None

    if submitted:
        if not model_ok:
            st.error("No se puede evaluar: el modelo no está entrenado todavía.")
        else:
            lead_time = max((appointment_date - scheduled_date).days, 0)
            day_of_week = appointment_date.strftime("%A")

            patient_features = {
                "age": int(age), "gender": gender, "hypertension": int(hypertension),
                "diabetes": int(diabetes), "alcoholism": int(alcoholism),
                "scholarship": int(scholarship), "handicap": int(handicap),
                "sms_received": int(sms_received), "lead_time": lead_time,
                "appointment_day_of_week": day_of_week,
                "contact_channel_available": contact_channel_available,
            }
            result = train.predict_case(patient_features)
            has_chronic = bool(hypertension or diabetes)
            recommendation = decision_engine.recommend_intervention(
                result["risk_level"], contact_channel_available, has_chronic
            )

            if registrar_en_bd:
                with database.connect() as conn:
                    pid = database.insert_patient(
                        conn, external_patient_id=f"UI-{datetime.now().timestamp():.0f}",
                        age=int(age), gender=gender, hypertension=int(hypertension),
                        diabetes=int(diabetes), scholarship=int(scholarship),
                        contact_channel=contact_channel_available,
                    )
                    appointment_id = database.insert_appointment(
                        conn, patient_id=pid,
                        specialty=database.SPECIALTIES[hash(gender) % len(database.SPECIALTIES)],
                        scheduled_date=datetime.combine(scheduled_date, datetime.min.time()).isoformat(),
                        appointment_date=datetime.combine(appointment_date, datetime.min.time()).isoformat(),
                        lead_time=lead_time, status="AGENDADA", is_demo=1,
                    )
                    database.insert_risk_prediction(
                        conn, appointment_id=appointment_id, probability=result["probability"],
                        risk_level=result["risk_level"], top_features=result["top_features"],
                    )

    with col_result:
        section_header("Resultado de la evaluación")
        if result is None:
            st.markdown(
                '<div class="empty-state">Complete el formulario y presione '
                '"Evaluar riesgo y recomendar intervención" para ver el resultado aquí.</div>',
                unsafe_allow_html=True,
            )
        else:
            risk_color = STATUS_COLORS[result["risk_level"]]
            with st.container(border=True):
                gc1, gc2 = st.columns([1, 1])
                with gc1:
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["probability"] * 100,
                        number={"suffix": "%", "font": {"size": 30, "color": INK_PRIMARY}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#c3c2b7",
                                     "tickfont": {"color": INK_MUTED, "size": 10}},
                            "bar": {"color": risk_color, "thickness": 0.32},
                            "bgcolor": SURFACE,
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 35], "color": STATUS_TINTS["BAJO"]},
                                {"range": [35, 65], "color": STATUS_TINTS["MEDIO"]},
                                {"range": [65, 100], "color": STATUS_TINTS["ALTO"]},
                            ],
                        },
                    ))
                    gauge.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=0),
                                         paper_bgcolor="rgba(0,0,0,0)", font={"family": FONT_STACK})
                    st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})
                with gc2:
                    st.markdown("<br/>", unsafe_allow_html=True)
                    st.markdown(status_chip(result["risk_level"]), unsafe_allow_html=True)
                    st.markdown(
                        f'<p class="field-label" style="margin-top:14px;">Modelo utilizado</p>'
                        f'<p class="field-value">{result["model_name"]}</p>',
                        unsafe_allow_html=True,
                    )
                    if appointment_id:
                        st.markdown(
                            f'<p class="field-label">Registro</p>'
                            f'<p class="field-value">Cita #{appointment_id} guardada en hospital.db</p>',
                            unsafe_allow_html=True,
                        )

            with st.container(border=True):
                st.markdown('<p class="field-label">Variables con mayor influencia (SHAP)</p>',
                            unsafe_allow_html=True)
                shap_df = pd.DataFrame(result["top_features"])
                colors = [DIVERGING_INASISTENCIA if v == "inasistencia" else DIVERGING_ASISTENCIA
                          for v in shap_df["empuja_hacia"]]
                fig_shap = go.Figure(go.Bar(
                    x=shap_df["shap_value"], y=shap_df["feature"], orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.3f}" for v in shap_df["shap_value"]],
                    textposition="outside", textfont={"color": INK_SECONDARY, "size": 11},
                ))
                fig_shap.update_layout(
                    xaxis=dict(zeroline=True, zerolinecolor="#c3c2b7", gridcolor=GRIDLINE,
                               title=None, tickfont={"color": INK_MUTED}),
                    yaxis=dict(autorange="reversed", tickfont={"color": INK_PRIMARY}),
                )
                apply_chart_layout(fig_shap, height=190)
                st.plotly_chart(fig_shap, width="stretch", config={"displayModeBar": False})
                st.caption("Rojo: empuja hacia la inasistencia · Azul: empuja hacia la asistencia")

            with st.container(border=True):
                st.markdown('<p class="field-label">Intervención recomendada (menor costo suficiente)</p>',
                            unsafe_allow_html=True)
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Acción", ACTION_LABELS.get(recommendation["accion_recomendada"], "—"))
                rc2.metric("Canal", CANAL_USO_LABELS.get(recommendation["canal_a_usar"], "—"))
                rc3.metric("Costo estimado", f"S/ {recommendation['costo_pen']:.2f}")
                ventanas_txt = " · ".join(
                    f"{w['etiqueta'].replace('_', ' ')} ({w['horas_antes_min']}-{w['horas_antes_max']} h antes)"
                    for w in recommendation["ventanas_contacto"]
                )
                st.markdown(f'<p class="field-label" style="margin-top:8px;">Ventana de contacto</p>'
                            f'<p class="field-value">{ventanas_txt}</p>', unsafe_allow_html=True)
                if recommendation["es_prioridad_equidad"]:
                    st.info(
                        "Mecanismo de equidad activado: paciente con condición crónica y sin celular "
                        "registrado — se prioriza la visita domiciliaria."
                    )


# ==========================================================================
# PESTAÑA 2 — Consola de Gestión de Citas & Auditoría
# ==========================================================================
with tab2:
    section_header("Citas en ventana operativa", "Próximas 48 horas, ordenadas por fecha de cita")

    top_c1, top_c2 = st.columns([1, 5])
    with top_c1:
        refresh = st.button("Actualizar", width="stretch")
    if refresh:
        st.rerun()

    with database.connect() as conn:
        window_rows = database.get_appointments_in_window(conn, hours_min=0, hours_max=48)

    overview = []
    for row in window_rows:
        with database.connect() as conn:
            pred = decision_engine.get_or_compute_prediction(conn, row)
        appt_dt = datetime.fromisoformat(row["appointment_date"])
        horas_restantes = (appt_dt - datetime.now()).total_seconds() / 3600.0
        overview.append({
            "ID": row["id"],
            "Especialidad": row["specialty"],
            "Fecha de cita": appt_dt.strftime("%d/%m %H:%M"),
            "Horas restantes": round(horas_restantes, 1),
            "Estado": row["status"],
            "Riesgo": pred["risk_level"],
            "Probabilidad (%)": round((pred["probability"] or 0.0) * 100, 1),
            "Canal": CHANNEL_LABELS.get(row["contact_channel"], row["contact_channel"]),
        })

    if not overview:
        st.markdown(
            '<div class="empty-state">No hay citas dentro de la ventana de 48 horas en este momento.</div>',
            unsafe_allow_html=True,
        )
    else:
        df_overview = pd.DataFrame(overview)
        st.dataframe(
            df_overview,
            hide_index=True,
            width="stretch",
            column_config={
                "Probabilidad (%)": st.column_config.ProgressColumn(
                    "Probabilidad (%)", format="%.0f%%", min_value=0, max_value=100
                ),
                "Horas restantes": st.column_config.NumberColumn("Horas restantes", format="%.1f h"),
            },
        )

        st.divider()
        section_header("Gestionar una cita", "Seleccione un registro para contactar al paciente o registrar su respuesta")

        options = {f"#{r['ID']} · {r['Especialidad']} · {r['Fecha de cita']} · Riesgo {r['Riesgo'].lower()}": r["ID"]
                   for r in overview}
        selected_label = st.selectbox("Cita", list(options.keys()), label_visibility="collapsed")
        selected_id = options[selected_label]
        selected_row = next(r for r in window_rows if r["id"] == selected_id)
        selected_overview = next(r for r in overview if r["ID"] == selected_id)

        with st.container(border=True):
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.markdown(f'<p class="field-label">Estado</p><p class="field-value">{selected_overview["Estado"]}</p>',
                         unsafe_allow_html=True)
            dc2.markdown(status_chip(selected_overview["Riesgo"]), unsafe_allow_html=True)
            dc3.markdown(f'<p class="field-label">Canal</p><p class="field-value">{selected_overview["Canal"]}</p>',
                         unsafe_allow_html=True)
            dc4.markdown(f'<p class="field-label">Horas restantes</p>'
                         f'<p class="field-value">{selected_overview["Horas restantes"]} h</p>',
                         unsafe_allow_html=True)

            st.markdown('<p class="field-label" style="margin-top:6px;">Acciones de contacto</p>',
                        unsafe_allow_html=True)
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                if st.button("Disparar recordatorio", key=f"rec_{selected_id}", width="stretch"):
                    rec = decision_engine.recommend_intervention(
                        selected_overview["Riesgo"], selected_row["contact_channel"]
                    )
                    if rec["canal_a_usar"] == "WHATSAPP":
                        notification_service.send_whatsapp(selected_row["external_patient_id"],
                                                             "Recordatorio de su cita médica.")
                    else:
                        notification_service.send_sms(selected_row["external_patient_id"],
                                                        "Recordatorio de su cita médica.")
                    with database.connect() as conn:
                        decision_engine.register_contact_outcome(
                            conn, selected_id, rec["canal_a_usar"], "SIN_RESPUESTA"
                        )
                    st.rerun()
            with ac2:
                if st.button("Llamar (IVR / simulación)", key=f"call_{selected_id}", width="stretch"):
                    notification_service.make_call(selected_row["external_patient_id"])
                    with database.connect() as conn:
                        decision_engine.register_contact_outcome(
                            conn, selected_id, "LLAMADA_TELEFONICA", "SIN_RESPUESTA"
                        )
                    st.rerun()
            with ac3:
                visita_habilitada = (selected_row["contact_channel"] == "NO_PHONE"
                                      and selected_overview["Riesgo"] == "ALTO")
                if st.button("Alertar visita domiciliaria", key=f"visit_{selected_id}",
                             width="stretch", disabled=not visita_habilitada):
                    notification_service.log_visita_domiciliaria(selected_row["external_patient_id"])
                    with database.connect() as conn:
                        decision_engine.register_contact_outcome(
                            conn, selected_id, "VISITA_DOMICILIARIA", "SIN_RESPUESTA"
                        )
                    st.rerun()
                if not visita_habilitada:
                    st.caption("Solo disponible para riesgo alto sin celular registrado")

            st.markdown('<p class="field-label" style="margin-top:10px;">Registrar respuesta del paciente</p>',
                        unsafe_allow_html=True)
            oc1, oc2 = st.columns([2, 1])
            with oc1:
                outcome_choice = st.selectbox(
                    "Resultado", ["CONFIRMÓ", "CANCELÓ", "REPROGRAMÓ", "SIN_RESPUESTA"],
                    key=f"outcome_{selected_id}", label_visibility="collapsed",
                )
            with oc2:
                if st.button("Registrar respuesta", key=f"regbtn_{selected_id}",
                             type="primary", width="stretch"):
                    with database.connect() as conn:
                        res = decision_engine.register_contact_outcome(
                            conn, selected_id, "CONTACTO_MANUAL", outcome_choice
                        )
                    msg = f"Estado actualizado a {res['nuevo_estado']}."
                    if res["reasignacion"] and res["reasignacion"].get("reasignado"):
                        msg += f" Cupo reasignado al paciente #{res['reasignacion']['patient_id']}."
                    st.success(msg)
                    st.rerun()

    st.divider()
    section_header("Liberación y reasignación automática", "RF-07 / RF-08 — citas de riesgo medio/alto sin confirmar dentro de la ventana")
    if st.button("Ejecutar liberación y reasignación", type="primary"):
        with database.connect() as conn:
            resultados = decision_engine.evaluate_and_release_expired(conn)
        if resultados:
            st.success(f"Se procesaron {len(resultados)} cita(s) sin confirmación.")
            res_df = pd.DataFrame(resultados).rename(columns={
                "appointment_id": "Cita liberada", "specialty": "Especialidad",
                "reasignado": "Reasignada", "waitlist_id": "ID lista de espera",
                "patient_id": "Paciente asignado", "motivo": "Motivo",
            })
            st.dataframe(res_df, hide_index=True, width="stretch")
        else:
            st.markdown(
                '<div class="empty-state">No había citas de riesgo medio/alto vencidas sin confirmar.</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    col_wl, col_re = st.columns(2)
    with col_wl:
        section_header("Lista de espera")
        with database.connect() as conn:
            wl = database.fetch_all(
                conn,
                """SELECT w.id AS "ID", w.specialty AS "Especialidad", w.queue_date AS "Ingreso",
                          w.status AS "Estado", p.external_patient_id AS "Paciente"
                   FROM waitlist w JOIN patients p ON p.id = w.patient_id
                   ORDER BY w.queue_date""",
            )
        st.dataframe(pd.DataFrame([dict(r) for r in wl]), hide_index=True,
                     width="stretch", height=260)

    with col_re:
        section_header("Cupos reasignados")
        with database.connect() as conn:
            re_rows = database.fetch_all(
                conn,
                """SELECT r.id AS "ID", r.freed_appointment_id AS "Cita liberada",
                          r.waitlist_id AS "ID lista de espera", r.reassigned_at AS "Fecha"
                   FROM reassignments r ORDER BY r.reassigned_at DESC""",
            )
        st.dataframe(pd.DataFrame([dict(r) for r in re_rows]), hide_index=True,
                     width="stretch", height=260)


# ==========================================================================
# PESTAÑA 3 — Panel Gerencial de Indicadores de Impacto (RF-10, Sec. 3.6)
# ==========================================================================
with tab3:
    with database.connect() as conn:
        kpis = decision_engine.compute_kpis(conn)

    section_header("Indicadores clave")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tasa de inasistencia observada", f"{kpis['tasa_inasistencia_observada']:.1%}")
    c2.metric("Horas médicas recuperadas", f"{kpis['horas_medicas_recuperadas']:.1f} h")
    c3.metric("Reutilización de cupos", f"{kpis['tasa_reutilizacion_cupos']:.1%}")
    c4.metric("Balance económico", f"S/ {kpis['balance_economico_pen']:,.2f}")

    st.divider()
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        section_header("Inasistencia observada vs. estimada", "Banda de referencia de la literatura: 40%-66% de reducción")
        with st.container(border=True):
            banda = kpis["tasa_estimada_con_intervencion_banda"]
            labels = ["Observada", "Estimada (-40%)", "Estimada (-66%)"]
            values = [kpis["tasa_inasistencia_observada"], banda["conservadora_40pct"], banda["optimista_66pct"]]
            colors = ["#184f95", "#3987e5", "#86b6ef"]
            fig1 = go.Figure(go.Bar(
                x=values, y=labels, orientation="h", marker_color=colors,
                text=[f"{v:.1%}" for v in values], textposition="outside",
                textfont={"color": INK_SECONDARY, "size": 11},
            ))
            fig1.update_layout(
                xaxis=dict(tickformat=".0%", gridcolor=GRIDLINE, range=[0, max(values) * 1.35],
                           tickfont={"color": INK_MUTED}),
                yaxis=dict(autorange="reversed", tickfont={"color": INK_PRIMARY}),
            )
            apply_chart_layout(fig1, height=210)
            st.plotly_chart(fig1, width="stretch", config={"displayModeBar": False})

    with col_b:
        section_header("Balance económico", "Costo de canales utilizados frente al valor de horas recuperadas")
        with st.container(border=True):
            fig2 = go.Figure(go.Bar(
                x=["Costo total de canales", "Valor de horas recuperadas"],
                y=[kpis["costo_total_canales_pen"], kpis["valor_horas_recuperadas_pen"]],
                marker_color=["#eb6834", "#1baf7a"],
                text=[f"S/ {v:,.2f}" for v in
                      [kpis["costo_total_canales_pen"], kpis["valor_horas_recuperadas_pen"]]],
                textposition="outside", textfont={"color": INK_SECONDARY, "size": 11},
            ))
            fig2.update_layout(
                yaxis=dict(gridcolor=GRIDLINE, tickfont={"color": INK_MUTED}),
                xaxis=dict(tickfont={"color": INK_PRIMARY}),
            )
            apply_chart_layout(fig2, height=210)
            fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

    st.divider()
    col_c, col_d = st.columns(2, gap="large")

    with col_c:
        section_header("Contactabilidad por canal", "Proporción de pacientes según canal registrado")
        with st.container(border=True):
            canal_items = list(kpis["tasa_contactabilidad_por_canal"].items())
            canal_df = pd.DataFrame(canal_items, columns=["canal", "proporcion"])
            fig3 = go.Figure(go.Bar(
                x=canal_df["proporcion"], y=[CHANNEL_LABELS[c] for c in canal_df["canal"]],
                orientation="h", marker_color=[CHANNEL_COLORS[c] for c in canal_df["canal"]],
                text=[f"{v:.1%}" for v in canal_df["proporcion"]], textposition="outside",
                textfont={"color": INK_SECONDARY, "size": 11},
            ))
            fig3.update_layout(
                xaxis=dict(tickformat=".0%", range=[0, 1], gridcolor=GRIDLINE, tickfont={"color": INK_MUTED}),
                yaxis=dict(autorange="reversed", tickfont={"color": INK_PRIMARY}),
            )
            apply_chart_layout(fig3, height=190)
            st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})

    with col_d:
        section_header("Tasa de respuesta al contacto", "Proporción de intentos con resultado registrado")
        with st.container(border=True):
            st.markdown(
                f'<p style="font-size:2.0rem;font-weight:700;color:{INK_PRIMARY};margin:6px 0 2px 0;">'
                f'{kpis["tasa_respuesta_contacto"]:.1%}</p>',
                unsafe_allow_html=True,
            )
            st.progress(min(kpis["tasa_respuesta_contacto"], 1.0))
            st.caption(f"Total de intentos de contacto registrados hasta la fecha")

    with st.expander("Ver indicadores completos (equivalente a GET /api/v1/metrics/kpis)"):
        st.json(kpis)
