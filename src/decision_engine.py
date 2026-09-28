"""
Motor de decisión y costo-eficiencia (Componente C).

Separa deliberadamente "qué tan probable es" (modelo ML, ver train.py) de
"qué conviene hacer al respecto, dado el costo" (reglas de negocio de este
módulo), tal como se describe en la Sección 4.2 del informe. Los umbrales y
costos son constantes configurables (RNF-06): pueden sobreescribirse sin
tocar el modelo ni el resto del código.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from src import database

# --------------------------------------------------------------------------
# RF-03: Segmentación de niveles de riesgo
# --------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "BAJO": (0.00, 0.35),
    "MEDIO": (0.35, 0.65),
    "ALTO": (0.65, 1.0001),
}

# --------------------------------------------------------------------------
# RF-05: Costos de referencia por canal (Sección 3.9 del informe)
# --------------------------------------------------------------------------
COSTS_PEN = {
    "RECORDATORIO_AUTOMATICO": 0.09,   # SMS/WhatsApp, tarifario local (S/ 0.089)
    "LLAMADA_DIRIGIDA": 1.50,          # ~£0.50 de referencia internacional
    "VISITA_DOMICILIARIA": 8.00,       # Estimación propia (personal + transporte);
                                        # declarado como limitación en Sección 3.9 del
                                        # informe: costear con datos reales en la
                                        # siguiente fase del proyecto.
    "SIN_INTERVENCION": 0.0,
}

CONTACT_WINDOW_HOURS = (24, 48)  # ventana estándar de reconfirmación
EARLY_WARNING_HOURS_HIGH_RISK = 120  # aviso adicional ~5 días antes para riesgo alto


def classify_risk(probability: float, thresholds: dict | None = None) -> str:
    thresholds = thresholds or RISK_THRESHOLDS
    for level, (lo, hi) in thresholds.items():
        if lo <= probability < hi:
            return level
    return "ALTO" if probability >= 1.0 else "BAJO"


def get_contact_windows(risk_level: str) -> list[dict]:
    """RF-05 / Sección 3.8 paso 9: ventana de contacto estándar (48-24h),
    reforzada con un aviso más temprano solo para riesgo alto."""
    windows = [{"etiqueta": "ventana_confirmacion",
                "horas_antes_min": CONTACT_WINDOW_HOURS[0],
                "horas_antes_max": CONTACT_WINDOW_HOURS[1]}]
    if risk_level == "ALTO":
        windows.insert(0, {"etiqueta": "aviso_temprano",
                            "horas_antes_min": EARLY_WARNING_HOURS_HIGH_RISK,
                            "horas_antes_max": EARLY_WARNING_HOURS_HIGH_RISK + 24})
    return windows


def recommend_intervention(risk_level: str, contact_channel_available: str,
                            has_chronic_condition: bool = False) -> dict:
    """RF-05: Matriz Riesgo vs. Canal Disponible (menor costo suficiente)."""
    risk_level = risk_level.upper()
    channel = contact_channel_available.upper()

    if channel in ("WHATSAPP", "SMS_ONLY"):
        if risk_level == "BAJO":
            accion, canal_usar, costo = "RECORDATORIO_AUTOMATICO", channel, COSTS_PEN["RECORDATORIO_AUTOMATICO"]
        else:  # MEDIO o ALTO
            accion, canal_usar, costo = "LLAMADA_DIRIGIDA", "LLAMADA_TELEFONICA", COSTS_PEN["LLAMADA_DIRIGIDA"]
    elif channel == "NO_PHONE":
        if risk_level == "ALTO":
            accion = "VISITA_DOMICILIARIA"
            canal_usar = "VISITA_DOMICILIARIA"
            costo = COSTS_PEN["VISITA_DOMICILIARIA"]
        else:
            accion, canal_usar, costo = "SIN_INTERVENCION", "NINGUNO", COSTS_PEN["SIN_INTERVENCION"]
    else:
        accion, canal_usar, costo = "SIN_INTERVENCION", "NINGUNO", COSTS_PEN["SIN_INTERVENCION"]

    return {
        "risk_level": risk_level,
        "canal_disponible": channel,
        "accion_recomendada": accion,
        "canal_a_usar": canal_usar,
        "costo_pen": costo,
        "es_prioridad_equidad": accion == "VISITA_DOMICILIARIA" and has_chronic_condition,
        "ventanas_contacto": get_contact_windows(risk_level),
    }


# --------------------------------------------------------------------------
# RF-07 / RF-08: Liberación y reasignación de cupos
# --------------------------------------------------------------------------

def should_release_slot(risk_level: str, hours_to_appointment: float, confirmed: bool) -> bool:
    """Se libera el cupo si el paciente no confirmó dentro de la ventana de
    48-24h y su riesgo es medio/alto, o si canceló explícitamente."""
    if confirmed:
        return False
    within_or_past_window = hours_to_appointment <= CONTACT_WINDOW_HOURS[1]
    return within_or_past_window and risk_level in ("MEDIO", "ALTO")


def reassign_slot(conn: sqlite3.Connection, appointment_id: int) -> Optional[dict]:
    """RF-08: busca en waitlist al paciente con mayor antigüedad de espera
    para la misma especialidad y le asigna el cupo liberado."""
    appt = database.get_appointment(conn, appointment_id)
    if appt is None:
        return None

    database.update_appointment_status(conn, appointment_id, "LIBERADA")

    candidate = database.get_next_waitlist_patient(conn, appt["specialty"])
    if candidate is None:
        return {"appointment_id": appointment_id, "specialty": appt["specialty"],
                "reasignado": False, "motivo": "Sin pacientes en lista de espera"}

    database.mark_waitlist_assigned(conn, candidate["id"])
    database.update_appointment_status(conn, appointment_id, "REASIGNADA")
    reassignment_id = database.insert_reassignment(
        conn, freed_appointment_id=appointment_id, waitlist_id=candidate["id"]
    )
    return {
        "appointment_id": appointment_id,
        "specialty": appt["specialty"],
        "reasignado": True,
        "waitlist_id": candidate["id"],
        "patient_id": candidate["patient_id"],
        "reassignment_id": reassignment_id,
    }


def register_contact_outcome(conn: sqlite3.Connection, appointment_id: int, channel_used: str,
                              outcome: str) -> dict:
    """RF-06: registra un intento de contacto y, según el resultado, actualiza
    el estado de la cita y dispara la reasignación si corresponde (RF-07/08).
    Nunca sobrescribe auditoría previa (RNF-08): siempre inserta una fila nueva."""
    appt = database.get_appointment(conn, appointment_id)
    if appt is None:
        raise ValueError(f"Cita {appointment_id} no encontrada")

    audit_id = database.insert_contact_audit(
        conn, appointment_id=appointment_id, channel_used=channel_used, outcome=outcome
    )
    status_map = {
        "CONFIRMÓ": "CONFIRMADA", "CANCELÓ": "LIBERADA",
        "REPROGRAMÓ": "LIBERADA", "SIN_RESPUESTA": appt["status"],
    }
    new_status = status_map[outcome]
    database.update_appointment_status(conn, appointment_id, new_status)

    reassignment = None
    if new_status == "LIBERADA":
        reassignment = reassign_slot(conn, appointment_id)

    return {"audit_id": audit_id, "nuevo_estado": new_status, "reasignacion": reassignment}


def get_or_compute_prediction(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    """Reutiliza la última predicción guardada para la cita, o la calcula y
    persiste si aún no existe. Compartido por el dashboard (Tab 2) y por
    evaluate_and_release_expired para no duplicar la lógica de inferencia."""
    last_pred = conn.execute(
        """SELECT probability, risk_level FROM risk_predictions WHERE appointment_id = ?
           ORDER BY calculated_at DESC LIMIT 1""",
        (row["id"],),
    ).fetchone()
    if last_pred:
        return {"probability": last_pred["probability"], "risk_level": last_pred["risk_level"]}

    from src import train as _train  # import diferido: evita ciclo con train.py

    if not _train.artifacts_available():
        return {"probability": None, "risk_level": "MEDIO"}  # supuesto conservador sin modelo

    dow = datetime.fromisoformat(row["appointment_date"]).strftime("%A")
    pred = _train.predict_case({
        "age": row["age"], "gender": row["gender"], "hypertension": row["hypertension"],
        "diabetes": row["diabetes"], "alcoholism": 0, "scholarship": row["scholarship"],
        "handicap": 0, "sms_received": 0, "lead_time": row["lead_time"],
        "appointment_day_of_week": dow, "contact_channel_available": row["contact_channel"],
    })
    database.insert_risk_prediction(
        conn, appointment_id=row["id"], probability=pred["probability"],
        risk_level=pred["risk_level"], top_features=pred["top_features"],
    )
    return {"probability": pred["probability"], "risk_level": pred["risk_level"]}


def evaluate_and_release_expired(conn: sqlite3.Connection) -> list[dict]:
    """Recorre las citas en ventana 0-48h sin confirmar y de riesgo medio/alto,
    liberando y reasignando cada una. Usado por RF-07/RF-08 vía API/dashboard."""
    rows = database.get_appointments_in_window(conn, hours_min=0, hours_max=48)
    now = datetime.now()
    results = []
    for row in rows:
        if row["status"] != "AGENDADA":
            continue
        appt_dt = datetime.fromisoformat(row["appointment_date"])
        hours_to_appt = (appt_dt - now).total_seconds() / 3600.0
        risk_level = get_or_compute_prediction(conn, row)["risk_level"]
        if should_release_slot(risk_level, hours_to_appt, confirmed=False):
            result = reassign_slot(conn, row["id"])
            if result:
                results.append(result)
    return results


# --------------------------------------------------------------------------
# Indicadores de impacto hospitalario (Sección 3.6 / GET /metrics/kpis)
# --------------------------------------------------------------------------

CONSULTATION_DURATION_HOURS = 0.5
EXPECTED_REDUCTION_RANGE = (0.40, 0.66)  # banda de referencia de la literatura


def compute_kpis(conn: sqlite3.Connection) -> dict:
    total_hist = database.count_table(conn, "appointments", "is_demo = 0")
    no_shows_hist = database.count_table(conn, "appointments", "is_demo = 0 AND no_show_actual = 1")
    tasa_inasistencia_observada = (no_shows_hist / total_hist) if total_hist else 0.0

    reduccion_min, reduccion_max = EXPECTED_REDUCTION_RANGE
    tasa_estimada_con_intervencion = tasa_inasistencia_observada * (1 - reduccion_min)
    tasa_estimada_optimista = tasa_inasistencia_observada * (1 - reduccion_max)

    cupos_liberados = database.count_table(conn, "appointments", "status IN ('LIBERADA', 'REASIGNADA')")
    cupos_reasignados = database.count_table(conn, "appointments", "status = 'REASIGNADA'")
    tasa_reutilizacion_cupos = (cupos_reasignados / cupos_liberados) if cupos_liberados else 0.0

    horas_recuperadas = cupos_reasignados * CONSULTATION_DURATION_HOURS

    audit_rows = database.fetch_all(conn, "SELECT channel_used, outcome FROM contact_audit")
    channel_cost_map = {
        "WHATSAPP": COSTS_PEN["RECORDATORIO_AUTOMATICO"],
        "SMS_ONLY": COSTS_PEN["RECORDATORIO_AUTOMATICO"],
        "LLAMADA_TELEFONICA": COSTS_PEN["LLAMADA_DIRIGIDA"],
        "VISITA_DOMICILIARIA": COSTS_PEN["VISITA_DOMICILIARIA"],
    }
    costo_total_canales = sum(channel_cost_map.get(r["channel_used"], 0.0) for r in audit_rows)
    total_intentos = len(audit_rows)
    respondidos = sum(1 for r in audit_rows if r["outcome"] != "SIN_RESPUESTA")
    tasa_respuesta_contacto = (respondidos / total_intentos) if total_intentos else 0.0

    total_pacientes_canal = database.count_table(conn, "patients")
    contactabilidad = {}
    for canal in ("WHATSAPP", "SMS_ONLY", "NO_PHONE"):
        n = conn.execute(
            "SELECT COUNT(*) FROM patients WHERE contact_channel = ?", (canal,)
        ).fetchone()[0]
        contactabilidad[canal] = (n / total_pacientes_canal) if total_pacientes_canal else 0.0

    valor_hora_medica_pen = 45.0  # estimación de costo de oportunidad por hora de consulta
    valor_horas_recuperadas_pen = horas_recuperadas * valor_hora_medica_pen

    return {
        "tasa_inasistencia_observada": round(tasa_inasistencia_observada, 4),
        "tasa_estimada_con_intervencion_banda": {
            "conservadora_40pct": round(tasa_estimada_con_intervencion, 4),
            "optimista_66pct": round(tasa_estimada_optimista, 4),
        },
        "cupos_liberados": cupos_liberados,
        "cupos_reasignados": cupos_reasignados,
        "tasa_reutilizacion_cupos": round(tasa_reutilizacion_cupos, 4),
        "horas_medicas_recuperadas": round(horas_recuperadas, 2),
        "valor_horas_recuperadas_pen": round(valor_horas_recuperadas_pen, 2),
        "costo_total_canales_pen": round(costo_total_canales, 2),
        "balance_economico_pen": round(valor_horas_recuperadas_pen - costo_total_canales, 2),
        "tasa_respuesta_contacto": round(tasa_respuesta_contacto, 4),
        "tasa_contactabilidad_por_canal": {k: round(v, 4) for k, v in contactabilidad.items()},
        "total_citas_historicas": total_hist,
    }
