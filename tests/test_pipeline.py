"""Pruebas unitarias básicas (tests/test_pipeline.py).

Ejecutar con:  pytest -v
No requieren descargar el dataset ni entrenar el modelo: usan DataFrames
sintéticos pequeños y una base SQLite temporal en memoria.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import data_pipeline, database, decision_engine  # noqa: E402


# --------------------------------------------------------------------------
# data_pipeline.py
# --------------------------------------------------------------------------

def _make_raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "PatientId": [1, 2, 3, 4],
        "AppointmentID": [100, 101, 102, 103],
        "Gender": ["F", "M", "F", "M"],
        "ScheduledDay": ["2016-04-01T10:00:00Z", "2016-04-05T10:00:00Z",
                         "2016-04-10T10:00:00Z", "2016-04-10T10:00:00Z"],
        "AppointmentDay": ["2016-04-10T00:00:00Z", "2016-04-03T00:00:00Z",
                          "2016-04-10T00:00:00Z", "2016-04-10T00:00:00Z"],
        "Age": [30, -5, 45, 200],
        "Neighbourhood": ["A", "B", "C", "D"],
        "Scholarship": [0, 1, 0, 0],
        "Hipertension": [1, 0, 0, 1],
        "Diabetes": [0, 0, 1, 0],
        "Alcoholism": [0, 0, 0, 0],
        "Handcap": [0, 0, 0, 0],
        "SMS_received": [1, 0, 1, 0],
        "No-show": ["No", "Yes", "No", "Yes"],
    })


def test_age_filtering_removes_invalid_ages():
    df = data_pipeline.clean_and_engineer(_make_raw_df())
    assert (df["age"] >= 0).all()
    assert (df["age"] <= 110).all()
    # Las filas con Age=-5 y Age=200 deben haberse eliminado
    assert len(df) == 2


def test_lead_time_never_negative():
    df = data_pipeline.clean_and_engineer(_make_raw_df())
    # PatientId=2 tiene AppointmentDay anterior a ScheduledDay -> debe fijarse a 0
    assert (df["lead_time"] >= 0).all()


def test_no_show_mapping_binary():
    df = data_pipeline.clean_and_engineer(_make_raw_df())
    assert set(df["no_show"].unique()).issubset({0, 1})


def test_contact_channel_distribution_within_tolerance():
    rng_seed = 7
    n = 20000
    raw = pd.DataFrame({
        "PatientId": range(n), "AppointmentID": range(n),
        "Gender": ["F"] * n,
        "ScheduledDay": ["2016-04-01T10:00:00Z"] * n,
        "AppointmentDay": ["2016-04-05T00:00:00Z"] * n,
        "Age": [30] * n, "Neighbourhood": ["A"] * n,
        "Scholarship": [0] * n, "Hipertension": [0] * n, "Diabetes": [0] * n,
        "Alcoholism": [0] * n, "Handcap": [0] * n, "SMS_received": [0] * n,
        "No-show": ["No"] * n,
    })
    df = data_pipeline.clean_and_engineer(raw, seed=rng_seed)
    proportions = df["contact_channel_available"].value_counts(normalize=True)
    assert abs(proportions.get("WHATSAPP", 0) - 0.70) < 0.02
    assert abs(proportions.get("SMS_ONLY", 0) - 0.20) < 0.02
    assert abs(proportions.get("NO_PHONE", 0) - 0.10) < 0.02


# --------------------------------------------------------------------------
# decision_engine.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize("probability,expected", [
    (0.0, "BAJO"), (0.34, "BAJO"), (0.35, "MEDIO"), (0.64, "MEDIO"),
    (0.65, "ALTO"), (1.0, "ALTO"),
])
def test_classify_risk_boundaries(probability, expected):
    assert decision_engine.classify_risk(probability) == expected


@pytest.mark.parametrize("risk,channel,expected_accion", [
    ("BAJO", "WHATSAPP", "RECORDATORIO_AUTOMATICO"),
    ("BAJO", "SMS_ONLY", "RECORDATORIO_AUTOMATICO"),
    ("MEDIO", "WHATSAPP", "LLAMADA_DIRIGIDA"),
    ("ALTO", "SMS_ONLY", "LLAMADA_DIRIGIDA"),
    ("ALTO", "NO_PHONE", "VISITA_DOMICILIARIA"),
    ("BAJO", "NO_PHONE", "SIN_INTERVENCION"),
])
def test_recommend_intervention_matrix(risk, channel, expected_accion):
    result = decision_engine.recommend_intervention(risk, channel)
    assert result["accion_recomendada"] == expected_accion


def test_visita_domiciliaria_is_costliest_intervention():
    costs = decision_engine.COSTS_PEN
    assert costs["VISITA_DOMICILIARIA"] > costs["LLAMADA_DIRIGIDA"] > costs["RECORDATORIO_AUTOMATICO"]


# --------------------------------------------------------------------------
# database.py + reasignación de cupos (integración ligera con SQLite :memory:)
# --------------------------------------------------------------------------

@pytest.fixture()
def memory_conn():
    conn = database.get_connection(":memory:")
    database.init_db(conn)
    yield conn
    conn.close()


def test_init_db_creates_expected_tables(memory_conn):
    tables = {r["name"] for r in memory_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    expected = {"patients", "appointments", "risk_predictions", "contact_audit",
                "waitlist", "reassignments"}
    assert expected.issubset(tables)


def test_reassign_slot_moves_waitlist_patient_into_freed_slot(memory_conn):
    conn = memory_conn
    patient_a = database.insert_patient(
        conn, external_patient_id="A", age=40, gender="F", hypertension=0,
        diabetes=0, scholarship=0, contact_channel="WHATSAPP",
    )
    patient_b = database.insert_patient(
        conn, external_patient_id="B", age=50, gender="M", hypertension=1,
        diabetes=0, scholarship=0, contact_channel="NO_PHONE",
    )
    appt_id = database.insert_appointment(
        conn, patient_id=patient_a, specialty="Cardiología",
        scheduled_date=datetime.now().isoformat(),
        appointment_date=(datetime.now() + timedelta(hours=30)).isoformat(),
        lead_time=5, status="AGENDADA",
    )
    waitlist_id = database.insert_waitlist_entry(conn, patient_id=patient_b, specialty="Cardiología")

    result = decision_engine.reassign_slot(conn, appt_id)

    assert result["reasignado"] is True
    assert result["waitlist_id"] == waitlist_id
    updated_appt = database.get_appointment(conn, appt_id)
    assert updated_appt["status"] == "REASIGNADA"
    waitlist_row = conn.execute("SELECT status FROM waitlist WHERE id = ?", (waitlist_id,)).fetchone()
    assert waitlist_row["status"] == "ASIGNADO"


def test_reassign_slot_without_waitlist_candidates(memory_conn):
    conn = memory_conn
    patient_a = database.insert_patient(
        conn, external_patient_id="A", age=40, gender="F", hypertension=0,
        diabetes=0, scholarship=0, contact_channel="WHATSAPP",
    )
    appt_id = database.insert_appointment(
        conn, patient_id=patient_a, specialty="Pediatría",
        scheduled_date=datetime.now().isoformat(),
        appointment_date=(datetime.now() + timedelta(hours=30)).isoformat(),
        lead_time=5, status="AGENDADA",
    )
    result = decision_engine.reassign_slot(conn, appt_id)
    assert result["reasignado"] is False
    updated_appt = database.get_appointment(conn, appt_id)
    assert updated_appt["status"] == "LIBERADA"


def test_register_contact_outcome_confirms_appointment(memory_conn):
    conn = memory_conn
    patient_a = database.insert_patient(
        conn, external_patient_id="A", age=40, gender="F", hypertension=0,
        diabetes=0, scholarship=0, contact_channel="WHATSAPP",
    )
    appt_id = database.insert_appointment(
        conn, patient_id=patient_a, specialty="Medicina General",
        scheduled_date=datetime.now().isoformat(),
        appointment_date=(datetime.now() + timedelta(hours=30)).isoformat(),
        lead_time=5, status="AGENDADA",
    )
    result = decision_engine.register_contact_outcome(conn, appt_id, "WHATSAPP", "CONFIRMÓ")
    assert result["nuevo_estado"] == "CONFIRMADA"
    assert result["reasignacion"] is None
