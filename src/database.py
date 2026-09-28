"""
Capa de persistencia SQLite y esquemas de auditoría (Componente D).

Entidades: patients, appointments, risk_predictions, contact_audit,
waitlist, reassignments. Sin dependencias externas (usa sqlite3 de la
librería estándar) para maximizar portabilidad (RNF-05).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

DB_PATH = Path("data/processed/hospital.db")

SPECIALTIES = ["Medicina General", "Cardiología", "Pediatría", "Ginecología", "Traumatología"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_patient_id TEXT,
    age INTEGER,
    gender TEXT,
    hypertension INTEGER DEFAULT 0,
    diabetes INTEGER DEFAULT 0,
    scholarship INTEGER DEFAULT 0,
    contact_channel TEXT DEFAULT 'NO_PHONE'
);

CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    specialty TEXT DEFAULT 'Medicina General',
    scheduled_date TEXT,
    appointment_date TEXT,
    lead_time INTEGER,
    status TEXT DEFAULT 'AGENDADA',
    no_show_actual INTEGER,
    is_demo INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS risk_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    appointment_id INTEGER NOT NULL REFERENCES appointments(id),
    probability REAL,
    risk_level TEXT,
    top_features TEXT,
    calculated_at TEXT
);

CREATE TABLE IF NOT EXISTS contact_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    appointment_id INTEGER NOT NULL REFERENCES appointments(id),
    channel_used TEXT,
    attempt_timestamp TEXT,
    outcome TEXT
);

CREATE TABLE IF NOT EXISTS waitlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    specialty TEXT,
    queue_date TEXT,
    status TEXT DEFAULT 'ESPERANDO'
);

CREATE TABLE IF NOT EXISTS reassignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    freed_appointment_id INTEGER NOT NULL REFERENCES appointments(id),
    waitlist_id INTEGER NOT NULL REFERENCES waitlist(id),
    reassigned_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_appt_date ON appointments(appointment_date);
CREATE INDEX IF NOT EXISTS idx_appt_status ON appointments(status);
CREATE INDEX IF NOT EXISTS idx_waitlist_status ON waitlist(specialty, status);
"""


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def connect(db_path: Path | str = DB_PATH):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Inserts
# --------------------------------------------------------------------------

def insert_patient(conn: sqlite3.Connection, *, external_patient_id: str, age: int,
                    gender: str, hypertension: int, diabetes: int, scholarship: int,
                    contact_channel: str) -> int:
    cur = conn.execute(
        """INSERT INTO patients
           (external_patient_id, age, gender, hypertension, diabetes, scholarship, contact_channel)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (external_patient_id, age, gender, hypertension, diabetes, scholarship, contact_channel),
    )
    return cur.lastrowid


def insert_appointment(conn: sqlite3.Connection, *, patient_id: int, specialty: str,
                        scheduled_date: str, appointment_date: str, lead_time: int,
                        status: str = "AGENDADA", no_show_actual: Optional[int] = None,
                        is_demo: int = 0) -> int:
    cur = conn.execute(
        """INSERT INTO appointments
           (patient_id, specialty, scheduled_date, appointment_date, lead_time, status,
            no_show_actual, is_demo)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (patient_id, specialty, scheduled_date, appointment_date, lead_time, status,
         no_show_actual, is_demo),
    )
    return cur.lastrowid


def bulk_insert_patients_and_appointments(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    """Inserción masiva y eficiente del histórico (~110k citas)."""
    patient_rows = []
    appt_rows = []
    for r in rows:
        patient_rows.append((
            r["external_patient_id"], r["age"], r["gender"], r["hypertension"],
            r["diabetes"], r["scholarship"], r["contact_channel"],
        ))
    conn.executemany(
        """INSERT INTO patients
           (external_patient_id, age, gender, hypertension, diabetes, scholarship, contact_channel)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        patient_rows,
    )
    # Los ids se asignan de forma contigua y en el mismo orden de inserción.
    base_id = conn.execute("SELECT MAX(id) FROM patients").fetchone()[0] - len(patient_rows) + 1
    for idx, r in enumerate(rows):
        appt_rows.append((
            base_id + idx, r["specialty"], r["scheduled_date"], r["appointment_date"],
            r["lead_time"], r["status"], r["no_show_actual"], r.get("is_demo", 0),
        ))
    conn.executemany(
        """INSERT INTO appointments
           (patient_id, specialty, scheduled_date, appointment_date, lead_time, status,
            no_show_actual, is_demo)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        appt_rows,
    )


def insert_risk_prediction(conn: sqlite3.Connection, *, appointment_id: int, probability: float,
                            risk_level: str, top_features: list) -> int:
    cur = conn.execute(
        """INSERT INTO risk_predictions
           (appointment_id, probability, risk_level, top_features, calculated_at)
           VALUES (?, ?, ?, ?, ?)""",
        (appointment_id, probability, risk_level, json.dumps(top_features, ensure_ascii=False),
         now_iso()),
    )
    return cur.lastrowid


def insert_contact_audit(conn: sqlite3.Connection, *, appointment_id: int, channel_used: str,
                          outcome: str) -> int:
    cur = conn.execute(
        """INSERT INTO contact_audit (appointment_id, channel_used, attempt_timestamp, outcome)
           VALUES (?, ?, ?, ?)""",
        (appointment_id, channel_used, now_iso(), outcome),
    )
    return cur.lastrowid


def insert_waitlist_entry(conn: sqlite3.Connection, *, patient_id: int, specialty: str,
                           queue_date: Optional[str] = None) -> int:
    cur = conn.execute(
        """INSERT INTO waitlist (patient_id, specialty, queue_date, status)
           VALUES (?, ?, ?, 'ESPERANDO')""",
        (patient_id, specialty, queue_date or now_iso()),
    )
    return cur.lastrowid


def insert_reassignment(conn: sqlite3.Connection, *, freed_appointment_id: int,
                         waitlist_id: int) -> int:
    cur = conn.execute(
        """INSERT INTO reassignments (freed_appointment_id, waitlist_id, reassigned_at)
           VALUES (?, ?, ?)""",
        (freed_appointment_id, waitlist_id, now_iso()),
    )
    return cur.lastrowid


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------

def update_appointment_status(conn: sqlite3.Connection, appointment_id: int, status: str) -> None:
    conn.execute("UPDATE appointments SET status = ? WHERE id = ?", (status, appointment_id))


def get_appointment(conn: sqlite3.Connection, appointment_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        """SELECT a.*, p.age, p.gender, p.hypertension, p.diabetes, p.scholarship,
                  p.contact_channel, p.external_patient_id
           FROM appointments a JOIN patients p ON p.id = a.patient_id
           WHERE a.id = ?""",
        (appointment_id,),
    ).fetchone()


def get_appointments_in_window(conn: sqlite3.Connection, hours_min: float = 24,
                                hours_max: float = 48) -> list[sqlite3.Row]:
    """Citas cuya appointment_date cae dentro de la ventana [ahora+hours_min, ahora+hours_max]."""
    rows = conn.execute(
        """SELECT a.*, p.age, p.gender, p.hypertension, p.diabetes, p.scholarship,
                  p.contact_channel, p.external_patient_id
           FROM appointments a JOIN patients p ON p.id = a.patient_id
           WHERE a.status IN ('AGENDADA', 'CONFIRMADA')
           ORDER BY a.appointment_date"""
    ).fetchall()
    now = datetime.now()
    out = []
    for r in rows:
        try:
            appt_dt = datetime.fromisoformat(r["appointment_date"])
        except (ValueError, TypeError):
            continue
        delta_hours = (appt_dt - now).total_seconds() / 3600.0
        if hours_min <= delta_hours <= hours_max:
            out.append(r)
    return out


def get_next_waitlist_patient(conn: sqlite3.Connection, specialty: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM waitlist
           WHERE specialty = ? AND status = 'ESPERANDO'
           ORDER BY queue_date ASC LIMIT 1""",
        (specialty,),
    ).fetchone()


def mark_waitlist_assigned(conn: sqlite3.Connection, waitlist_id: int) -> None:
    conn.execute("UPDATE waitlist SET status = 'ASIGNADO' WHERE id = ?", (waitlist_id,))


def count_table(conn: sqlite3.Connection, table: str, where: str = "1=1") -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]  # nosec B608


def fetch_all(conn: sqlite3.Connection, query: str, params: tuple = ()) -> list[sqlite3.Row]:
    return conn.execute(query, params).fetchall()
