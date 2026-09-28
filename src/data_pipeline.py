"""
Pipeline de datos e ingesta robusta (Componente A).

Estrategia de conexión en 3 niveles (RNF-05, portabilidad sin configuración
previa):
  Nivel 1 - Archivo local ya presente en data/raw/.
  Nivel 2 - Kaggle API SDK, si hay credenciales configuradas.
  Nivel 3 - Descarga HTTP directa desde un espejo público, con barra de
            progreso (tqdm), como último recurso.

También incluye limpieza, ingeniería de variables (lead_time, día de la
semana, canal de contacto sintético) y la población de la base SQLite
(hospital.db) tanto con el histórico de Kaggle como con un pequeño lote de
citas "demo" con fechas relativas a hoy, para que la Consola de Gestión
(app/main_dashboard.py, pestaña 2) tenga datos dentro de la ventana de
48-24 h sin importar cuándo se ejecute el proyecto.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import database  # noqa: E402

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOCAL_FILENAMES = ["KaggleV2-May-2016.csv", "noshowappointments.csv"]
KAGGLE_DATASET = "joniarroba/noshowappointments"

# Mirror público verificado (contiene el mismo CSV de 110,527 filas). Se usa
# únicamente como último recurso si no hay archivo local ni credenciales de
# Kaggle configuradas.
MIRROR_URL = (
    "https://raw.githubusercontent.com/ksatola/Medical-Appointments-No-Shows/"
    "master/noshowappointments-kagglev2-may-2016.csv"
)

CHANNEL_DISTRIBUTION = {"WHATSAPP": 0.70, "SMS_ONLY": 0.20, "NO_PHONE": 0.10}
RANDOM_SEED = 42


# --------------------------------------------------------------------------
# Nivel 1-3: carga del dataset
# --------------------------------------------------------------------------

def _find_local_file() -> Path | None:
    for name in LOCAL_FILENAMES:
        candidate = RAW_DIR / name
        if candidate.exists():
            return candidate
    return None


def _try_kaggle_api() -> Path | None:
    has_env_creds = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    has_json_creds = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not (has_env_creds or has_json_creds):
        return None
    try:
        import kaggle  # import diferido: solo se necesita en Nivel 2

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True)
        print("[data_pipeline] Nivel 2 (Kaggle API): descarga completada.")
        return _find_local_file()
    except Exception as exc:  # pragma: no cover - depende de credenciales reales
        print(f"[data_pipeline] Nivel 2 (Kaggle API) falló: {exc}")
        return None


def _try_http_mirror() -> Path | None:
    try:
        import requests
        from tqdm import tqdm

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        dest = RAW_DIR / "noshowappointments.csv"
        print(f"[data_pipeline] Nivel 3 (HTTP fallback): descargando desde {MIRROR_URL}")
        with requests.get(MIRROR_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc="noshowappointments.csv"
            ) as bar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        return dest
    except Exception as exc:  # pragma: no cover - depende de red
        print(f"[data_pipeline] Nivel 3 (HTTP fallback) falló: {exc}")
        return None


def load_dataset() -> pd.DataFrame:
    """Estrategia de conexión en 3 niveles descrita en la directriz (Sección A.1)."""
    path = _find_local_file()
    if path:
        print(f"[data_pipeline] Nivel 1 (archivo local): {path}")
    else:
        path = _try_kaggle_api()
    if not path:
        path = _try_http_mirror()
    if not path:
        raise FileNotFoundError(
            "No se pudo obtener el dataset por ningún nivel. Descargue manualmente "
            "'Medical Appointment No Shows' desde Kaggle (joniarroba/noshowappointments) "
            f"y coloque el CSV en {RAW_DIR}/noshowappointments.csv"
        )
    df = pd.read_csv(path)
    return df


# --------------------------------------------------------------------------
# Limpieza e ingeniería de variables
# --------------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "PatientId": "patient_id_ext",
        "AppointmentID": "appointment_id_ext",
        "Gender": "gender",
        "ScheduledDay": "scheduled_day",
        "AppointmentDay": "appointment_day",
        "Age": "age",
        "Neighbourhood": "neighbourhood",
        "Scholarship": "scholarship",
        "Hipertension": "hypertension",
        "Hypertension": "hypertension",
        "Diabetes": "diabetes",
        "Alcoholism": "alcoholism",
        "Handcap": "handicap",
        "Handicap": "handicap",
        "SMS_received": "sms_received",
        "No-show": "no_show_raw",
        "No_show": "no_show_raw",
    }
    df = df.rename(columns=rename_map)
    return df


def clean_and_engineer(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Aplica limpieza, filtrado de anomalías e ingeniería de variables
    (Sección A.2-A.4 de la directriz)."""
    df = _normalize_columns(df).copy()

    # Filtrar anomalías de edad
    df = df[(df["age"] >= 0) & (df["age"] <= 110)].copy()

    # Parseo de fechas
    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"]).dt.tz_localize(None)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"]).dt.tz_localize(None)

    # lead_time: días de anticipación, corrigiendo negativos a 0
    lead_time = (df["appointment_day"].dt.date - df["scheduled_day"].dt.date)
    df["lead_time"] = lead_time.apply(lambda d: max(d.days, 0))

    # Día de la semana de la cita
    df["appointment_day_of_week"] = df["appointment_day"].dt.day_name()

    # Variable objetivo: Yes -> 1 (inasistencia), No -> 0
    df["no_show"] = (df["no_show_raw"].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Canal de contacto disponible (simulación sintética, Sección 3.5 del informe)
    rng = np.random.default_rng(seed)
    channels = list(CHANNEL_DISTRIBUTION.keys())
    probs = list(CHANNEL_DISTRIBUTION.values())
    df["contact_channel_available"] = rng.choice(channels, size=len(df), p=probs)

    # Especialidad sintética (el dataset no la incluye; necesaria para la
    # cola de lista de espera del motor de reasignación, Componente C).
    df["specialty"] = rng.choice(database.SPECIALTIES, size=len(df))

    # Tipos numéricos limpios
    for col in ["scholarship", "hypertension", "diabetes", "alcoholism", "handicap", "sms_received"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["gender"] = df["gender"].astype(str).str.strip().str.upper()

    keep_cols = [
        "patient_id_ext", "appointment_id_ext", "age", "gender", "neighbourhood",
        "scholarship", "hypertension", "diabetes", "alcoholism", "handicap",
        "sms_received", "scheduled_day", "appointment_day", "lead_time",
        "appointment_day_of_week", "contact_channel_available", "specialty", "no_show",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


FEATURE_COLUMNS_NUMERIC = [
    "age", "scholarship", "hypertension", "diabetes", "alcoholism", "handicap",
    "sms_received", "lead_time",
]
FEATURE_COLUMNS_CATEGORICAL = ["gender", "appointment_day_of_week", "contact_channel_available"]
TARGET_COLUMN = "no_show"


# --------------------------------------------------------------------------
# Población de SQLite (histórico + citas demo)
# --------------------------------------------------------------------------

def _row_to_db_dict(row: pd.Series, status: str, is_demo: int,
                     scheduled_date: datetime, appointment_date: datetime) -> dict:
    return {
        "external_patient_id": str(row.get("patient_id_ext", "")),
        "age": int(row["age"]),
        "gender": row["gender"],
        "hypertension": int(row["hypertension"]),
        "diabetes": int(row["diabetes"]),
        "scholarship": int(row["scholarship"]),
        "contact_channel": row["contact_channel_available"],
        "specialty": row["specialty"],
        "scheduled_date": scheduled_date.isoformat(),
        "appointment_date": appointment_date.isoformat(),
        "lead_time": int(row["lead_time"]),
        "status": status,
        "no_show_actual": int(row["no_show"]),
        "is_demo": is_demo,
    }


def generate_demo_rows(df: pd.DataFrame, n: int = 60, seed: int = RANDOM_SEED) -> list[dict]:
    """Genera citas 'vigentes' (fechas relativas a hoy) muestreando perfiles
    reales del dataset, para que la Consola de Gestión tenga contenido
    dentro de la ventana de 48-24h sin depender de la fecha de ejecución."""
    rng = np.random.default_rng(seed + 1)
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    now = datetime.now()
    rows = []
    for _, row in sample.iterrows():
        hours_ahead = rng.uniform(6, 96)  # reparte citas antes, dentro y después de la ventana
        appointment_date = now + timedelta(hours=hours_ahead)
        lead_days = int(rng.integers(1, 15))
        scheduled_date = appointment_date - timedelta(days=lead_days)
        row = row.copy()
        row["lead_time"] = lead_days
        rows.append(_row_to_db_dict(row, status="AGENDADA", is_demo=1,
                                     scheduled_date=scheduled_date,
                                     appointment_date=appointment_date))
    return rows


def generate_demo_waitlist(df: pd.DataFrame, n: int = 15, seed: int = RANDOM_SEED) -> list[dict]:
    rng = np.random.default_rng(seed + 2)
    sample = df.sample(n=min(n, len(df)), random_state=seed + 2).reset_index(drop=True)
    now = datetime.now()
    rows = []
    for i, row in sample.iterrows():
        queue_date = now - timedelta(days=int(rng.integers(1, 20)))
        rows.append({
            "external_patient_id": str(row.get("patient_id_ext", f"WL-{i}")),
            "age": int(row["age"]),
            "gender": row["gender"],
            "hypertension": int(row["hypertension"]),
            "diabetes": int(row["diabetes"]),
            "scholarship": int(row["scholarship"]),
            "contact_channel": row["contact_channel_available"],
            "specialty": row["specialty"],
            "queue_date": queue_date.isoformat(),
        })
    return rows


def populate_database(df: pd.DataFrame, db_path: str | Path = database.DB_PATH,
                       historical_limit: int | None = None) -> None:
    with database.connect(db_path) as conn:
        database.init_db(conn)

        hist = df if historical_limit is None else df.head(historical_limit)
        hist_rows = [
            _row_to_db_dict(
                row, status=("NO_SHOW" if row["no_show"] == 1 else "ATENDIDA"), is_demo=0,
                scheduled_date=row["scheduled_day"], appointment_date=row["appointment_day"],
            )
            for _, row in hist.iterrows()
        ]
        print(f"[data_pipeline] Insertando {len(hist_rows)} citas históricas en hospital.db ...")
        database.bulk_insert_patients_and_appointments(conn, hist_rows)

        demo_rows = generate_demo_rows(df)
        print(f"[data_pipeline] Insertando {len(demo_rows)} citas demo (ventana operativa) ...")
        database.bulk_insert_patients_and_appointments(conn, demo_rows)

        waitlist_rows = generate_demo_waitlist(df)
        print(f"[data_pipeline] Insertando {len(waitlist_rows)} pacientes en lista de espera ...")
        for w in waitlist_rows:
            pid = database.insert_patient(
                conn, external_patient_id=w["external_patient_id"], age=w["age"],
                gender=w["gender"], hypertension=w["hypertension"], diabetes=w["diabetes"],
                scholarship=w["scholarship"], contact_channel=w["contact_channel"],
            )
            database.insert_waitlist_entry(conn, patient_id=pid, specialty=w["specialty"],
                                            queue_date=w["queue_date"])


def run(save_processed_csv: bool = True) -> pd.DataFrame:
    raw_df = load_dataset()
    print(f"[data_pipeline] Dataset crudo: {len(raw_df)} filas")
    clean_df = clean_and_engineer(raw_df)
    print(f"[data_pipeline] Dataset limpio: {len(clean_df)} filas "
          f"({clean_df['no_show'].mean():.4f} tasa de inasistencia)")

    if save_processed_csv:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "processed_appointments.csv"
        clean_df.to_csv(out_path, index=False)
        print(f"[data_pipeline] Guardado: {out_path}")

    populate_database(clean_df)
    print("[data_pipeline] hospital.db poblado correctamente.")
    return clean_df


if __name__ == "__main__":
    run()
