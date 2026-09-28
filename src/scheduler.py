"""
Automatización de producción: dispara el contacto recomendado y ejecuta la
liberación/reasignación de cupos SIN intervención manual (a diferencia del
dashboard, donde un operador presiona los botones).

En el MVP académico, todo se dispara desde app/main_dashboard.py con clics.
Para un despliegue real, este proceso debe correr en segundo plano de forma
continua (como servicio systemd, contenedor Docker, o tarea de Windows) junto
al backend y al dashboard.

Ejecutar con:  python src/scheduler.py
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import database, decision_engine, notification_service, train  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scheduler")

OUTREACH_INTERVAL_MINUTES = 15
RELEASE_INTERVAL_MINUTES = 30
REMINDER_MESSAGE = "Le recordamos su cita médica. Responda para confirmar o cancelar su asistencia."


def job_send_outreach() -> None:
    """Recorre las citas en ventana operativa que aún no recibieron ningún
    contacto y dispara automáticamente el canal recomendado por el motor de
    decisión (RF-05), sin esperar a que un operador presione un botón."""
    if not train.artifacts_available():
        logger.warning("Modelo no entrenado; se omite el ciclo de contacto automático.")
        return

    with database.connect() as conn:
        window_rows = database.get_appointments_in_window(conn, hours_min=0, hours_max=144)

    enviados = 0
    for row in window_rows:
        with database.connect() as conn:
            if row["status"] != "AGENDADA":
                continue
            already_contacted = conn.execute(
                "SELECT 1 FROM contact_audit WHERE appointment_id = ? LIMIT 1", (row["id"],)
            ).fetchone()
            if already_contacted:
                continue
            pred = decision_engine.get_or_compute_prediction(conn, row)

        risk_level = pred["risk_level"]
        appt_dt = datetime.fromisoformat(row["appointment_date"])
        hours_to_appt = (appt_dt - datetime.now()).total_seconds() / 3600.0
        due = any(
            w["horas_antes_min"] <= hours_to_appt <= w["horas_antes_max"]
            for w in decision_engine.get_contact_windows(risk_level)
        )
        if not due:
            continue

        rec = decision_engine.recommend_intervention(risk_level, row["contact_channel"])
        canal = rec["canal_a_usar"]
        patient_ref = row["external_patient_id"]

        if canal in ("WHATSAPP",):
            notification_service.send_whatsapp(patient_ref, REMINDER_MESSAGE)
        elif canal == "SMS_ONLY":
            notification_service.send_sms(patient_ref, REMINDER_MESSAGE)
        elif canal == "LLAMADA_TELEFONICA":
            notification_service.make_call(patient_ref, appointment_id=row["id"])
        elif canal == "VISITA_DOMICILIARIA":
            notification_service.log_visita_domiciliaria(patient_ref)
        else:
            continue  # SIN_INTERVENCION: no es costo-eficiente, no se contacta

        with database.connect() as conn:
            decision_engine.register_contact_outcome(conn, row["id"], canal, "SIN_RESPUESTA")
        enviados += 1

    logger.info("Ciclo de contacto automático: %d intervención(es) disparada(s).", enviados)


def job_release_expired() -> None:
    """RF-07/RF-08: libera y reasigna automáticamente los cupos de riesgo
    medio/alto que vencieron su ventana de confirmación sin respuesta."""
    with database.connect() as conn:
        resultados = decision_engine.evaluate_and_release_expired(conn)
    if resultados:
        reasignados = sum(1 for r in resultados if r.get("reasignado"))
        logger.info("Liberación automática: %d cita(s) procesada(s), %d reasignada(s).",
                    len(resultados), reasignados)


def main() -> None:
    with database.connect() as conn:
        database.init_db(conn)

    scheduler = BlockingScheduler(timezone="America/Lima")
    scheduler.add_job(job_send_outreach, "interval", minutes=OUTREACH_INTERVAL_MINUTES,
                       next_run_time=datetime.now())
    scheduler.add_job(job_release_expired, "interval", minutes=RELEASE_INTERVAL_MINUTES,
                       next_run_time=datetime.now())

    logger.info(
        "Scheduler iniciado (contacto cada %d min, liberación cada %d min). MOCK_NOTIFICATIONS=%s",
        OUTREACH_INTERVAL_MINUTES, RELEASE_INTERVAL_MINUTES, notification_service.MOCK_NOTIFICATIONS,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler detenido.")


if __name__ == "__main__":
    main()
