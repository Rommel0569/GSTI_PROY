"""
Adaptadores de canal y llamada telefónica (Componente E).

Por defecto opera en Modo Simulación Académica Offline (MOCK_NOTIFICATIONS=True)
para no requerir saldo ni tarjetas de crédito (RNF-05). Si se configuran
credenciales reales de Twilio y se fija MOCK_NOTIFICATIONS=False, usa la API
real de Twilio (SMS, WhatsApp y llamada de voz con Text-to-Speech).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

MOCK_NOTIFICATIONS = os.getenv("MOCK_NOTIFICATIONS", "true").lower() != "false"

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

IVR_SCRIPT = (
    "Estimado paciente, le llamamos del Hospital para reconfirmar su cita medica "
    "programada para mañana. Marque 1 para confirmar su asistencia o 2 para liberar "
    "su cupo."
)


@dataclass
class NotificationResult:
    channel: str
    to: str
    status: str          # SENT_MOCK, SENT_REAL, FAILED
    outcome: str | None = None   # CONFIRMÓ, CANCELÓ, REPROGRAMÓ, SIN_RESPUESTA (si aplica)
    detail: str = ""


def _get_twilio_client():
    from twilio.rest import Client  # import diferido: opcional

    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
        raise RuntimeError("Credenciales de Twilio no configuradas (TWILIO_ACCOUNT_SID/TOKEN).")
    return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def send_sms(to: str, message: str) -> NotificationResult:
    if MOCK_NOTIFICATIONS:
        print(f"[MOCK SMS -> {to}] {message}")
        return NotificationResult(channel="SMS_ONLY", to=to, status="SENT_MOCK")
    try:
        client = _get_twilio_client()
        client.messages.create(body=message, from_=TWILIO_FROM_NUMBER, to=to)
        return NotificationResult(channel="SMS_ONLY", to=to, status="SENT_REAL")
    except Exception as exc:
        return NotificationResult(channel="SMS_ONLY", to=to, status="FAILED", detail=str(exc))


def send_whatsapp(to: str, message: str) -> NotificationResult:
    if MOCK_NOTIFICATIONS:
        print(f"[MOCK WHATSAPP -> {to}] {message}")
        return NotificationResult(channel="WHATSAPP", to=to, status="SENT_MOCK")
    try:
        client = _get_twilio_client()
        client.messages.create(
            body=message, from_=TWILIO_WHATSAPP_FROM, to=f"whatsapp:{to}"
        )
        return NotificationResult(channel="WHATSAPP", to=to, status="SENT_REAL")
    except Exception as exc:
        return NotificationResult(channel="WHATSAPP", to=to, status="FAILED", detail=str(exc))


def make_call(to: str, webhook_url: str | None = None) -> NotificationResult:
    """Dispara una llamada saliente con TTS. En modo simulación, la
    respuesta del paciente (tecla 1 o 2) se captura desde la UI mediante
    `simulate_ivr_response`, no aquí."""
    if MOCK_NOTIFICATIONS:
        print(f"[MOCK LLAMADA -> {to}] Guion IVR: {IVR_SCRIPT}")
        return NotificationResult(channel="LLAMADA_TELEFONICA", to=to, status="SENT_MOCK")
    try:
        client = _get_twilio_client()
        twiml = (
            f'<Response><Gather numDigits="1" action="{webhook_url}">'
            f"<Say language=\"es-MX\">{IVR_SCRIPT}</Say></Gather></Response>"
        )
        client.calls.create(twiml=twiml, from_=TWILIO_FROM_NUMBER, to=to)
        return NotificationResult(channel="LLAMADA_TELEFONICA", to=to, status="SENT_REAL")
    except Exception as exc:
        return NotificationResult(channel="LLAMADA_TELEFONICA", to=to, status="FAILED", detail=str(exc))


def simulate_ivr_response(to: str, digit_pressed: str) -> NotificationResult:
    """Simula (o, en producción, procesa el webhook DTMF real de) la
    respuesta del paciente durante la llamada IVR."""
    outcome_map = {"1": "CONFIRMÓ", "2": "CANCELÓ"}
    outcome = outcome_map.get(str(digit_pressed), "SIN_RESPUESTA")
    return NotificationResult(channel="LLAMADA_TELEFONICA", to=to, status="SENT_MOCK",
                               outcome=outcome, detail=f"Tecla pulsada: {digit_pressed}")


def process_twilio_dtmf_webhook(digits: str) -> str:
    """Traduce el parámetro 'Digits' del webhook real de Twilio al mismo
    vocabulario de resultado usado en la auditoría."""
    outcome_map = {"1": "CONFIRMÓ", "2": "CANCELÓ"}
    return outcome_map.get(str(digits), "SIN_RESPUESTA")


def log_visita_domiciliaria(to: str) -> NotificationResult:
    """La visita domiciliaria no es un canal digital: se registra como
    alerta operativa para personal de salud (RF-05)."""
    print(f"[ALERTA OPERATIVA] Visita domiciliaria requerida para paciente {to}")
    return NotificationResult(channel="VISITA_DOMICILIARIA", to=to, status="SENT_MOCK",
                               detail="Alerta generada para personal de salud")
