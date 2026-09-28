"""
Adaptadores de canal y llamada telefónica (Componente E).

Por defecto opera en Modo Simulación Académica Offline (MOCK_NOTIFICATIONS=True)
para no requerir saldo ni tarjetas de crédito (RNF-05). Si se configuran
credenciales reales de Twilio y se fija MOCK_NOTIFICATIONS=False, usa la API
real de Twilio (SMS, WhatsApp y llamada de voz con Text-to-Speech).

Nota de producción (WhatsApp): Meta exige que todo mensaje de WhatsApp
iniciado por el negocio (fuera de una conversación de 24h ya abierta por el
paciente) use una plantilla pre-aprobada por Meta, registrada en Twilio vía
Content API (content_sid). Enviar un `body` libre solo funciona en el sandbox
de pruebas de Twilio, no en un número de WhatsApp Business real. Por eso
send_whatsapp() usa TWILIO_WHATSAPP_CONTENT_SID cuando está configurado, y
cae a mensaje libre (solo válido en sandbox) si no lo está.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv es una comodidad, no un requisito duro
    pass

logger = logging.getLogger("notification_service")

MOCK_NOTIFICATIONS = os.getenv("MOCK_NOTIFICATIONS", "true").lower() != "false"

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TWILIO_WHATSAPP_CONTENT_SID = os.getenv("TWILIO_WHATSAPP_CONTENT_SID", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

# Proveedor de WhatsApp: "twilio" (por defecto) o "meta_cloud" (API oficial de
# Meta, directa, sin margen de intermediario por mensaje). Ambos exigen
# igualmente verificación de negocio y plantilla aprobada por Meta: esa regla
# es de Meta, no del proveedor.
WHATSAPP_PROVIDER = os.getenv("WHATSAPP_PROVIDER", "twilio").lower()
META_WHATSAPP_TOKEN = os.getenv("META_WHATSAPP_TOKEN", "")
META_WHATSAPP_PHONE_NUMBER_ID = os.getenv("META_WHATSAPP_PHONE_NUMBER_ID", "")
META_WHATSAPP_TEMPLATE_NAME = os.getenv("META_WHATSAPP_TEMPLATE_NAME", "")
META_WHATSAPP_TEMPLATE_LANG = os.getenv("META_WHATSAPP_TEMPLATE_LANG", "es_MX")
META_GRAPH_API_VERSION = os.getenv("META_GRAPH_API_VERSION", "v20.0")

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
    provider_sid: str | None = None


def _get_twilio_client():
    from twilio.rest import Client  # import diferido: opcional

    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
        raise RuntimeError("Credenciales de Twilio no configuradas (TWILIO_ACCOUNT_SID/TOKEN).")
    return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def send_sms(to: str, message: str) -> NotificationResult:
    if MOCK_NOTIFICATIONS:
        logger.info("[MOCK SMS -> %s] %s", to, message)
        return NotificationResult(channel="SMS_ONLY", to=to, status="SENT_MOCK")
    if not TWILIO_FROM_NUMBER:
        logger.error("TWILIO_FROM_NUMBER no configurado; no se puede enviar SMS a %s", to)
        return NotificationResult(channel="SMS_ONLY", to=to, status="FAILED",
                                   detail="TWILIO_FROM_NUMBER no configurado")
    try:
        client = _get_twilio_client()
        msg = client.messages.create(body=message, from_=TWILIO_FROM_NUMBER, to=to)
        logger.info("SMS enviado a %s (sid=%s)", to, msg.sid)
        return NotificationResult(channel="SMS_ONLY", to=to, status="SENT_REAL", provider_sid=msg.sid)
    except Exception as exc:
        logger.exception("Fallo al enviar SMS a %s", to)
        return NotificationResult(channel="SMS_ONLY", to=to, status="FAILED", detail=str(exc))


def send_whatsapp(to: str, message: str, template_variables: dict | None = None) -> NotificationResult:
    """Envía un recordatorio por WhatsApp mediante el proveedor configurado
    en WHATSAPP_PROVIDER ("twilio" o "meta_cloud"). En producción, ambos
    exigen una plantilla de mensaje aprobada por Meta (regla de Meta, no del
    proveedor); `template_variables` llena las variables de esa plantilla."""
    if MOCK_NOTIFICATIONS:
        logger.info("[MOCK WHATSAPP -> %s] %s", to, message)
        return NotificationResult(channel="WHATSAPP", to=to, status="SENT_MOCK")
    if WHATSAPP_PROVIDER == "meta_cloud":
        return _send_whatsapp_meta_cloud(to, template_variables)
    return _send_whatsapp_twilio(to, message, template_variables)


def _send_whatsapp_twilio(to: str, message: str, template_variables: dict | None) -> NotificationResult:
    if not TWILIO_WHATSAPP_CONTENT_SID:
        logger.warning(
            "TWILIO_WHATSAPP_CONTENT_SID no configurado: el envío solo funcionará en el "
            "sandbox de pruebas de Twilio, no en un número de WhatsApp Business real."
        )
    try:
        import json

        client = _get_twilio_client()
        kwargs = {"from_": TWILIO_WHATSAPP_FROM, "to": f"whatsapp:{to}"}
        if TWILIO_WHATSAPP_CONTENT_SID:
            kwargs["content_sid"] = TWILIO_WHATSAPP_CONTENT_SID
            kwargs["content_variables"] = json.dumps(template_variables or {})
        else:
            kwargs["body"] = message
        msg = client.messages.create(**kwargs)
        logger.info("WhatsApp (Twilio) enviado a %s (sid=%s)", to, msg.sid)
        return NotificationResult(channel="WHATSAPP", to=to, status="SENT_REAL", provider_sid=msg.sid)
    except Exception as exc:
        logger.exception("Fallo al enviar WhatsApp (Twilio) a %s", to)
        return NotificationResult(channel="WHATSAPP", to=to, status="FAILED", detail=str(exc))


def _send_whatsapp_meta_cloud(to: str, template_variables: dict | None) -> NotificationResult:
    """API oficial de Meta (WhatsApp Cloud API), directa — sin el margen por
    mensaje que cobra un intermediario como Twilio. Igual requiere número de
    WhatsApp Business verificado y plantilla aprobada en Meta Business
    Manager; solo cambia quién factura el mensaje, no el requisito de Meta."""
    import requests

    if not (META_WHATSAPP_TOKEN and META_WHATSAPP_PHONE_NUMBER_ID and META_WHATSAPP_TEMPLATE_NAME):
        detail = ("META_WHATSAPP_TOKEN / META_WHATSAPP_PHONE_NUMBER_ID / "
                  "META_WHATSAPP_TEMPLATE_NAME no configurados")
        logger.error(detail)
        return NotificationResult(channel="WHATSAPP", to=to, status="FAILED", detail=detail)

    url = f"https://graph.facebook.com/{META_GRAPH_API_VERSION}/{META_WHATSAPP_PHONE_NUMBER_ID}/messages"
    components = []
    if template_variables:
        components = [{
            "type": "body",
            "parameters": [{"type": "text", "text": str(v)} for v in template_variables.values()],
        }]
    payload = {
        "messaging_product": "whatsapp",
        "to": to.lstrip("+"),
        "type": "template",
        "template": {
            "name": META_WHATSAPP_TEMPLATE_NAME,
            "language": {"code": META_WHATSAPP_TEMPLATE_LANG},
            "components": components,
        },
    }
    try:
        resp = requests.post(
            url, json=payload,
            headers={"Authorization": f"Bearer {META_WHATSAPP_TOKEN}"}, timeout=15,
        )
        resp.raise_for_status()
        msg_id = resp.json().get("messages", [{}])[0].get("id")
        logger.info("WhatsApp (Meta Cloud API) enviado a %s (id=%s)", to, msg_id)
        return NotificationResult(channel="WHATSAPP", to=to, status="SENT_REAL", provider_sid=msg_id)
    except Exception as exc:
        logger.exception("Fallo al enviar WhatsApp (Meta Cloud API) a %s", to)
        return NotificationResult(channel="WHATSAPP", to=to, status="FAILED", detail=str(exc))


def make_call(to: str, appointment_id: int | None = None) -> NotificationResult:
    """Dispara una llamada saliente con TTS y un <Gather> que apunta al
    webhook real de DTMF (app/api.py: POST /api/v1/webhook/twilio-dtmf). En
    modo simulación, la respuesta se captura desde la UI con
    `simulate_ivr_response`, no aquí."""
    if MOCK_NOTIFICATIONS:
        logger.info("[MOCK LLAMADA -> %s] Guion IVR: %s", to, IVR_SCRIPT)
        return NotificationResult(channel="LLAMADA_TELEFONICA", to=to, status="SENT_MOCK")
    if not TWILIO_FROM_NUMBER:
        logger.error("TWILIO_FROM_NUMBER no configurado; no se puede llamar a %s", to)
        return NotificationResult(channel="LLAMADA_TELEFONICA", to=to, status="FAILED",
                                   detail="TWILIO_FROM_NUMBER no configurado")
    try:
        from twilio.rest import Client
        from twilio.twiml.voice_response import Gather, VoiceResponse

        client: Client = _get_twilio_client()
        webhook_url = f"{PUBLIC_BASE_URL}/api/v1/webhook/twilio-dtmf"
        if appointment_id is not None:
            webhook_url += f"?appointment_id={appointment_id}"

        response = VoiceResponse()
        gather = Gather(num_digits=1, action=webhook_url, method="POST")
        gather.say(IVR_SCRIPT, language="es-MX")
        response.append(gather)
        response.say("No se recibió respuesta. Hasta luego.", language="es-MX")

        call = client.calls.create(twiml=str(response), from_=TWILIO_FROM_NUMBER, to=to)
        logger.info("Llamada iniciada a %s (sid=%s)", to, call.sid)
        return NotificationResult(channel="LLAMADA_TELEFONICA", to=to, status="SENT_REAL",
                                   provider_sid=call.sid)
    except Exception as exc:
        logger.exception("Fallo al llamar a %s", to)
        return NotificationResult(channel="LLAMADA_TELEFONICA", to=to, status="FAILED", detail=str(exc))


def validate_twilio_signature(url: str, params: dict, signature: str) -> bool:
    """Valida que un webhook entrante realmente provenga de Twilio (evita que
    cualquiera falsifique una respuesta de llamada). Debe usarse en
    app/api.py antes de procesar el webhook de DTMF en producción."""
    if MOCK_NOTIFICATIONS or not TWILIO_AUTH_TOKEN:
        return True  # sin credenciales reales no hay firma que validar (modo simulación/dev)
    from twilio.request_validator import RequestValidator

    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    return validator.validate(url, params, signature)


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
    logger.info("[ALERTA OPERATIVA] Visita domiciliaria requerida para paciente %s", to)
    return NotificationResult(channel="VISITA_DOMICILIARIA", to=to, status="SENT_MOCK",
                               detail="Alerta generada para personal de salud")
