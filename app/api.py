"""
Backend FastAPI (Componente F / Capa 5 de Servicio).

Ejecutar con:  uvicorn app.api:app --reload
Documentación interactiva (Swagger UI): http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import database, decision_engine, notification_service, train  # noqa: E402

app = FastAPI(
    title="Sistema Predictivo de No-Show Médico",
    description="API de predicción de riesgo, recomendación de intervención y "
                 "reasignación de cupos (UNSA - Ingeniería de Sistemas).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    with database.connect() as conn:
        database.init_db(conn)


# --------------------------------------------------------------------------
# Esquemas Pydantic
# --------------------------------------------------------------------------

class PredictRiskRequest(BaseModel):
    age: int = Field(ge=0, le=110)
    gender: Literal["M", "F"]
    hypertension: int = Field(ge=0, le=1)
    diabetes: int = Field(ge=0, le=1)
    alcoholism: int = Field(ge=0, le=1)
    scholarship: int = Field(ge=0, le=1)
    handicap: int = Field(ge=0, le=4)
    sms_received: int = Field(ge=0, le=1)
    lead_time: int = Field(ge=0)
    appointment_day_of_week: str
    contact_channel_available: Literal["WHATSAPP", "SMS_ONLY", "NO_PHONE"]
    appointment_id: Optional[int] = None


class PredictRiskResponse(BaseModel):
    probability: float
    risk_level: str
    top_features: list[dict]
    model_name: str


class RecommendInterventionRequest(BaseModel):
    risk_level: Literal["BAJO", "MEDIO", "ALTO"]
    contact_channel_available: Literal["WHATSAPP", "SMS_ONLY", "NO_PHONE"]
    has_chronic_condition: bool = False


class LogContactRequest(BaseModel):
    appointment_id: int
    channel_used: str
    outcome: Literal["CONFIRMÓ", "CANCELÓ", "REPROGRAMÓ", "SIN_RESPUESTA"]


class ReassignRequest(BaseModel):
    appointment_id: Optional[int] = None


# --------------------------------------------------------------------------
# RF-02/RF-03/RF-04
# --------------------------------------------------------------------------

@app.post("/api/v1/predict-risk", response_model=PredictRiskResponse)
def predict_risk(req: PredictRiskRequest):
    if not train.artifacts_available():
        raise HTTPException(503, "Modelo no entrenado. Ejecute 'python src/train.py' primero.")

    result = train.predict_case(req.model_dump(exclude={"appointment_id"}))

    if req.appointment_id is not None:
        with database.connect() as conn:
            database.insert_risk_prediction(
                conn, appointment_id=req.appointment_id, probability=result["probability"],
                risk_level=result["risk_level"], top_features=result["top_features"],
            )
    return result


# --------------------------------------------------------------------------
# RF-05
# --------------------------------------------------------------------------

@app.post("/api/v1/recommend-intervention")
def recommend_intervention(req: RecommendInterventionRequest):
    return decision_engine.recommend_intervention(
        req.risk_level, req.contact_channel_available, req.has_chronic_condition
    )


# --------------------------------------------------------------------------
# RF-06
# --------------------------------------------------------------------------

@app.post("/api/v1/audit/log-contact")
def log_contact(req: LogContactRequest):
    with database.connect() as conn:
        try:
            return decision_engine.register_contact_outcome(
                conn, req.appointment_id, req.channel_used, req.outcome
            )
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc


# --------------------------------------------------------------------------
# RF-07/RF-08
# --------------------------------------------------------------------------

@app.post("/api/v1/slots/reassign")
def reassign_slots(req: ReassignRequest):
    with database.connect() as conn:
        if req.appointment_id is not None:
            result = decision_engine.reassign_slot(conn, req.appointment_id)
            return {"reasignaciones": [result] if result else []}
        results = decision_engine.evaluate_and_release_expired(conn)
    return {"reasignaciones": results}


# --------------------------------------------------------------------------
# RF-10
# --------------------------------------------------------------------------

@app.get("/api/v1/metrics/kpis")
def get_kpis():
    with database.connect() as conn:
        return decision_engine.compute_kpis(conn)


# --------------------------------------------------------------------------
# Webhooks Twilio (Componente E, opcional - solo si MOCK_NOTIFICATIONS=False)
# --------------------------------------------------------------------------

@app.post("/api/v1/webhook/twilio-dtmf")
def twilio_dtmf_webhook(appointment_id: int, Digits: str = ""):
    outcome = notification_service.process_twilio_dtmf_webhook(Digits)
    with database.connect() as conn:
        database.insert_contact_audit(
            conn, appointment_id=appointment_id, channel_used="LLAMADA_TELEFONICA",
            outcome=outcome,
        )
    return Response(content="<Response><Say>Gracias.</Say></Response>",
                     media_type="application/xml")


@app.get("/")
def root():
    return {
        "sistema": "Sistema Predictivo de No-Show Médico",
        "docs": "/docs",
        "modelo_entrenado": train.artifacts_available(),
        "timestamp": datetime.now().isoformat(),
    }
