"""
FastAPI Backend for AI-Powered Early Disease Risk Prediction & Remote Patient Monitoring System
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware

from backend.models import (
    VitalsReading,
    PredictRequest,
    AlertRecord,
    PatientConsentUpdate,
    PatientProfile,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
    AuthResponse,
    PatientVitalsIngestRequest,
    ClinicianNoteRequest,
    AlertAcknowledgeRequest,
    SimulatorScenarioRequest
)
from backend.baseline_engine import PersonalizedBaselineEngine
from backend.alert_engine import AlertEngine
from backend.database import db


app = FastAPI(
    title="VitalGuard AI — Remote Patient Monitoring API",
    description="Backend API for continuous remote patient monitoring, baseline learning, XGBoost ML risk prediction, and explainable alerts.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
baseline_engine = PersonalizedBaselineEngine(smoothing_factor=0.15)
alert_engine = AlertEngine()

# Model and Explainer holders
model = None
scaler = None
explainer = None
model_metadata = {}

MODEL_DIR = "ml/models"

def load_ml_assets():
    global model, scaler, explainer, model_metadata
    model_path = os.path.join(MODEL_DIR, "best_diabetes_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    explainer_path = os.path.join(MODEL_DIR, "shap_explainer.joblib")
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Loaded ML model from {model_path}")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    if os.path.exists(explainer_path):
        explainer = joblib.load(explainer_path)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            model_metadata = json.load(f)

load_ml_assets()

# Active WebSocket connections for live remote monitoring
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

ws_manager = ConnectionManager()


# --- Explainability Helper ---
def compute_ml_prediction_and_shap(vitals_dict: Dict[str, Any], baseline_dict: Dict[str, float] = None) -> Dict[str, Any]:
    features_list = model_metadata.get("features", [
        "glucose", "glucose_variability", "blood_pressure_systolic", "blood_pressure_diastolic",
        "heart_rate", "spo2", "bmi", "activity_steps", "sleep_hours", "age"
    ])
    
    # Prepare feature vector
    input_row = [float(vitals_dict.get(feat, 0.0)) for feat in features_list]
    X_df = pd.DataFrame([input_row], columns=features_list)
    
    # Predict Probability
    if model is not None:
        try:
            # XGBoost or tree models
            prob = float(model.predict_proba(X_df)[0][1])
        except Exception:
            prob = 0.5
    else:
        # Fallback heuristic if model not yet loaded
        prob = min(1.0, max(0.0, (vitals_dict.get("glucose", 90) - 70) / 140.0))

    # Compute SHAP Values
    shap_contributions = {}
    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(X_df)
            if isinstance(shap_vals, list):
                # Class 1 (positive risk)
                vals = shap_vals[1][0]
            else:
                vals = shap_vals[0]
            for feat, val in zip(features_list, vals):
                shap_contributions[feat] = round(float(val), 4)
        except Exception as e:
            # Fallback to feature importance weighting
            for feat in features_list:
                shap_contributions[feat] = 0.05
    else:
        for feat in features_list:
            shap_contributions[feat] = 0.05

    # Generate Human-Friendly Explanation with personalized baseline
    factors = []
    sorted_factors = sorted(shap_contributions.items(), key=lambda x: x[1], reverse=True)
    
    for feature, shap_val in sorted_factors:
        curr_val = vitals_dict.get(feature)
        if curr_val is None:
            continue
        base_val = baseline_dict.get(feature) if baseline_dict else None
        
        diff_str = ""
        if base_val and base_val > 0:
            pct = ((curr_val - base_val) / base_val) * 100
            if abs(pct) >= 4:
                diff_str = f" ({'+' if pct > 0 else ''}{pct:.1f}% vs your baseline of {base_val:.1f})"
                
        if feature == "glucose" and (curr_val >= 105 or shap_val > 0.02):
            factors.append(f"Blood glucose elevated at {curr_val:.1f} mg/dL{diff_str}")
        elif feature == "glucose_variability" and curr_val > 15:
            factors.append(f"High glucose fluctuation ±{curr_val:.1f} mg/dL{diff_str}")
        elif feature == "bmi" and curr_val >= 28:
            factors.append(f"Elevated BMI at {curr_val:.1f}")
        elif feature == "blood_pressure_systolic" and curr_val >= 130:
            dia = vitals_dict.get("blood_pressure_diastolic", 80)
            factors.append(f"Elevated blood pressure {curr_val:.0f}/{dia:.0f} mmHg{diff_str}")
        elif feature == "activity_steps" and curr_val < 5000:
            factors.append(f"Low physical activity ({int(curr_val):,} steps){diff_str}")
        elif feature == "sleep_hours" and curr_val < 6.0:
            factors.append(f"Insufficient sleep ({curr_val:.1f} hrs){diff_str}")
        elif feature == "heart_rate" and curr_val > 88:
            factors.append(f"Elevated resting heart rate ({int(curr_val)} bpm){diff_str}")

    top_factors = factors[:3] if factors else ["All monitored parameters are in target ranges."]
    
    risk_percentage = round(prob * 100, 1)
    if prob >= 0.70:
        tier_label = "Urgent/Critical"
    elif prob >= 0.35:
        tier_label = "Moderate/Warning"
    else:
        tier_label = "Normal"

    return {
        "risk_probability": round(prob, 3),
        "risk_percentage": risk_percentage,
        "risk_tier": tier_label,
        "shap_contributions": shap_contributions,
        "top_contributing_factors": top_factors,
        "plain_language_summary": f"Risk primarily influenced by: " + "; ".join(top_factors)
    }


# --- API Routes ---

@app.get("/")
def read_root():
    return {
        "service": "VitalGuard AI — AI-Powered Early Disease Risk Prediction & Remote Patient Monitoring API",
        "status": "online",
        "model_loaded": model is not None,
        "best_model_name": model_metadata.get("best_model_name", "XGBoost Classifier"),
        "version": "2.0.0"
    }


@app.get("/api/model/metrics")
def get_model_metrics():
    """Returns trained ML model performance metrics, comparison table, and feature importances."""
    if not model_metadata:
        load_ml_assets()
    return model_metadata


# --- Authentication Endpoints ---

@app.post("/api/auth/login", response_model=AuthResponse)
def login(request: UserLoginRequest):
    """
    Role-based authentication endpoint for Doctor and Patient accounts.
    Pre-seeded demo credentials:
      - Doctor: doctor@demo.com / password123
      - Patient: patient@demo.com / password123
    """
    res = db.authenticate(request.email, request.password)
    if not res:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return res


@app.post("/api/auth/register", response_model=AuthResponse)
def register(request: UserRegisterRequest):
    """
    Registers a new Doctor or Patient account and issues a session token.
    """
    try:
        res = db.register(
            name=request.name,
            email=request.email,
            password=request.password,
            role=request.role,
            patient_id=request.patient_id
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/auth/me", response_model=UserResponse)
def get_current_user(authorization: Optional[str] = Header(None)):
    """Validates session token and returns active logged-in user profile."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing.")
    token = authorization.replace("Bearer ", "").strip()
    user = db.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid token.")
    return user


# --- Telemetry Ingestion Core Engine ---

async def process_and_record_telemetry(patient_id: str, reading_dict: Dict[str, Any]) -> Dict[str, Any]:
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not registered")

    # 1. Privacy & Consent Enforcement Check
    if not patient.get("consent_given", False):
        raise HTTPException(
            status_code=403,
            detail=f"Patient {patient_id} has not granted data telemetry and AI risk processing consent."
        )

    reading_dict["patient_id"] = patient_id
    if reading_dict.get("age") is None:
        reading_dict["age"] = patient.get("age", 45)

    current_baseline = patient.get("baseline_vitals", {})
    
    # 2. ML Risk Scoring & SHAP Attribution
    ml_result = compute_ml_prediction_and_shap(reading_dict, current_baseline)
    risk_score = ml_result["risk_probability"]

    # 3. Personalized Baseline Evaluation
    baseline_comp = baseline_engine.generate_comprehensive_comparison(reading_dict, current_baseline)

    # 4. Severity Tiering & Routing
    alert_record = alert_engine.evaluate_and_route_alert(
        patient_id=patient["patient_id"],
        patient_name=patient["name"],
        risk_probability=risk_score,
        current_vitals=reading_dict,
        baseline_comparison=baseline_comp,
        shap_explanation=ml_result,
        caregiver_name=patient.get("caregiver_name"),
        caregiver_contact=patient.get("caregiver_contact")
    )

    # 5. Update Personalized Baseline (Learns rolling individual range)
    new_baseline = baseline_engine.update_baseline(current_baseline, reading_dict)

    # 6. Database Updates & Audit Logging
    db.update_patient_vitals(
        patient_id=patient["patient_id"],
        new_vitals=reading_dict,
        new_baseline=new_baseline,
        risk_score=risk_score,
        severity=alert_record.severity
    )
    db.record_alert(alert_record)

    # 7. Real-Time Broadcast via WebSocket
    payload = {
        "event": "VITALS_UPDATE",
        "patient_id": patient["patient_id"],
        "patient_name": patient["name"],
        "severity": alert_record.severity,
        "risk_tier": alert_record.severity,
        "risk_score": risk_score,
        "risk_percentage": ml_result.get("risk_percentage", round(risk_score * 100, 1)),
        "vitals": reading_dict,
        "baseline_comparison": baseline_comp,
        "alert": alert_record.dict(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    await ws_manager.broadcast(payload)

    return {
        "status": "success",
        "patient_id": patient["patient_id"],
        "patient_name": patient["name"],
        "severity": alert_record.severity,
        "risk_tier": alert_record.severity,
        "risk_score": risk_score,
        "risk_percentage": ml_result.get("risk_percentage", round(risk_score * 100, 1)),
        "contributing_factors": ml_result["top_contributing_factors"],
        "plain_language_summary": ml_result.get("plain_language_summary", ""),
        "caregiver_notified": alert_record.caregiver_notified,
        "alert_id": alert_record.alert_id,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# --- Doctor Dashboard Endpoints ---

@app.get("/api/doctor/patients")
def get_doctor_patients():
    """
    Returns monitored patient list sorted by triage risk score,
    enriched with active alert counts and clinical notes.
    """
    patients = db.get_all_patients()
    enriched = []
    for p in patients:
        item = dict(p)
        p_id = item["patient_id"]
        item["notes"] = db.get_notes(p_id)
        item["active_alerts_count"] = len([
            a for a in db.get_alerts(patient_id=p_id)
            if not a.get("acknowledged_by_clinician", False)
        ])
        if "contact_hash" not in item:
            item["contact_hash"] = "hashed_" + p_id
        enriched.append(item)
    return enriched


@app.get("/api/doctor/patients/{patient_id}")
def get_doctor_patient_detail(patient_id: str):
    """
    Returns full patient detail, trend trajectory history,
    active alerts, clinical notes, baseline comparison, and XAI explanations.
    """
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    comparison = baseline_engine.generate_comprehensive_comparison(
        patient.get("current_vitals", {}),
        patient.get("baseline_vitals", {})
    )
    history = db.get_patient_history(patient_id)
    alerts = db.get_alerts(patient_id=patient_id)
    notes = db.get_notes(patient_id)
    explanation = compute_ml_prediction_and_shap(
        patient.get("current_vitals", {}),
        patient.get("baseline_vitals", {})
    )
    
    return {
        "profile": patient,
        "baseline_comparison": comparison,
        "vitals_history": history,
        "alerts": alerts,
        "notes": notes,
        "xai_explanation": explanation
    }


@app.post("/api/doctor/patients/{patient_id}/notes")
def add_patient_clinical_note(patient_id: str, request: ClinicianNoteRequest):
    """Adds a clinician note/triage action for a patient."""
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    note = db.add_note(patient_id, request.note, request.doctor_name or "Dr. Alex Carter")
    return {"status": "success", "note": note}


# Backwards compatibility endpoints
@app.get("/api/patients")
def list_patients_compat():
    return get_doctor_patients()


@app.get("/api/patients/{patient_id}")
def get_patient_detail_compat(patient_id: str):
    return get_doctor_patient_detail(patient_id)


# --- Patient Portal Endpoints ---

@app.get("/api/patient/{patient_id}/vitals")
def get_patient_vitals_summary(patient_id: str):
    """
    Returns historical and current vitals, baseline comparison,
    safety indicators, and empathetic plain-language AI feedback.
    """
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    comparison = baseline_engine.generate_comprehensive_comparison(
        patient.get("current_vitals", {}),
        patient.get("baseline_vitals", {})
    )
    history = db.get_patient_history(patient_id)
    explanation = compute_ml_prediction_and_shap(
        patient.get("current_vitals", {}),
        patient.get("baseline_vitals", {})
    )
    alerts = db.get_alerts(patient_id=patient_id)

    return {
        "patient_id": patient_id,
        "patient_name": patient.get("name"),
        "current_vitals": patient.get("current_vitals", {}),
        "baseline_vitals": patient.get("baseline_vitals", {}),
        "latest_risk_score": patient.get("latest_risk_score", 0.0),
        "latest_severity": patient.get("latest_severity", "Normal"),
        "risk_tier": patient.get("risk_tier", "Normal"),
        "vitals_history": history,
        "baseline_comparison": comparison,
        "plain_language_summary": explanation.get("plain_language_summary"),
        "top_contributing_factors": explanation.get("top_contributing_factors", []),
        "active_alerts": alerts[:5]
    }


@app.post("/api/patient/{patient_id}/vitals")
async def ingest_patient_vitals(patient_id: str, reading: PatientVitalsIngestRequest):
    """Ingests new vitals for a specific patient, scores ML risk, and evaluates baseline."""
    reading_dict = reading.dict()
    return await process_and_record_telemetry(patient_id, reading_dict)


@app.post("/api/vitals/stream")
async def ingest_vitals_stream(reading: VitalsReading):
    """Continuous stream telemetry ingestion for wearable devices & simulator."""
    reading_dict = reading.dict()
    return await process_and_record_telemetry(reading.patient_id, reading_dict)


# --- Realistic IoT / Sensor Stream Simulator Endpoints ---

@app.post("/api/simulator/scenario")
async def simulate_scenario(req: SimulatorScenarioRequest):
    """
    Simulates predefined clinical presets:
      - 'stable': Normal healthy readings within baseline
      - 'spike': Pre-diabetic elevation with baseline deviation
      - 'critical': Immediate urgent tier metabolic spike with caregiver dispatch
    """
    p_id = req.patient_id or "P003"
    patient = db.get_patient(p_id)
    if not patient:
        p_id = "P003"
        patient = db.get_patient("P003")

    scenario_name = req.scenario.lower().strip()
    base = patient.get("baseline_vitals", {})

    if scenario_name == "stable":
        reading = {
            "glucose": 90.5,
            "glucose_variability": 7.8,
            "blood_pressure_systolic": 114.0,
            "blood_pressure_diastolic": 74.0,
            "heart_rate": 64.0,
            "spo2": 99.1,
            "bmi": base.get("bmi", 23.5),
            "activity_steps": 8900.0,
            "sleep_hours": 7.6,
            "age": patient.get("age", 40)
        }
    elif scenario_name == "spike":
        reading = {
            "glucose": 138.0,
            "glucose_variability": 22.5,
            "blood_pressure_systolic": 134.0,
            "blood_pressure_diastolic": 86.0,
            "heart_rate": 78.0,
            "spo2": 96.6,
            "bmi": base.get("bmi", 28.5),
            "activity_steps": 3900.0,
            "sleep_hours": 5.8,
            "age": patient.get("age", 52)
        }
    else:  # 'critical'
        reading = {
            "glucose": 218.4,
            "glucose_variability": 42.0,
            "blood_pressure_systolic": 168.0,
            "blood_pressure_diastolic": 102.0,
            "heart_rate": 96.0,
            "spo2": 93.4,
            "bmi": base.get("bmi", 34.6),
            "activity_steps": 1100.0,
            "sleep_hours": 4.5,
            "age": patient.get("age", 65)
        }

    return await process_and_record_telemetry(p_id, reading)


# --- Alerts & Acknowledgment Endpoints ---

@app.get("/api/alerts")
def get_alerts(
    severity: Optional[str] = Query(None, description="urgent, moderate, informational"),
    patient_id: Optional[str] = Query(None)
):
    """Returns alert feed and model performance audit log."""
    return db.get_alerts(severity_filter=severity, patient_id=patient_id)


@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, request: Optional[AlertAcknowledgeRequest] = None):
    """Allows clinician to review and acknowledge an alert with optional clinical note."""
    note = request.note if request else None
    doctor = request.doctor_name if request else "Dr. Alex Carter"
    success = db.acknowledge_alert(alert_id, note=note, doctor_name=doctor)
    if not success:
        raise HTTPException(status_code=404, detail="Alert ID not found")
    return {"status": "acknowledged", "alert_id": alert_id, "acknowledged_by": doctor}


@app.post("/api/consent")
def update_patient_consent(consent: PatientConsentUpdate):
    """
    Patient Consent Management: Records agreement or revocation before data is processed.
    """
    success = db.update_consent(
        patient_id=consent.patient_id,
        consent_given=consent.consent_given,
        scopes=consent.scopes
    )
    if not success:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {
        "status": "updated",
        "patient_id": consent.patient_id,
        "consent_given": consent.consent_given,
        "scopes": consent.scopes,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.websocket("/ws/monitoring")
async def websocket_monitoring(websocket: WebSocket):
    """WebSocket endpoint for real-time live remote patient monitoring stream."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; can receive client control messages
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
