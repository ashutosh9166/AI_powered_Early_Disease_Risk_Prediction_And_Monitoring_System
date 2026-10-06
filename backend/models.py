"""
Pydantic Data Models for Remote Patient Monitoring Backend
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class VitalsReading(BaseModel):
    patient_id: str
    glucose: float = Field(..., description="Blood glucose in mg/dL")
    glucose_variability: Optional[float] = Field(12.0, description="Recent glucose variability SD")
    blood_pressure_systolic: float = Field(..., description="Systolic BP mmHg")
    blood_pressure_diastolic: float = Field(..., description="Diastolic BP mmHg")
    heart_rate: float = Field(..., description="Heart rate in bpm")
    spo2: float = Field(..., description="Blood oxygen saturation %")
    bmi: float = Field(..., description="Body Mass Index")
    activity_steps: float = Field(..., description="Daily steps")
    sleep_hours: float = Field(..., description="Sleep duration in hours")
    age: Optional[int] = Field(None, description="Patient age")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class PredictRequest(BaseModel):
    vitals: VitalsReading
    use_personalized_baseline: bool = True


class ContributingFactor(BaseModel):
    feature: str
    importance: float
    current_value: float
    baseline_value: Optional[float] = None
    delta_percent: Optional[float] = None
    clinical_insight: str


class AlertRecord(BaseModel):
    alert_id: str
    patient_id: str
    patient_name: str
    severity: str  # Informational, Moderate, Urgent
    risk_score: float
    timestamp: str
    summary_message: str
    patient_friendly_message: str
    contributing_factors: List[str]
    raw_vitals_snapshot: Dict[str, Any]
    baseline_comparison: Dict[str, Any]
    caregiver_notified: bool = False
    caregiver_message: Optional[str] = None
    acknowledged_by_clinician: bool = False
    acknowledged_at: Optional[str] = None


class PatientConsentUpdate(BaseModel):
    patient_id: str
    consent_given: bool
    scopes: List[str] = ["wearable_telemetry", "ai_risk_scoring", "clinician_sharing"]


class PatientProfile(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    avatar: str
    contact_hash: str
    caregiver_name: Optional[str] = None
    caregiver_contact: Optional[str] = None
    risk_tier: str
    consent_given: bool
    consent_timestamp: Optional[str] = None
    consent_scopes: List[str] = []
    baseline_vitals: Dict[str, float]
    current_vitals: Dict[str, float]
    latest_risk_score: float
    latest_severity: str
    doctor_id: str


class UserLoginRequest(BaseModel):
    email: str
    password: str


class UserRegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = Field(..., description="'Doctor' or 'Patient'")
    patient_id: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    title: Optional[str] = None
    patient_id: Optional[str] = None
    avatar: Optional[str] = None


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


class PatientVitalsIngestRequest(BaseModel):
    glucose: float = Field(..., description="Blood glucose in mg/dL")
    blood_pressure_systolic: float = Field(..., description="Systolic BP mmHg")
    blood_pressure_diastolic: float = Field(..., description="Diastolic BP mmHg")
    heart_rate: float = Field(..., description="Heart rate in bpm")
    spo2: float = Field(..., description="Blood oxygen saturation %")
    activity_steps: Optional[float] = Field(5000.0, description="Daily steps")
    sleep_hours: Optional[float] = Field(7.0, description="Sleep duration in hours")
    glucose_variability: Optional[float] = Field(12.0, description="Glucose variability SD")
    bmi: Optional[float] = Field(26.0, description="Body Mass Index")
    age: Optional[int] = Field(None, description="Patient age")


class ClinicianNoteRequest(BaseModel):
    note: str
    doctor_name: Optional[str] = "Dr. Alex Carter"


class AlertAcknowledgeRequest(BaseModel):
    note: Optional[str] = None
    doctor_name: Optional[str] = "Dr. Alex Carter"


class SimulatorScenarioRequest(BaseModel):
    scenario: str = Field(..., description="'stable', 'spike', 'critical'")
    patient_id: Optional[str] = "P003"
