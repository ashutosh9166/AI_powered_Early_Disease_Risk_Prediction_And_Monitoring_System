"""
In-Memory & Persistent Storage for Patients, Vitals History, and Alert Audit Logs.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from backend.models import PatientProfile, AlertRecord


DATABASE_FILE = "data/rpm_database.json"
SEED_FILE = "data/seed_patients.json"


class RPMDatabase:
    def __init__(self):
        self.patients: Dict[str, Dict[str, Any]] = {}
        self.vitals_history: Dict[str, List[Dict[str, Any]]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.users: Dict[str, Dict[str, Any]] = {}
        self.tokens: Dict[str, str] = {}  # token -> email
        self.notes: Dict[str, List[Dict[str, Any]]] = {}
        self._initialize_data()
        self._initialize_users()
        self._initialize_notes()

    def _initialize_data(self):
        # Load seed patients
        if os.path.exists(SEED_FILE):
            with open(SEED_FILE, "r") as f:
                seed_data = json.load(f)
                for item in seed_data:
                    p_id = item["patient_id"]
                    self.patients[p_id] = item
                    # Generate some realistic historical points for trend charts
                    self._generate_initial_history(p_id, item)
        else:
            print("Warning: Seed file not found.")

    def _generate_initial_history(self, patient_id: str, patient: Dict[str, Any]):
        """Generates past 12 hourly readings for trend visualization."""
        base = patient.get("baseline_vitals", {})
        curr = patient.get("current_vitals", {})
        now = datetime.utcnow()
        history = []
        
        for i in range(12, 0, -1):
            ts = (now - timedelta(hours=i)).strftime("%H:%M")
            # Interpolate between baseline and current with slight noise
            factor = (12 - i) / 12.0
            pt_glucose = base.get("glucose", 95) + factor * (curr.get("glucose", 95) - base.get("glucose", 95))
            pt_sys = base.get("blood_pressure_systolic", 120) + factor * (curr.get("blood_pressure_systolic", 120) - base.get("blood_pressure_systolic", 120))
            pt_hr = base.get("heart_rate", 68) + factor * (curr.get("heart_rate", 68) - base.get("heart_rate", 68))
            pt_spo2 = base.get("spo2", 98) + factor * (curr.get("spo2", 98) - base.get("spo2", 98))
            
            history.append({
                "time": ts,
                "glucose": round(pt_glucose, 1),
                "blood_pressure_systolic": round(pt_sys, 0),
                "blood_pressure_diastolic": round(base.get("blood_pressure_diastolic", 80), 0),
                "heart_rate": round(pt_hr, 0),
                "spo2": round(pt_spo2, 1),
                "activity_steps": int(base.get("activity_steps", 6000) * (0.8 + 0.4 * (12 - i) / 12))
            })
            
        self.vitals_history[patient_id] = history

    def get_all_patients(self) -> List[Dict[str, Any]]:
        # Return sorted by risk score descending (urgent triage first)
        patients_list = list(self.patients.values())
        return sorted(patients_list, key=lambda x: x.get("latest_risk_score", 0.0), reverse=True)

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        return self.patients.get(patient_id)

    def update_patient_vitals(
        self,
        patient_id: str,
        new_vitals: Dict[str, Any],
        new_baseline: Dict[str, float],
        risk_score: float,
        severity: str
    ):
        if patient_id not in self.patients:
            return
            
        self.patients[patient_id]["current_vitals"] = new_vitals
        self.patients[patient_id]["baseline_vitals"] = new_baseline
        self.patients[patient_id]["latest_risk_score"] = round(risk_score, 3)
        self.patients[patient_id]["latest_severity"] = severity
        self.patients[patient_id]["risk_tier"] = severity
        
        # Append to vitals history
        now_str = datetime.utcnow().strftime("%H:%M:%S")
        entry = {
            "time": now_str,
            "glucose": new_vitals.get("glucose"),
            "blood_pressure_systolic": new_vitals.get("blood_pressure_systolic"),
            "blood_pressure_diastolic": new_vitals.get("blood_pressure_diastolic"),
            "heart_rate": new_vitals.get("heart_rate"),
            "spo2": new_vitals.get("spo2"),
            "activity_steps": new_vitals.get("activity_steps", 0)
        }
        
        if patient_id not in self.vitals_history:
            self.vitals_history[patient_id] = []
        self.vitals_history[patient_id].append(entry)
        # Keep recent 40 data points
        if len(self.vitals_history[patient_id]) > 40:
            self.vitals_history[patient_id] = self.vitals_history[patient_id][-40:]

    def get_patient_history(self, patient_id: str) -> List[Dict[str, Any]]:
        return self.vitals_history.get(patient_id, [])

    def record_alert(self, alert: AlertRecord):
        self.alerts.insert(0, alert.dict())
        # Keep latest 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[:100]

    def get_alerts(self, severity_filter: Optional[str] = None, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        result = self.alerts
        if severity_filter:
            result = [a for a in result if a["severity"].lower() == severity_filter.lower()]
        if patient_id:
            result = [a for a in result if a["patient_id"] == patient_id]
        return result

    def _initialize_users(self):
        # Pre-seeded accounts requested for instant demo evaluation
        self.users["doctor@demo.com"] = {
            "id": "DOC-01",
            "email": "doctor@demo.com",
            "password": "password123",
            "name": "Dr. Alex Carter",
            "role": "Doctor",
            "title": "Clinician • Lead Endocrinologist",
            "patient_id": None,
            "avatar": "https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=150&auto=format&fit=crop&q=80"
        }
        self.users["patient@demo.com"] = {
            "id": "PAT-05",
            "email": "patient@demo.com",
            "password": "password123",
            "name": "Ashutosh",
            "role": "Patient",
            "title": "Patient • Monitored Cohort",
            "patient_id": "P005",
            "avatar": "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150&auto=format&fit=crop&q=80"
        }

    def _initialize_notes(self):
        now = datetime.utcnow()
        self.notes["P003"] = [
            {
                "note_id": "NOTE-101",
                "patient_id": "P003",
                "doctor_name": "Dr. Alex Carter",
                "timestamp": (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "content": "Alert reviewed. Elevated postprandial glucose (+23% above baseline). Patient advised hydration and scheduled tele-consultation. Caregiver David Vance verified."
            }
        ]
        self.notes["P002"] = [
            {
                "note_id": "NOTE-102",
                "patient_id": "P002",
                "doctor_name": "Dr. Alex Carter",
                "timestamp": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "content": "Mild pre-diabetic fluctuation. Recommended 30-minute daily walking and evening glycemic tracking."
            }
        ]
        self.notes["P005"] = [
            {
                "note_id": "NOTE-103",
                "patient_id": "P005",
                "doctor_name": "Dr. Alex Carter",
                "timestamp": (now - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
                "content": "Continuous telemetry baseline established. Vitals steady with mild afternoon glucose variation. Keep up 6,000 steps daily."
            }
        ]

    def authenticate(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        clean_email = email.strip().lower()
        user = self.users.get(clean_email)
        if user and user["password"] == password:
            token = f"vg_tok_{uuid.uuid4().hex}"
            self.tokens[token] = clean_email
            user_safe = dict(user)
            user_safe.pop("password", None)
            return {"token": token, "user": user_safe}
        return None

    def register(self, name: str, email: str, password: str, role: str, patient_id: Optional[str] = None) -> Dict[str, Any]:
        clean_email = email.strip().lower()
        if not clean_email or not password:
            raise ValueError("Email and password are required.")
        if clean_email in self.users:
            raise ValueError("An account with this email already exists. Please sign in.")
        
        normalized_role = "Doctor" if role.lower() == "doctor" else "Patient"
        
        if normalized_role == "Doctor":
            new_id = f"DOC-{uuid.uuid4().hex[:4].upper()}"
            doctor_name = name.strip() if name and name.strip() else "Dr. Clinician"
            user = {
                "id": new_id,
                "email": clean_email,
                "password": password,
                "name": doctor_name,
                "role": "Doctor",
                "title": "Clinician • Registered Physician",
                "patient_id": None,
                "avatar": "https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=150&auto=format&fit=crop&q=80"
            }
        else:
            patient_name = name.strip() if name and name.strip() else "Registered Patient"
            # Generate next patient ID if not provided
            if not patient_id:
                existing_pids = [int(p[1:]) for p in self.patients.keys() if p.startswith("P") and p[1:].isdigit()]
                next_num = max(existing_pids) + 1 if existing_pids else 6
                p_id = f"P{next_num:03d}"
            else:
                p_id = patient_id

            # Create patient record in database so doctor can view & triage
            self.patients[p_id] = {
                "patient_id": p_id,
                "name": patient_name,
                "age": 42,
                "gender": "Other",
                "avatar": "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150&auto=format&fit=crop&q=80",
                "contact_hash": f"usr_{uuid.uuid4().hex[:8]}",
                "caregiver_name": "Emergency Contact",
                "caregiver_contact": "+1-555-010-0000",
                "risk_tier": "Low",
                "consent_given": True,
                "consent_timestamp": datetime.utcnow().isoformat() + "Z",
                "consent_scopes": ["wearable_telemetry", "ai_risk_scoring", "clinician_sharing"],
                "baseline_vitals": {
                    "glucose": 95.0,
                    "blood_pressure_systolic": 120.0,
                    "blood_pressure_diastolic": 78.0,
                    "heart_rate": 68.0,
                    "spo2": 98.5,
                    "bmi": 24.5,
                    "activity_steps": 6500.0,
                    "sleep_hours": 7.2
                },
                "current_vitals": {
                    "glucose": 96.0,
                    "blood_pressure_systolic": 122.0,
                    "blood_pressure_diastolic": 80.0,
                    "heart_rate": 70.0,
                    "spo2": 98.2,
                    "bmi": 24.5,
                    "activity_steps": 6200.0,
                    "sleep_hours": 7.0
                },
                "latest_risk_score": 0.12,
                "latest_severity": "Informational",
                "doctor_id": "DOC-01"
            }
            # Generate trend history for patient
            self._generate_initial_history(p_id, self.patients[p_id])

            new_id = f"PAT-{uuid.uuid4().hex[:4].upper()}"
            user = {
                "id": new_id,
                "email": clean_email,
                "password": password,
                "name": patient_name,
                "role": "Patient",
                "title": "Patient • Monitored Cohort",
                "patient_id": p_id,
                "avatar": "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150&auto=format&fit=crop&q=80"
            }
        
        self.users[clean_email] = user
        token = f"vg_tok_{uuid.uuid4().hex}"
        self.tokens[token] = clean_email
        user_safe = dict(user)
        user_safe.pop("password", None)
        return {"token": token, "user": user_safe}

    def get_user_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        email = self.tokens.get(token)
        if email and email in self.users:
            user_safe = dict(self.users[email])
            user_safe.pop("password", None)
            return user_safe
        return None

    def acknowledge_alert(self, alert_id: str, note: Optional[str] = None, doctor_name: str = "Dr. Alex Carter") -> bool:
        for alert in self.alerts:
            if alert["alert_id"] == alert_id:
                alert["acknowledged_by_clinician"] = True
                alert["acknowledged_at"] = datetime.utcnow().isoformat() + "Z"
                alert["acknowledged_by"] = doctor_name
                if note:
                    alert["clinician_note"] = note
                    p_id = alert.get("patient_id")
                    if p_id:
                        self.add_note(p_id, f"[Alert Ack Note] {note}", doctor_name)
                return True
        return False

    def add_note(self, patient_id: str, note_text: str, doctor_name: str = "Dr. Alex Carter") -> Dict[str, Any]:
        if patient_id not in self.notes:
            self.notes[patient_id] = []
        note_entry = {
            "note_id": f"NOTE-{uuid.uuid4().hex[:6].upper()}",
            "patient_id": patient_id,
            "doctor_name": doctor_name,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "content": note_text
        }
        self.notes[patient_id].insert(0, note_entry)
        return note_entry

    def get_notes(self, patient_id: str) -> List[Dict[str, Any]]:
        return self.notes.get(patient_id, [])

    def update_consent(self, patient_id: str, consent_given: bool, scopes: List[str]) -> bool:
        if patient_id in self.patients:
            self.patients[patient_id]["consent_given"] = consent_given
            self.patients[patient_id]["consent_scopes"] = scopes
            self.patients[patient_id]["consent_timestamp"] = datetime.utcnow().isoformat() + "Z"
            return True
        return False


db = RPMDatabase()
