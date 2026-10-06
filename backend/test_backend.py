"""
Unit & Integration Test Suite for FastAPI RPM Backend
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root_status():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True

def test_list_patients():
    response = client.get("/api/patients")
    assert response.status_code == 200
    patients = response.json()
    assert len(patients) >= 3
    # Verify sorted by risk score
    scores = [p["latest_risk_score"] for p in patients]
    assert scores == sorted(scores, reverse=True)

def test_patient_detail_and_baseline_comparison():
    response = client.get("/api/patients/P001")
    assert response.status_code == 200
    data = response.json()
    assert data["profile"]["patient_id"] == "P001"
    assert "baseline_comparison" in data
    assert "glucose" in data["baseline_comparison"]
    assert "vitals_history" in data

def test_stream_vitals_low_risk():
    reading = {
        "patient_id": "P001",
        "glucose": 89.0,
        "glucose_variability": 7.5,
        "blood_pressure_systolic": 112.0,
        "blood_pressure_diastolic": 72.0,
        "heart_rate": 62.0,
        "spo2": 99.0,
        "bmi": 22.4,
        "activity_steps": 9500.0,
        "sleep_hours": 7.8,
        "age": 29
    }
    response = client.post("/api/vitals/stream", json=reading)
    assert response.status_code == 200
    res = response.json()
    assert res["severity"] == "Informational"
    assert res["risk_score"] < 0.35

def test_stream_vitals_moderate_risk():
    reading = {
        "patient_id": "P002",
        "glucose": 116.0,
        "glucose_variability": 15.0,
        "blood_pressure_systolic": 126.0,
        "blood_pressure_diastolic": 82.0,
        "heart_rate": 72.0,
        "spo2": 97.5,
        "bmi": 28.0,
        "activity_steps": 5400.0,
        "sleep_hours": 6.5,
        "age": 52
    }
    response = client.post("/api/vitals/stream", json=reading)
    assert response.status_code == 200
    res = response.json()
    assert res["severity"] == "Moderate"
    assert len(res["contributing_factors"]) > 0

def test_stream_vitals_urgent_risk_and_caregiver_alert():
    reading = {
        "patient_id": "P003",
        "glucose": 215.0,
        "glucose_variability": 38.0,
        "blood_pressure_systolic": 165.0,
        "blood_pressure_diastolic": 98.0,
        "heart_rate": 88.0,
        "spo2": 93.8,
        "bmi": 35.0,
        "activity_steps": 1200.0,
        "sleep_hours": 4.8,
        "age": 65
    }
    response = client.post("/api/vitals/stream", json=reading)
    assert response.status_code == 200
    res = response.json()
    assert res["severity"] == "Urgent"
    assert res["caregiver_notified"] is True

def test_privacy_consent_revocation_enforcement():
    # Revoke consent for P004
    revoke_res = client.post("/api/consent", json={
        "patient_id": "P004",
        "consent_given": False,
        "scopes": []
    })
    assert revoke_res.status_code == 200
    
    # Try sending telemetry without consent -> must return 403 Forbidden
    reading = {
        "patient_id": "P004",
        "glucose": 95.0,
        "blood_pressure_systolic": 118.0,
        "blood_pressure_diastolic": 76.0,
        "heart_rate": 65.0,
        "spo2": 98.5,
        "bmi": 25.1,
        "activity_steps": 7000.0,
        "sleep_hours": 7.0,
        "age": 38
    }
    blocked_res = client.post("/api/vitals/stream", json=reading)
    assert blocked_res.status_code == 403
    assert "consent" in blocked_res.json()["detail"].lower()
    
    # Re-grant consent
    grant_res = client.post("/api/consent", json={
        "patient_id": "P004",
        "consent_given": True,
        "scopes": ["wearable_telemetry", "ai_risk_scoring"]
    })
    assert grant_res.status_code == 200
    allowed_res = client.post("/api/vitals/stream", json=reading)
    assert allowed_res.status_code == 200

def test_alerts_audit_and_acknowledgment():
    alerts_res = client.get("/api/alerts")
    assert alerts_res.status_code == 200
    alerts = alerts_res.json()
    assert len(alerts) > 0
    
    alert_to_ack = alerts[0]["alert_id"]
    ack_res = client.post(f"/api/alerts/{alert_to_ack}/acknowledge", json={"note": "Reviewed by clinician"})
    assert ack_res.status_code == 200
    assert ack_res.json()["status"] == "acknowledged"


def test_auth_doctor_and_patient_login():
    # Doctor Demo Account
    doc_res = client.post("/api/auth/login", json={"email": "doctor@demo.com", "password": "password123"})
    assert doc_res.status_code == 200
    doc_data = doc_res.json()
    assert doc_data["user"]["role"] == "Doctor"
    assert "token" in doc_data
    
    # Patient Demo Account
    pat_res = client.post("/api/auth/login", json={"email": "patient@demo.com", "password": "password123"})
    assert pat_res.status_code == 200
    pat_data = pat_res.json()
    assert pat_data["user"]["role"] == "Patient"
    assert pat_data["user"]["name"] == "Ashutosh"
    
    # Invalid credentials
    bad_res = client.post("/api/auth/login", json={"email": "doctor@demo.com", "password": "wrong"})
    assert bad_res.status_code == 401


def test_auth_registration():
    # Register new Doctor
    new_doc_res = client.post("/api/auth/register", json={
        "name": "Dr. Meredith Grey",
        "email": "meredith@hospital.org",
        "password": "pass_doctor_456",
        "role": "Doctor"
    })
    assert new_doc_res.status_code == 200
    doc_data = new_doc_res.json()
    assert doc_data["user"]["role"] == "Doctor"
    assert doc_data["user"]["name"] == "Dr. Meredith Grey"
    assert "token" in doc_data

    # Login with newly registered doctor
    login_doc_res = client.post("/api/auth/login", json={
        "email": "meredith@hospital.org",
        "password": "pass_doctor_456"
    })
    assert login_doc_res.status_code == 200

    # Register new Patient
    new_pat_res = client.post("/api/auth/register", json={
        "name": "John Doe",
        "email": "john.doe@example.com",
        "password": "pass_patient_123",
        "role": "Patient"
    })
    assert new_pat_res.status_code == 200
    pat_data = new_pat_res.json()
    assert pat_data["user"]["role"] == "Patient"
    assert pat_data["user"]["name"] == "John Doe"
    assert pat_data["user"]["patient_id"] is not None

    # Verify patient profile was created in database
    p_id = pat_data["user"]["patient_id"]
    pt_res = client.get(f"/api/patients/{p_id}")
    assert pt_res.status_code == 200
    assert pt_res.json()["profile"]["name"] == "John Doe"

    # Test duplicate registration error
    dup_res = client.post("/api/auth/register", json={
        "name": "Duplicate User",
        "email": "meredith@hospital.org",
        "password": "some_password",
        "role": "Doctor"
    })
    assert dup_res.status_code == 400
    assert "already exists" in dup_res.json()["detail"].lower()


def test_doctor_patients_and_notes():
    res = client.get("/api/doctor/patients")
    assert res.status_code == 200
    patients = res.json()
    assert len(patients) >= 4
    
    # Add clinician note to Eleanor Vance (P003)
    note_res = client.post("/api/doctor/patients/P003/notes", json={
        "note": "Prescribed continuous glucose monitor and ordered HbA1c panel.",
        "doctor_name": "Dr. Alex Carter"
    })
    assert note_res.status_code == 200
    assert note_res.json()["status"] == "success"


def test_patient_vitals_summary_and_ingestion():
    # GET vitals for P005 (Ashutosh)
    vitals_res = client.get("/api/patient/P005/vitals")
    assert vitals_res.status_code == 200
    data = vitals_res.json()
    assert "current_vitals" in data
    assert "baseline_comparison" in data
    assert "plain_language_summary" in data

    # POST new vital reading
    ingest_res = client.post("/api/patient/P005/vitals", json={
        "glucose": 132.0,
        "blood_pressure_systolic": 128.0,
        "blood_pressure_diastolic": 82.0,
        "heart_rate": 72.0,
        "spo2": 98.0,
        "activity_steps": 5400.0,
        "sleep_hours": 6.8
    })
    assert ingest_res.status_code == 200
    assert ingest_res.json()["status"] == "success"
    assert "risk_score" in ingest_res.json()


def test_simulator_scenarios():
    # Test Stable Preset
    stable_res = client.post("/api/simulator/scenario", json={"scenario": "stable", "patient_id": "P001"})
    assert stable_res.status_code == 200
    assert stable_res.json()["severity"] == "Informational"

    # Test Critical Preset
    critical_res = client.post("/api/simulator/scenario", json={"scenario": "critical", "patient_id": "P003"})
    assert critical_res.status_code == 200
    assert critical_res.json()["severity"] == "Urgent"
    assert critical_res.json()["caregiver_notified"] is True


if __name__ == "__main__":
    test_root_status()
    test_list_patients()
    test_patient_detail_and_baseline_comparison()
    test_stream_vitals_low_risk()
    test_stream_vitals_moderate_risk()
    test_stream_vitals_urgent_risk_and_caregiver_alert()
    test_privacy_consent_revocation_enforcement()
    test_alerts_audit_and_acknowledgment()
    test_auth_doctor_and_patient_login()
    test_auth_registration()
    test_doctor_patients_and_notes()
    test_patient_vitals_summary_and_ingestion()
    test_simulator_scenarios()
    print("ALL 13 BACKEND INTEGRATION & SPEC TESTS PASSED SUCCESSFULLY!")
