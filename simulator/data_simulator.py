"""
Wearable & Lifestyle Continuous Data Simulator
Generates realistic physiological streams (Heart Rate, Blood Pressure, SpO2, Blood Glucose, Steps, Sleep)
and sends them to the FastAPI backend API in real time.
"""

import time
import math
import random
import argparse
import sys
import httpx
from datetime import datetime
from typing import Dict, Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_ENDPOINT = "http://localhost:8000/api/vitals/stream"

# Patient Behavioral Profiles
PATIENT_PROFILES = {
    "P001": {
        "name": "Sarah Chen (Low Risk)",
        "base_glucose": 88.0,
        "glucose_noise": 4.0,
        "base_bp_sys": 112.0,
        "base_bp_dia": 72.0,
        "base_hr": 62.0,
        "base_spo2": 99.0,
        "bmi": 22.4,
        "base_steps": 9200,
        "sleep_hours": 7.8,
        "age": 29
    },
    "P002": {
        "name": "Robert Miller (Moderate / Pre-Diabetic)",
        "base_glucose": 118.0,
        "glucose_noise": 7.0,
        "base_bp_sys": 130.0,
        "base_bp_dia": 84.0,
        "base_hr": 74.0,
        "base_spo2": 97.2,
        "bmi": 28.8,
        "base_steps": 4800,
        "sleep_hours": 6.2,
        "age": 52
    },
    "P003": {
        "name": "Eleanor Vance (High Risk / Urgent)",
        "base_glucose": 182.0,
        "glucose_noise": 14.0,
        "base_bp_sys": 152.0,
        "base_bp_dia": 94.0,
        "base_hr": 84.0,
        "base_spo2": 94.5,
        "bmi": 34.6,
        "base_steps": 2100,
        "sleep_hours": 5.2,
        "age": 65
    },
    "P004": {
        "name": "Marcus Thorne (Dynamic Acute Spike Demo)",
        "base_glucose": 94.0,
        "glucose_noise": 5.0,
        "base_bp_sys": 118.0,
        "base_bp_dia": 76.0,
        "base_hr": 66.0,
        "base_spo2": 98.2,
        "bmi": 25.1,
        "base_steps": 7100,
        "sleep_hours": 7.0,
        "age": 38
    },
    "P005": {
        "name": "Ashutosh (Demo Patient)",
        "base_glucose": 110.0,
        "glucose_noise": 6.5,
        "base_bp_sys": 124.0,
        "base_bp_dia": 80.0,
        "base_hr": 70.0,
        "base_spo2": 98.2,
        "bmi": 27.4,
        "base_steps": 6200,
        "sleep_hours": 6.8,
        "age": 42
    }
}


def generate_vitals_sample(patient_id: str, tick: int, inject_spike: bool = False) -> Dict[str, Any]:
    prof = PATIENT_PROFILES[patient_id]
    
    # Circadian diurnal wave
    diurnal_wave = math.sin(tick * 0.2)
    
    # Blood glucose simulation
    glucose = prof["base_glucose"] + diurnal_wave * (prof["glucose_noise"] * 0.8) + random.gauss(0, 2.0)
    variability = prof["glucose_noise"] + abs(random.gauss(0, 1.5))
    
    # If acute spike requested or for P004 intermittently
    if inject_spike or (patient_id == "P004" and tick % 6 == 0 and tick > 0):
        glucose += random.uniform(55.0, 85.0)
        variability += 15.0
        
    # Blood Pressure
    bp_sys = prof["base_bp_sys"] + diurnal_wave * 3.0 + random.gauss(0, 2.5)
    bp_dia = prof["base_bp_dia"] + diurnal_wave * 2.0 + random.gauss(0, 1.8)
    
    # Heart Rate & SpO2
    hr = prof["base_hr"] + diurnal_wave * 3.5 + random.gauss(0, 2.0)
    spo2 = prof["base_spo2"] + random.uniform(-0.5, 0.4)
    
    # Activity steps accumulation
    steps = int(prof["base_steps"] + (tick % 24) * 120 + random.randint(-50, 80))
    
    if inject_spike:
        bp_sys += 15.0
        hr += 10.0

    return {
        "patient_id": patient_id,
        "glucose": round(max(65.0, glucose), 1),
        "glucose_variability": round(max(5.0, variability), 1),
        "blood_pressure_systolic": round(max(90.0, bp_sys), 0),
        "blood_pressure_diastolic": round(max(55.0, bp_dia), 0),
        "heart_rate": round(max(50.0, hr), 0),
        "spo2": round(min(100.0, max(88.0, spo2)), 1),
        "bmi": prof["bmi"],
        "activity_steps": max(500, steps),
        "sleep_hours": prof["sleep_hours"],
        "age": prof["age"],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def run_simulator(
    api_url: str = API_ENDPOINT,
    interval: float = 2.5,
    max_ticks: int = None,
    trigger_spike: bool = False
):
    print("=" * 70)
    print("  AI-Powered Remote Patient Monitoring - Wearable Data Simulator")
    print(f"  Target API: {api_url}")
    print(f"  Stream Interval: {interval}s")
    print("=" * 70)
    
    tick = 0
    with httpx.Client(timeout=5.0) as client:
        try:
            while True:
                tick += 1
                print(f"\n--- [Cycle {tick}] Streaming Wearable Telemetry ---")
                
                for p_id in PATIENT_PROFILES:
                    # Decide if this reading should spike
                    is_spike = trigger_spike and (p_id == "P004" or p_id == "P003")
                    vitals = generate_vitals_sample(p_id, tick, inject_spike=is_spike)
                    
                    try:
                        resp = client.post(api_url, json=vitals)
                        if resp.status_code == 200:
                            data = resp.json()
                            sev = data["severity"].upper()
                            score = data["risk_score"]
                            factors = ", ".join(data.get("contributing_factors", [])[:2])
                            
                            badge = "[INFO]" if sev == "INFORMATIONAL" else ("[MODERATE]" if sev == "MODERATE" else "[URGENT!]")
                            cg_alert = " -> [CAREGIVER DISPATCH NOTIFIED]" if data.get("caregiver_notified") else ""
                            
                            print(
                                f"{badge} [{p_id}] {PATIENT_PROFILES[p_id]['name']}\n"
                                f"    Vitals: Glucose: {vitals['glucose']} mg/dL | BP: {int(vitals['blood_pressure_systolic'])}/{int(vitals['blood_pressure_diastolic'])} | HR: {int(vitals['heart_rate'])} bpm\n"
                                f"    AI Risk: {score*100:.1f}% | Tier: {sev}{cg_alert}\n"
                                f"    Key Factors: {factors}"
                            )
                        elif resp.status_code == 403:
                            print(f"[PRIVACY BLOCKED] [{p_id}] Telemetry Blocked: Patient has not provided privacy consent.")
                        else:
                            print(f"[ERROR] [{p_id}] Error {resp.status_code}: {resp.text}")
                    except Exception as err:
                        print(f"[CONN ERROR] Error sending to {api_url}: {err}")
                        print("   (Ensure backend FastAPI server is running: 'python -m uvicorn backend.main:app --port 8000')")
                        return

                if max_ticks and tick >= max_ticks:
                    print(f"\nCompleted {max_ticks} simulation cycles.")
                    break
                    
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nWearable data simulator stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wearable Data Stream Simulator for RPM")
    parser.add_argument("--url", type=str, default=API_ENDPOINT, help="FastAPI vitals ingestion URL")
    parser.add_argument("--interval", type=float, default=2.5, help="Seconds between streaming batches")
    parser.add_argument("--ticks", type=int, default=None, help="Number of cycles to simulate (default: infinite)")
    parser.add_argument("--spike", action="store_true", help="Inject acute metabolic spikes to test urgent alert triggers")
    args = parser.parse_args()

    run_simulator(api_url=args.url, interval=args.interval, max_ticks=args.ticks, trigger_spike=args.spike)
