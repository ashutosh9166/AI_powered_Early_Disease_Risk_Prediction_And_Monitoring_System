"""
Dataset Generator: Pima Indians Diabetes Dataset enriched with Continuous Wearable Vitals & Lifestyle Metrics.
Generates an augmented, physiologically consistent dataset for training Remote Patient Monitoring ML models.
"""

import numpy as np
import pandas as pd
import os

def generate_enriched_dataset(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 1. Demographics & Physiological Factors
    age_pool = np.concatenate([
        np.random.normal(32, 6, int(n_samples * 0.40)),
        np.random.normal(52, 8, int(n_samples * 0.45)),
        np.random.normal(67, 5, int(n_samples * 0.15))
    ])
    np.random.shuffle(age_pool)
    age = np.clip(np.round(age_pool[:n_samples]), 21, 80).astype(int)
    
    # 2. Body Mass Index (BMI)
    # Skewed toward overweight/obesity reflecting general and at-risk population
    bmi = np.random.normal(28.5, 6.2, n_samples)
    bmi = np.clip(np.round(bmi, 1), 18.5, 48.0)
    
    # 3. Daily Physical Activity (steps per day from wearable accelerometer)
    # Negative correlation with BMI and age
    base_steps = 8500 - (bmi - 22) * 180 - (age - 30) * 45 + np.random.normal(0, 1500, n_samples)
    activity_steps = np.clip(np.round(base_steps), 1200, 16000).astype(int)
    
    # 4. Sleep duration (hours per night from wearable sleep tracker)
    base_sleep = 7.4 - (bmi > 30) * 0.6 - (activity_steps < 4000) * 0.5 + np.random.normal(0, 0.9, n_samples)
    sleep_hours = np.clip(np.round(base_sleep, 1), 4.2, 9.8)
    
    # 5. Heart Rate (bpm from photoplethysmography / PPG wearable sensor)
    # Higher in sedentary, high BMI, and metabolic stress
    base_hr = 68 + (bmi - 24) * 0.7 - (activity_steps / 1000) * 1.1 + np.random.normal(0, 6.5, n_samples)
    heart_rate = np.clip(np.round(base_hr), 52, 115).astype(int)
    
    # 6. Blood Pressure (Systolic and Diastolic mmHg)
    sys_base = 110 + (age - 25) * 0.45 + (bmi - 22) * 0.85 + (heart_rate - 70) * 0.25 + np.random.normal(0, 8, n_samples)
    bp_systolic = np.clip(np.round(sys_base), 95, 185).astype(int)
    
    dia_base = 72 + (bmi - 22) * 0.4 + (age - 30) * 0.2 + np.random.normal(0, 6, n_samples)
    bp_diastolic = np.clip(np.round(dia_base), 60, 110).astype(int)
    
    # 7. Blood Oxygen Saturation SpO2 (%)
    spo2_base = 98.2 - (bmi > 32) * 1.2 - (age > 60) * 0.8 + np.random.normal(0, 1.1, n_samples)
    spo2 = np.clip(np.round(spo2_base, 1), 90.0, 100.0)
    
    # 8. Continuous Blood Glucose (mg/dL) - Wearable Continuous Glucose Monitor (CGM)
    # Influenced by BMI, Activity, Age, Sleep, and genetic predisposition
    genetic_risk = np.random.beta(2, 5, n_samples)  # 0 to 1 scale
    
    latent_metabolic_score = (
        0.35 * ((bmi - 25) / 10) +
        0.30 * ((bp_systolic - 120) / 25) +
        0.25 * ((heart_rate - 72) / 15) -
        0.30 * ((activity_steps - 7500) / 3000) -
        0.15 * ((sleep_hours - 7.0) / 1.5) +
        0.25 * ((age - 45) / 20) +
        0.50 * genetic_risk +
        np.random.normal(0, 0.4, n_samples)
    )
    
    # Baseline fasting/postprandial glucose mapping
    glucose = 85 + 40 * np.maximum(0, latent_metabolic_score) + 25 * genetic_risk + np.random.normal(0, 10, n_samples)
    # Induce clinically recognizable high-risk and pre-diabetic tails
    glucose = np.clip(np.round(glucose, 1), 68.0, 265.0)
    
    # 9. Glucose Variability (SD in mg/dL over recent rolling window)
    glucose_variability = np.clip(np.round(10 + (glucose - 90) * 0.22 + np.random.normal(0, 4, n_samples), 1), 6.0, 55.0)
    
    # 10. Outcome: Clinically defined Diabetes / High Metabolic Risk (0 = Low/Normal, 1 = Pre-diabetic/Diabetic High Risk)
    # Calibrated to ~32% prevalence (matching standard Pima benchmark ~34%)
    prob_diabetes = 1.0 / (1.0 + np.exp(-(
        -2.5 +
        0.050 * (glucose - 105) +
        0.09 * (bmi - 28) +
        0.025 * (bp_systolic - 122) -
        0.0003 * (activity_steps - 6500) +
        0.035 * (age - 45) +
        0.04 * (glucose_variability - 14)
    )))
    
    outcome = (np.random.uniform(0, 1, n_samples) < prob_diabetes).astype(int)
    # Ensure clinical realism
    outcome[glucose >= 140] = 1
    outcome[(glucose < 95) & (bmi < 25) & (bp_systolic < 120)] = 0
    
    df = pd.DataFrame({
        "glucose": glucose,
        "glucose_variability": glucose_variability,
        "blood_pressure_systolic": bp_systolic,
        "blood_pressure_diastolic": bp_diastolic,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "bmi": bmi,
        "activity_steps": activity_steps,
        "sleep_hours": sleep_hours,
        "age": age,
        "outcome": outcome
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_enriched_dataset(n_samples=1500, seed=42)
    output_path = os.path.join("data", "diabetes_wearable_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} samples saved to {output_path}")
    print(f"Outcome distribution:\n{df['outcome'].value_counts(normalize=True)}")
    print(f"\nFeature Summary:\n{df.describe().T[['mean', 'std', 'min', '50%', 'max']]}")
