"""Data loading, feature engineering, and preprocessing."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

ACADEMIC_FILE = DATA_DIR / "student_academic_data_anonymized.csv"
SURVEY_FILE = DATA_DIR / "student_survey_data_anonymized.csv"

def load_data():
    """Load and combine the academic and survey tables by row alignment, matching the original notebook."""
    data1 = pd.read_csv(ACADEMIC_FILE)
    data2 = pd.read_csv(SURVEY_FILE)
    if len(data1) != len(data2):
        raise ValueError("Academic and survey datasets must contain the same number of students.")
    for col in data2.columns:
        if col != "Name":
            data1[col] = data2[col]
    data1["Name"] = data2["Name"]
    data = data1.copy()
    data.rename(columns={
        "Physics": "Physics_Score",
        "Chemistry": "Chemistry_Score",
        "Maths": "Maths_Score",
    }, inplace=True)
    arrears = [f"NO OF ARREARS S{i}" for i in range(1, 6)]
    credits = [f"CREDITS EARNED S{i}" for i in range(1, 6)]
    data["Total_Arrears"] = data[arrears].sum(axis=1)
    data["Total_Credits_Earned"] = data[credits].sum(axis=1)
    return data, arrears, credits

def encode_features(data):
    """Encode categorical features while leaving student identity columns unchanged."""
    encoded = data.copy()
    encoders = {}
    for col in encoded.columns:
        if col in {"Name", "Student ID", "University No"}:
            continue
        if encoded[col].dtype == "object":
            le = LabelEncoder()
            encoded[col] = le.fit_transform(encoded[col].astype(str))
            encoders[col] = le
    return encoded, encoders

def build_cluster_features(data):
    attendance = [c for c in data.columns if c.endswith("_Atten %")]
    assignment = [c for c in data.columns if c.endswith("_Assign")]
    university = [c for c in data.columns if c.startswith("UR_")]
    important = ["Physics_Score", "Chemistry_Score", "Maths_Score",
                 "Total_Arrears", "Total_Credits_Earned"]
    social = [
        "1. What is the highest level of education completed by your parents?",
        "3. Do your parents have any influence on your academic decisions?",
        "4. How often do you discuss your academic progress with your parents?",
        "5. Where do you reside?",
        "6. What is your current relationship status?",
        "7. Do you receive any scholarship from your college?",
        "8. Do you receive any financial assistance from the government?",
        "9. How do you manage your finances while attending college?",
        "10. Have you ever had to take a loan to finance your education?",
        "11. How often do you feel stressed or overwhelmed due to college-related pressures?",
        "12. Have you ever sought help from a counselor or mental health professional due to college-related stress?",
        "13. How would you rate your overall mental well-being?",
        "14. How important is social media in your life?",
        "15. Have you ever experienced or witnessed bullying or harassment on campus?",
        "16. Do you think your college does enough to promote diversity, equity, and inclusion?"
    ]
    selected = [c for c in important + attendance + assignment + university + social if c in data.columns]
    features = data[selected].copy()
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.mean(numeric_only=True))
    return features
