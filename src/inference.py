import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .models import CLASSIFICATION_FEATURES


def train_full_logistic_model(df):
    X = df[CLASSIFICATION_FEATURES]
    y = df["cardio"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_scaled,
        y
    )

    return model, scaler


def predict_patient(
    df,
    age_years,
    height,
    weight,
    ap_hi,
    ap_lo,
    cholesterol,
    gluc,
    smoke,
    alco,
    active
):

    model, scaler = train_full_logistic_model(df)

    patient = pd.DataFrame([{
        "age_years": age_years,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }])

    patient = patient[CLASSIFICATION_FEATURES]

    patient_scaled = scaler.transform(patient)

    prediction = model.predict(patient_scaled)[0]

    probability = model.predict_proba(
        patient_scaled
    )[0][1]

    bmi = weight / ((height / 100) ** 2)

    return prediction, probability, bmi
