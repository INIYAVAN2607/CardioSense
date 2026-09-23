import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


CLASSIFICATION_FEATURES = [
    "age_years",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
]


def classification_data(df):
    X = df[CLASSIFICATION_FEATURES].copy()
    y = df["cardio"].copy()

    return X, y


def train_test_data(df, test_size=0.2):
    X, y = classification_data(df)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )


def evaluate_classifier(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Confusion Matrix": confusion_matrix(
            y_true,
            y_pred
        ),
        "Classification Report": classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    }


def train_knn(df, k=5):
    X_train, X_test, y_train, y_test = train_test_data(df)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(X_test_scaled)

    results = evaluate_classifier(
        y_test,
        predictions
    )

    return model, scaler, results


def train_logistic_regression(df):
    X_train, X_test, y_train, y_test = train_test_data(df)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(X_test_scaled)

    results = evaluate_classifier(
        y_test,
        predictions
    )

    return model, scaler, results


def train_linear_regression(df):
    """
    Predict systolic blood pressure (ap_hi)
    from selected demographic/health variables.
    """

    features = [
        "age_years",
        "height",
        "weight",
        "ap_lo",
        "cholesterol",
        "gluc"
    ]

    data = df[features + ["ap_hi"]].dropna()

    X = data[features]
    y = data["ap_hi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions
    )

    results = {
        "MAE": mean_absolute_error(
            y_test,
            predictions
        ),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(
            y_test,
            predictions
        )
    }

    return model, X_test, y_test, predictions, results


def kmeans_analysis(df, k=3):
    features = [
        "age_years",
        "weight",
        "bmi",
        "ap_hi",
        "ap_lo"
    ]

    data = df[features].dropna().copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(data)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X_scaled)

    data["Cluster"] = clusters

    return model, scaler, data


def elbow_values(df, max_k=8):
    features = [
        "age_years",
        "weight",
        "bmi",
        "ap_hi",
        "ap_lo"
    ]

    data = df[features].dropna()

    scaler = StandardScaler()

    X = scaler.fit_transform(data)

    values = []

    for k in range(1, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X)

        values.append({
            "K": k,
            "Inertia": model.inertia_
        })

    return pd.DataFrame(values)
