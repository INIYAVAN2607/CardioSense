import pandas as pd
import numpy as np


def load_data(file_path="cardio_train.csv"):
    """Load the cardiovascular disease dataset."""
    df = pd.read_csv(file_path, sep=";")
    return df


def prepare_data(df):
    """Create useful derived features."""
    data = df.copy()

    # Convert age from days to years
    data["age_years"] = data["age"] / 365.25

    # BMI
    data["height_m"] = data["height"] / 100
    data["bmi"] = data["weight"] / (data["height_m"] ** 2)

    # Pulse pressure
    data["bp_difference"] = data["ap_hi"] - data["ap_lo"]

    # Remove temporary column
    data.drop(columns=["height_m"], inplace=True)

    return data


def missing_values(df):
    """Return missing-value counts."""
    return df.isnull().sum()


def duplicate_count(df):
    """Return number of duplicate rows."""
    return df.duplicated().sum()


def remove_duplicates(df):
    """Remove duplicate records."""
    return df.drop_duplicates().copy()


def outlier_summary(df, columns):
    """Calculate IQR-based potential outliers."""
    results = []

    for column in columns:
        if column not in df.columns:
            continue

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = ((df[column] < lower) | (df[column] > upper)).sum()

        results.append({
            "Variable": column,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Potential Outliers": count
        })

    return pd.DataFrame(results)


def clean_data(df):
    """Basic cleaning pipeline."""
    data = df.copy()

    data = data.drop_duplicates()

    numeric_columns = data.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        data[column] = data[column].fillna(data[column].median())

    return data
