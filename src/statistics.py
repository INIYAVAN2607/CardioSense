import numpy as np
import pandas as pd
from scipy import stats


def descriptive_statistics(df, column):
    """Calculate descriptive statistics."""
    s = df[column].dropna()

    return {
        "Mean": s.mean(),
        "Median": s.median(),
        "Variance": s.var(),
        "Standard Deviation": s.std(),
        "Minimum": s.min(),
        "Q1": s.quantile(0.25),
        "Q3": s.quantile(0.75),
        "Maximum": s.max(),
        "Skewness": s.skew()
    }


def confidence_interval(df, column, confidence=0.95):
    """Calculate t-based confidence interval for the mean."""
    values = df[column].dropna()

    n = len(values)
    mean = values.mean()
    sem = stats.sem(values)

    interval = stats.t.interval(
        confidence,
        df=n - 1,
        loc=mean,
        scale=sem
    )

    return {
        "sample_mean": mean,
        "lower": interval[0],
        "upper": interval[1],
        "confidence": confidence
    }


def welch_t_test(df, column, target="cardio"):
    """Compare a numeric variable between two target groups."""
    group0 = df[df[target] == 0][column].dropna()
    group1 = df[df[target] == 1][column].dropna()

    statistic, p_value = stats.ttest_ind(
        group0,
        group1,
        equal_var=False
    )

    return statistic, p_value


def one_way_anova(df, value_column, group_column):
    """Perform one-way ANOVA."""
    groups = []

    for value in sorted(df[group_column].dropna().unique()):
        group = df[df[group_column] == value][value_column].dropna()

        if len(group) > 0:
            groups.append(group)

    statistic, p_value = stats.f_oneway(*groups)

    return statistic, p_value


def pearson_correlation(df, columns):
    return df[columns].corr(method="pearson")


def spearman_correlation(df, columns):
    return df[columns].corr(method="spearman")


def covariance_matrix(df, columns):
    return df[columns].cov()
