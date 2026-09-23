import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def histogram(df, column):
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        df[column].dropna(),
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    return fig


def boxplot(df, column, target=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    if target and target in df.columns:
        sns.boxplot(
            data=df,
            x=target,
            y=column,
            ax=ax
        )
    else:
        sns.boxplot(
            data=df,
            y=column,
            ax=ax
        )

    ax.set_title(f"Boxplot of {column}")

    return fig


def scatter_plot(df, x, y, hue=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        alpha=0.5,
        ax=ax
    )

    ax.set_title(f"{x} vs {y}")

    return fig


def correlation_heatmap(corr):
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    return fig


def normal_distribution_plot(df, column):
    values = df[column].dropna()

    mean = values.mean()
    std = values.std()

    x = np.linspace(
        values.min(),
        values.max(),
        300
    )

    y = (
        1 / (std * np.sqrt(2 * np.pi))
    ) * np.exp(
        -0.5 * ((x - mean) / std) ** 2
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        values,
        stat="density",
        kde=False,
        ax=ax
    )

    ax.plot(
        x,
        y,
        linewidth=2,
        label="Normal Curve"
    )

    ax.legend()
    ax.set_title(f"Normal Distribution Demonstration: {column}")

    return fig


def exponential_distribution_plot():
    x = np.linspace(0, 10, 300)

    lam = 0.5
    y = lam * np.exp(-lam * x)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, y, linewidth=2)

    ax.set_title("Exponential Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")

    return fig
