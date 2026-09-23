import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)

from src.preprocessing import (
    load_data,
    prepare_data,
    missing_values,
    duplicate_count,
    outlier_summary
)

from src.statistics import (
    descriptive_statistics,
    confidence_interval,
    welch_t_test,
    one_way_anova,
    pearson_correlation,
    spearman_correlation,
    covariance_matrix
)

from src.visualization import (
    histogram,
    boxplot,
    scatter_plot,
    correlation_heatmap,
    normal_distribution_plot,
    exponential_distribution_plot
)

from src.models import (
    train_knn,
    train_logistic_regression,
    train_linear_regression,
    kmeans_analysis,
    elbow_values
)

from src.inference import predict_patient


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="CardioSense",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f7fa;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def get_data():

    df = load_data("cardio_train.csv")

    df = prepare_data(df)

    return df


try:

    df = get_data()

except Exception as e:

    st.error("Unable to load cardio_train.csv")

    st.exception(e)

    st.stop()



# =========================================================
# PDF REPORT GENERATION
# =========================================================

def generate_patient_pdf(result):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=11,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15
    )

    story = []

    story.append(
        Paragraph(
            "CardioSense",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Cardiovascular Disease Analysis and Prediction System",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "Patient Prediction Report",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "1. Patient Details",
            heading_style
        )
    )

    cholesterol_label = {
        1: "Normal",
        2: "Above Normal",
        3: "Well Above Normal"
    }[result["cholesterol"]]

    glucose_label = {
        1: "Normal",
        2: "Above Normal",
        3: "Well Above Normal"
    }[result["gluc"]]

    patient_data = [
        ["Parameter", "Value"],
        ["Age", f"{result['age']:.0f} years"],
        ["Height", f"{result['height']:.0f} cm"],
        ["Weight", f"{result['weight']:.0f} kg"],
        ["Systolic BP", str(result["ap_hi"])],
        ["Diastolic BP", str(result["ap_lo"])],
        ["BMI", f"{result['bmi']:.2f}"],
        ["BP Difference", str(result["bp_difference"])],
        ["Cholesterol", cholesterol_label],
        ["Glucose", glucose_label],
        [
            "Smoking",
            "Yes" if result["smoke"] else "No"
        ],
        [
            "Alcohol Intake",
            "Yes" if result["alco"] else "No"
        ],
        [
            "Physically Active",
            "Yes" if result["active"] else "No"
        ]
    ]

    patient_table = Table(
        patient_data,
        colWidths=[220, 220]
    )

    patient_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E86C1")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ])
    )

    story.append(patient_table)

    # ---------------------------------------------------------
    # 2. PREDICTION RESULT
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "2. Prediction Result",
            heading_style
        )
    )

    prediction_text = (
        "Cardiovascular Disease"
        if result["prediction"] == 1
        else "No Cardiovascular Disease"
    )

    prediction_data = [
        ["Prediction Result", prediction_text],
        [
            "Disease Probability",
            f"{result['disease_probability']:.2f}%"
        ],
        [
            "No Disease Probability",
            f"{result['no_disease_probability']:.2f}%"
        ]
    ]

    prediction_table = Table(
        prediction_data,
        colWidths=[220, 220]
    )

    prediction_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F2F2")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8)
        ])
    )

    story.append(prediction_table)

    # ---------------------------------------------------------
    # 3. PROBABILITY
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "3. Probability",
            heading_style
        )
    )

    # ---------------------------------------------------------
    # DONUT GRAPH
    # ---------------------------------------------------------

    graph_buffer = BytesIO()

    fig, ax = plt.subplots(
        figsize=(3.4, 2.5)
    )

    values = [
        result["no_disease_probability"],
        result["disease_probability"]
    ]

    labels = [
        "No Disease",
        "Disease"
    ]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={
            "width": 0.42
        },
        pctdistance=0.78,
        textprops={
            "fontsize": 7
        }
    )

    # Reduce the percentage font size
    for autotext in autotexts:
        autotext.set_fontsize(7)

    # Reduce the label font size
    for text in texts:
        text.set_fontsize(7)

    # Center text
    ax.text(
        0,
        0.07,
        "Probability",
        ha="center",
        va="center",
        fontsize=8
    )

    ax.text(
        0,
        -0.07,
        "100%",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

    # Smaller graph title
    ax.set_title(
        "Percentage Distribution of Disease Probability",
        fontsize=9,
        pad=5
    )

    # Keep graph compact
    ax.axis("equal")

    fig.savefig(
        graph_buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    graph_buffer.seek(0)

    pdf_graph = Image(
        graph_buffer,
        width=250,
        height=185
    )

    # Graph appears immediately after "3. Probability"
    story.append(pdf_graph)

    # Small spacing after graph
    story.append(
        Spacer(1, 5)
    )

    # ---------------------------------------------------------
    # PROBABILITY VALUES
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            f"Disease: {result['disease_probability']:.2f}% "
            f"out of 100",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"No Disease: {result['no_disease_probability']:.2f}% "
            f"out of 100",
            normal_style
        )
    )

    # ---------------------------------------------------------
    # FOOTER
    # ---------------------------------------------------------

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Generated by CardioSense",
            subtitle_style
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()
# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("❤️ CardioSense")

st.sidebar.markdown(
    "### Cardiovascular Disease Analysis"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📁 Dataset Explorer",
        "🧹 Data Preprocessing",
        "📈 Descriptive Statistics",
        "🔍 Exploratory Data Analysis",
        "📊 Probability Distributions",
        "🔗 Correlation Analysis",
        "🧪 Statistical Inference",
        "📉 Linear Regression",
        "🤖 Machine Learning",
        "🔵 K-Means Clustering",
        "🎯 Patient Prediction",
        "ℹ️ About"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">❤️ CardioSense</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Cardiovascular Disease Data Science Analysis System'
        '</div>',
        unsafe_allow_html=True
    )

    total = len(df)

    disease_count = int(
        df["cardio"].sum()
    )

    disease_percentage = (
        disease_count / total * 100
    )

    avg_age = df["age_years"].mean()

    avg_bmi = df["bmi"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Patients",
        f"{total:,}"
    )

    c2.metric(
        "Disease Cases",
        f"{disease_count:,}"
    )

    c3.metric(
        "Disease Percentage",
        f"{disease_percentage:.2f}%"
    )

    c4.metric(
        "Average Age",
        f"{avg_age:.1f} years"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Cardiovascular Disease Distribution")

        counts = df["cardio"].value_counts()

        fig, ax = plt.subplots()

        ax.pie(
            counts.values,
            labels=["No Disease", "Disease"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title(
            "Cardio Target Distribution"
        )

        st.pyplot(fig)

    with col2:

        st.subheader("Average Blood Pressure")

        bp_data = pd.DataFrame({
            "Measure": [
                "Systolic",
                "Diastolic"
            ],
            "Average": [
                df["ap_hi"].mean(),
                df["ap_lo"].mean()
            ]
        })

        st.bar_chart(
            bp_data.set_index("Measure")
        )

    st.divider()

    st.subheader("Data Science Process")

    cols = st.columns(6)

    process = [
        "1. Define Goal",
        "2. Retrieve Data",
        "3. Prepare Data",
        "4. Explore Data",
        "5. Build Models",
        "6. Present Findings"
    ]

    for col, item in zip(cols, process):

        col.info(item)


# =========================================================
# DATASET EXPLORER
# =========================================================

elif page == "📁 Dataset Explorer":

    st.title("📁 Dataset Explorer")

    tab1, tab2, tab3 = st.tabs(
        [
            "Preview",
            "Information",
            "Summary"
        ]
    )

    # -----------------------------------------------------
    # PREVIEW
    # -----------------------------------------------------

    with tab1:

        st.subheader("Complete Dataset")

        st.write(
            f"Total records: {len(df):,}"
        )

        show_all = st.checkbox(
            "Show all 70,000 records"
        )

        if show_all:

            st.dataframe(
                df,
                use_container_width=True,
                height=600
            )

        else:

            n = st.slider(
                "Number of rows to display",
                5,
                100,
                10
            )

            st.dataframe(
                df.head(n),
                use_container_width=True,
                height=600
            )

        # -------------------------------------------------
        # DOWNLOAD DATASET
        # -------------------------------------------------

        st.download_button(
            label="⬇️ Download Complete Dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="cardiosense_dataset.csv",
            mime="text/csv"
        )

    # -----------------------------------------------------
    # INFORMATION
    # -----------------------------------------------------

    with tab2:

        st.subheader("Dataset Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing Values": [
                df[col].isnull().sum()
                for col in df.columns
            ],
            "Unique Values": [
                df[col].nunique()
                for col in df.columns
            ]
        })

        st.dataframe(
            info,
            use_container_width=True
        )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    with tab3:

        st.subheader("Statistical Summary")

        st.dataframe(
            df.describe().T,
            use_container_width=True
        )


# =========================================================
# PREPROCESSING
# =========================================================

elif page == "🧹 Data Preprocessing":

    st.title("🧹 Data Preprocessing")

    st.subheader("1. Missing Values")

    missing = missing_values(df)

    st.dataframe(
        missing.rename("Missing Values")
    )

    st.subheader("2. Duplicate Records")

    duplicates = duplicate_count(df)

    st.metric(
        "Number of duplicate rows",
        duplicates
    )

    st.subheader("3. Feature Engineering")

    st.write(
        "New features created:"
    )

    st.write(
        "- Age in years"
    )

    st.write(
        "- BMI = Weight / Height²"
    )

    st.write(
        "- Blood pressure difference"
    )

    st.dataframe(
        df[
            [
                "age_years",
                "bmi",
                "bp_difference"
            ]
        ].head(10)
    )

    st.subheader("4. Potential Outliers")

    numeric_features = [
        "age_years",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "bmi"
    ]

    outliers = outlier_summary(
        df,
        numeric_features
    )

    st.dataframe(
        outliers,
        use_container_width=True
    )


# =========================================================
# DESCRIPTIVE STATISTICS
# =========================================================

elif page == "📈 Descriptive Statistics":

    st.title("📈 Descriptive Statistics")

    numeric_columns = [
        "age_years",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "bmi"
    ]

    selected = st.selectbox(
        "Select variable",
        numeric_columns
    )

    results = descriptive_statistics(
        df,
        selected
    )

    result_df = pd.DataFrame(
        results.items(),
        columns=["Statistic", "Value"]
    )

    st.dataframe(
        result_df,
        use_container_width=True
    )

    st.subheader("Five-Number Summary")

    five_number = df[selected].describe()[
        [
            "min",
            "25%",
            "50%",
            "75%",
            "max"
        ]
    ]

    st.dataframe(
        five_number
    )


# =========================================================
# EDA
# =========================================================

elif page == "🔍 Exploratory Data Analysis":

    st.title("🔍 Exploratory Data Analysis")

    col = st.selectbox(
        "Select variable",
        [
            "age_years",
            "height",
            "weight",
            "bmi",
            "ap_hi",
            "ap_lo"
        ]
    )

    st.pyplot(
        histogram(df, col)
    )

    st.pyplot(
        boxplot(
            df,
            col,
            "cardio"
        )
    )

    st.subheader(
        "Relationship Between Age and BMI"
    )

    st.pyplot(
        scatter_plot(
            df.sample(
                min(5000, len(df)),
                random_state=42
            ),
            "age_years",
            "bmi",
            "cardio"
        )
    )


# =========================================================
# DISTRIBUTIONS
# =========================================================

elif page == "📊 Probability Distributions":

    st.title("📊 Probability Distributions")

    st.subheader(
        "Normal Distribution"
    )

    normal_col = st.selectbox(
        "Select variable",
        [
            "age_years",
            "height",
            "weight",
            "ap_hi",
            "ap_lo",
            "bmi"
        ]
    )

    st.pyplot(
        normal_distribution_plot(
            df,
            normal_col
        )
    )

    st.info(
        "The normal curve shown here is a demonstration "
        "for understanding the distribution concept."
    )

    st.subheader(
        "Exponential Distribution"
    )

    st.pyplot(
        exponential_distribution_plot()
    )

    st.info(
        "The exponential plot is a theoretical "
        "distribution demonstration and does not claim "
        "that cardiovascular variables follow an "
        "exponential distribution."
    )


# =========================================================
# CORRELATION
# =========================================================

elif page == "🔗 Correlation Analysis":

    st.title("🔗 Correlation Analysis")

    columns = [
        "age_years",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "bmi"
    ]

    st.subheader("Pearson Correlation")

    pearson = pearson_correlation(
        df,
        columns
    )

    st.pyplot(
        correlation_heatmap(
            pearson
        )
    )

    st.subheader("Spearman Correlation")

    spearman = spearman_correlation(
        df,
        columns
    )

    st.pyplot(
        correlation_heatmap(
            spearman
        )
    )

    st.subheader("Covariance Matrix")

    covariance = covariance_matrix(
        df,
        columns
    )

    st.dataframe(
        covariance,
        use_container_width=True
    )


# =========================================================
# STATISTICAL INFERENCE
# =========================================================

elif page == "🧪 Statistical Inference":

    st.title("🧪 Statistical Inference")

    st.subheader(
        "95% Confidence Interval"
    )

    ci_col = st.selectbox(
        "Select variable for confidence interval",
        [
            "age_years",
            "weight",
            "height",
            "ap_hi",
            "ap_lo",
            "bmi"
        ]
    )

    ci = confidence_interval(
        df,
        ci_col
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Sample Mean",
        f"{ci['sample_mean']:.3f}"
    )

    c2.metric(
        "Lower Limit",
        f"{ci['lower']:.3f}"
    )

    c3.metric(
        "Upper Limit",
        f"{ci['upper']:.3f}"
    )

    st.divider()

    st.subheader(
        "Welch Independent Samples t-Test"
    )

    t_stat, p_value = welch_t_test(
        df,
        "ap_hi"
    )

    st.write(
        f"t-statistic: **{t_stat:.4f}**"
    )

    st.write(
        f"p-value: **{p_value:.6f}**"
    )

    if p_value < 0.05:

        st.success(
            "At the 5% significance level, "
            "the test provides evidence of a difference "
            "in mean systolic blood pressure between "
            "the two cardio groups."
        )

    else:

        st.info(
            "At the 5% significance level, "
            "the test does not provide sufficient evidence "
            "of a difference."
        )

    st.divider()

    st.subheader(
        "One-Way ANOVA"
    )

    f_stat, anova_p = one_way_anova(
        df,
        "ap_hi",
        "cholesterol"
    )

    st.write(
        f"F-statistic: **{f_stat:.4f}**"
    )

    st.write(
        f"p-value: **{anova_p:.6f}**"
    )

    if anova_p < 0.05:

        st.success(
            "There is statistical evidence that "
            "at least one cholesterol group has a "
            "different mean systolic blood pressure."
        )

    else:

        st.info(
            "There is insufficient statistical evidence "
            "of a difference among the group means."
        )


# =========================================================
# LINEAR REGRESSION
# =========================================================

elif page == "📉 Linear Regression":

    st.title("📉 Linear Regression")

    st.write(
        "This model predicts systolic blood pressure "
        "(ap_hi) using selected numerical and categorical "
        "features."
    )

    model, X_test, y_test, predictions, results = (
        train_linear_regression(df)
    )

    st.subheader("Regression Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "MAE",
        f"{results['MAE']:.3f}"
    )

    c2.metric(
        "MSE",
        f"{results['MSE']:.3f}"
    )

    c3.metric(
        "RMSE",
        f"{results['RMSE']:.3f}"
    )

    c4.metric(
        "R²",
        f"{results['R2']:.3f}"
    )

    st.subheader(
        "Actual vs Predicted"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.scatter(
        y_test,
        predictions,
        alpha=0.4
    )

    minimum = min(
        y_test.min(),
        predictions.min()
    )

    maximum = max(
        y_test.max(),
        predictions.max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linewidth=2
    )

    ax.set_xlabel(
        "Actual Systolic BP"
    )

    ax.set_ylabel(
        "Predicted Systolic BP"
    )

    ax.set_title(
        "Actual vs Predicted Values"
    )

    st.pyplot(fig)

    st.subheader(
        "Regression Coefficients"
    )

    feature_names = [
        "age_years",
        "height",
        "weight",
        "ap_lo",
        "cholesterol",
        "gluc"
    ]

    coefficient_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_
    })

    st.dataframe(
        coefficient_df,
        use_container_width=True
    )


# =========================================================
# MACHINE LEARNING
# =========================================================

elif page == "🤖 Machine Learning":

    st.title("🤖 Machine Learning")

    st.write(
        "The target variable is cardiovascular disease "
        "status (cardio)."
    )

    tab1, tab2 = st.tabs(
        [
            "kNN",
            "Logistic Regression"
        ]
    )

    with tab1:

        st.subheader(
            "k-Nearest Neighbors (kNN)"
        )

        k = st.slider(
            "Number of neighbors (k)",
            min_value=1,
            max_value=15,
            value=5
        )

        model, scaler, results = train_knn(
            df,
            k
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{results['Accuracy']:.4f}"
        )

        c2.metric(
            "Precision",
            f"{results['Precision']:.4f}"
        )

        c3.metric(
            "Recall",
            f"{results['Recall']:.4f}"
        )

        c4.metric(
            "F1 Score",
            f"{results['F1 Score']:.4f}"
        )

        cm = results["Confusion Matrix"]

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "No Disease",
                "Disease"
            ],
            yticklabels=[
                "No Disease",
                "Disease"
            ],
            ax=ax
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        ax.set_title(
            "kNN Confusion Matrix"
        )

        st.pyplot(fig)

        st.text(
            results["Classification Report"]
        )

    with tab2:

        st.subheader(
            "Logistic Regression"
        )

        model, scaler, results = (
            train_logistic_regression(df)
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{results['Accuracy']:.4f}"
        )

        c2.metric(
            "Precision",
            f"{results['Precision']:.4f}"
        )

        c3.metric(
            "Recall",
            f"{results['Recall']:.4f}"
        )

        c4.metric(
            "F1 Score",
            f"{results['F1 Score']:.4f}"
        )

        cm = results["Confusion Matrix"]

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=[
                "No Disease",
                "Disease"
            ],
            yticklabels=[
                "No Disease",
                "Disease"
            ],
            ax=ax
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        ax.set_title(
            "Logistic Regression Confusion Matrix"
        )

        st.pyplot(fig)

        st.text(
            results["Classification Report"]
        )


# =========================================================
# K-MEANS
# =========================================================

elif page == "🔵 K-Means Clustering":

    st.title("🔵 K-Means Clustering")

    st.write(
        "K-Means is an unsupervised learning algorithm "
        "used to group similar observations."
    )

    max_k = st.slider(
        "Maximum K for Elbow Method",
        4,
        10,
        8
    )

    elbow = elbow_values(
        df,
        max_k
    )

    st.subheader(
        "Elbow Method"
    )

    fig, ax = plt.subplots()

    ax.plot(
        elbow["K"],
        elbow["Inertia"],
        marker="o"
    )

    ax.set_xlabel(
        "Number of Clusters (K)"
    )

    ax.set_ylabel(
        "Inertia"
    )

    ax.set_title(
        "Elbow Method"
    )

    st.pyplot(fig)

    k = st.slider(
        "Select number of clusters",
        2,
        max_k,
        3
    )

    model, scaler, cluster_data = (
        kmeans_analysis(
            df,
            k
        )
    )

    st.subheader(
        "Cluster Summary"
    )

    st.dataframe(
        cluster_data.groupby(
            "Cluster"
        ).mean(),
        use_container_width=True
    )

    st.subheader(
        "Age vs BMI Cluster Visualization"
    )

    sample = cluster_data.sample(
        min(5000, len(cluster_data)),
        random_state=42
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.scatterplot(
        data=sample,
        x="age_years",
        y="bmi",
        hue="Cluster",
        palette="viridis",
        alpha=0.6,
        ax=ax
    )

    ax.set_title(
        "K-Means Clusters"
    )

    st.pyplot(fig)


# =========================================================
# PATIENT PREDICTION
# =========================================================

elif page == "🎯 Patient Prediction":

    st.title("🎯 Individual Patient Prediction")

   

    # -----------------------------------------------------
    # PATIENT INPUT
    # -----------------------------------------------------

    st.subheader("👤 Patient Details")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=1.0,
            max_value=120.0,
            value=50.0,
            step=1.0
        )

        height = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=220.0,
            value=165.0,
            step=1.0
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=200.0,
            value=65.0,
            step=1.0
        )

        ap_hi = st.number_input(
            "Systolic Blood Pressure",
            min_value=50,
            max_value=250,
            value=120,
            step=1
        )

        ap_lo = st.number_input(
            "Diastolic Blood Pressure",
            min_value=30,
            max_value=150,
            value=80,
            step=1
        )

    with col2:

        cholesterol = st.selectbox(
            "Cholesterol",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[x]
        )

        gluc = st.selectbox(
            "Glucose",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[x]
        )

        smoke = st.selectbox(
            "Smoking",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

        alco = st.selectbox(
            "Alcohol Intake",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

        active = st.selectbox(
            "Physically Active",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

    # -----------------------------------------------------
    # PREDICTION BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔍 Predict Cardiovascular Disease",
        use_container_width=True
    ):

        bmi = weight / ((height / 100) ** 2)

        bp_difference = ap_hi - ap_lo

        prediction, probability, model_bmi = predict_patient(
            df,
            age,
            height,
            weight,
            ap_hi,
            ap_lo,
            cholesterol,
            gluc,
            smoke,
            alco,
            active
        )

        disease_probability = probability * 100

        no_disease_probability = 100 - disease_probability

        st.session_state["prediction_result"] = {
            "prediction": prediction,
            "disease_probability": disease_probability,
            "no_disease_probability": no_disease_probability,
            "bmi": bmi,
            "bp_difference": bp_difference,
            "age": age,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active
        }

        st.session_state["show_prediction_popup"] = True

    # -----------------------------------------------------
    # POPUP REPORT
    # -----------------------------------------------------

    if (
        "prediction_result" in st.session_state
        and st.session_state.get("show_prediction_popup", False)
    ):

        result = st.session_state["prediction_result"]

        # Dialog is used when supported by the installed
        # Streamlit version.
        if hasattr(st, "dialog"):

            @st.dialog("📋 Patient Prediction Report")
            def show_prediction_dialog():

                

                st.subheader("📊 Probability of Cardiovascular Disease")

                c1, c2 = st.columns(2)

                c1.metric(
                    "Disease",
                    f"{result['disease_probability']:.2f}%"
                )

                c2.metric(
                    "No Disease",
                    f"{result['no_disease_probability']:.2f}%"
                )

                # -----------------------------------------
                # DONUT GRAPH
                # -----------------------------------------

                fig, ax = plt.subplots(
                    figsize=(7, 5)
                )

                values = [
                    result["no_disease_probability"],
                    result["disease_probability"]
                ]

                labels = [
                    "No Disease",
                    "Disease"
                ]

                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    wedgeprops={
                        "width": 0.42
                    },
                    pctdistance=0.78
                )

                ax.text(
                    0,
                    0.08,
                    "Probability",
                    ha="center",
                    va="center",
                    fontsize=13
                )

                ax.text(
                    0,
                    -0.08,
                    "100%",
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold"
                )

                ax.set_title(
                    "Percentage Distribution of Disease Probability"
                )

                st.pyplot(fig)

               
               

                # -----------------------------------------
                # PATIENT DETAILS
                # -----------------------------------------

                st.subheader("👤 Patient Details")

                cholesterol_label = {
                    1: "Normal",
                    2: "Above Normal",
                    3: "Well Above Normal"
                }[result["cholesterol"]]

                glucose_label = {
                    1: "Normal",
                    2: "Above Normal",
                    3: "Well Above Normal"
                }[result["gluc"]]

                patient_details = pd.DataFrame({
                    "Parameter": [
                        "Age",
                        "Height",
                        "Weight",
                        "Systolic BP",
                        "Diastolic BP",
                        "BMI",
                        "BP Difference",
                        "Cholesterol",
                        "Glucose",
                        "Smoking",
                        "Alcohol Intake",
                        "Physically Active"
                    ],
                    "Value": [
                        f"{result['age']:.0f} years",
                        f"{result['height']:.0f} cm",
                        f"{result['weight']:.0f} kg",
                        result["ap_hi"],
                        result["ap_lo"],
                        f"{result['bmi']:.2f}",
                        result["bp_difference"],
                        cholesterol_label,
                        glucose_label,
                        "Yes" if result["smoke"] else "No",
                        "Yes" if result["alco"] else "No",
                        "Yes" if result["active"] else "No"
                    ]
                })

                st.dataframe(
                    patient_details,
                    use_container_width=True,
                    hide_index=True
                )

                # -----------------------------------------
                # PDF DOWNLOAD
                # -----------------------------------------

                pdf_report = generate_patient_pdf(result)

                st.download_button(
                    label="⬇️ Download Patient Report as PDF",
                    data=pdf_report,
                    file_name="cardiosense_patient_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            show_prediction_dialog()

        else:

            # Fallback for older Streamlit versions.
            st.warning(
                "Your installed Streamlit version does not support "
                "native popup dialogs. The report is shown below."
            )

            if result["prediction"] == 1:

                st.error(
                    "⚠️ Cardiovascular Disease Status = 1"
                )

            else:

                st.success(
                    "✅ Cardiovascular Disease Status = 0"
                )

            st.subheader("📊 Probability of Disease")

            c1, c2 = st.columns(2)

            c1.metric(
                "Disease",
                f"{result['disease_probability']:.2f}%"
            )

            c2.metric(
                "No Disease",
                f"{result['no_disease_probability']:.2f}%"
            )

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            values = [
                result["no_disease_probability"],
                result["disease_probability"]
            ]

            labels = [
                "No Disease",
                "Disease"
            ]

            ax.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={
                    "width": 0.42
                },
                pctdistance=0.78
            )

            ax.text(
                0,
                0.08,
                "Probability",
                ha="center",
                va="center",
                fontsize=13
            )

            ax.text(
                0,
                -0.08,
                "100%",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold"
            )

            ax.set_title(
                "Percentage Distribution of Disease Probability"
            )

            st.pyplot(fig)

            st.subheader("👤 Patient Details")

            cholesterol_label = {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[result["cholesterol"]]

            glucose_label = {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[result["gluc"]]

            patient_details = pd.DataFrame({
                "Parameter": [
                    "Age",
                    "Height",
                    "Weight",
                    "Systolic BP",
                    "Diastolic BP",
                    "BMI",
                    "BP Difference",
                    "Cholesterol",
                    "Glucose",
                    "Smoking",
                    "Alcohol Intake",
                    "Physically Active"
                ],
                "Value": [
                    f"{result['age']:.0f} years",
                    f"{result['height']:.0f} cm",
                    f"{result['weight']:.0f} kg",
                    result["ap_hi"],
                    result["ap_lo"],
                    f"{result['bmi']:.2f}",
                    result["bp_difference"],
                    cholesterol_label,
                    glucose_label,
                    "Yes" if result["smoke"] else "No",
                    "Yes" if result["alco"] else "No",
                    "Yes" if result["active"] else "No"
                ]
            })

            st.dataframe(
                patient_details,
                use_container_width=True,
                hide_index=True
            )

            pdf_report = generate_patient_pdf(result)

            st.download_button(
                label="⬇️ Download Patient Report as PDF",
                data=pdf_report,
                file_name="cardiosense_patient_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        st.session_state["show_prediction_popup"] = False




# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About CardioSense")

    st.write(
        """
        CardioSense is an academic Data Science project
        developed using the Cardiovascular Disease Dataset.

        The project demonstrates the complete Data Science
        workflow using Python.
        """
    )

    st.subheader(
        "Technologies Used"
    )

    st.write(
        "- Python"
    )

    st.write(
        "- Pandas"
    )

    st.write(
        "- NumPy"
    )

    st.write(
        "- Matplotlib"
    )

    st.write(
        "- Seaborn"
    )

    st.write(
        "- SciPy"
    )

    st.write(
        "- Scikit-learn"
    )

    st.write(
        "- Streamlit"
    )

    st.subheader(
        "Machine Learning Models"
    )

    st.write(
        "- Linear Regression"
    )

    st.write(
        "- Logistic Regression"
    )

    st.write(
        "- k-Nearest Neighbors"
    )

    st.write(
        "- K-Means Clustering"
    )

    st.subheader(
        "Academic Disclaimer"
    )

    st.info(
        "This application is developed for educational "
        "and academic purposes. The predictions should "
        "not be interpreted as medical diagnosis or "
        "clinical advice."
    )
