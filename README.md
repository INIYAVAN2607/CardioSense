# ❤️ CardioSense

### Cardiovascular Disease Analysis & Prediction Platform

CardioSense is an interactive **Streamlit-based cardiovascular health analytics and machine learning platform** designed to explore cardiovascular disease data, perform statistical analysis, visualize health patterns, and predict cardiovascular disease status using machine learning.

The platform brings multiple stages of the data science workflow into a single interactive application, from **data preprocessing and exploratory analysis to statistical inference, machine learning, clustering, and patient-level prediction**.

---

## 🚀 Features

### 📊 Dataset Explorer

* Explore the cardiovascular disease dataset.
* View dataset dimensions and attributes.
* Inspect patient health information.
* Understand the structure and distribution of the data.

### 🧹 Data Preprocessing

* Clean and prepare healthcare data for analysis.
* Handle data transformations required for statistical and machine learning operations.
* Prepare features for model training and prediction.

### 📈 Descriptive Statistics

Perform statistical analysis of important cardiovascular health parameters, including:

* Mean
* Median
* Standard deviation
* Minimum and maximum values
* Distribution analysis
* Other descriptive measures

### 🔍 Exploratory Data Analysis

Explore patterns and relationships within the dataset using interactive visualizations.

The analysis helps investigate relationships between:

* Age
* Gender
* Blood pressure
* Cholesterol
* Glucose
* Weight
* Height
* Lifestyle-related attributes
* Cardiovascular disease status

### 📉 Probability Distributions

Visualize probability distributions of selected health parameters to understand their statistical behavior and variation across the dataset.

### 🔗 Correlation Analysis

Analyze relationships between different numerical health attributes using correlation analysis and visualizations.

This helps identify variables that show stronger or weaker relationships with cardiovascular disease indicators.

### 🧪 Statistical Inference

Apply statistical techniques to investigate relationships and differences within the cardiovascular dataset.

### 📐 Linear Regression

Use linear regression to study relationships between selected health variables and understand how one variable changes with another.

### 🤖 Machine Learning

CardioSense includes machine learning functionality for cardiovascular disease classification.

The platform allows users to:

* Prepare machine learning data
* Train classification models
* Evaluate model performance
* Analyze prediction results
* Predict cardiovascular disease status

### 🔵 K-Means Clustering

Apply **K-Means clustering** to group patients based on similarities in their health characteristics.

This provides an unsupervised learning perspective on the dataset.

### 🎯 Patient Prediction

Enter patient health information and obtain a machine learning-based cardiovascular disease prediction.

The prediction interface allows users to provide relevant patient parameters and receive the model's predicted cardiovascular disease status.

### 📄 PDF Reports

CardioSense can generate reports containing analysis and prediction results for easier viewing and documentation.

---

## 🧠 Data Science Workflow

```text
                 ┌──────────────────────┐
                 │   Cardiovascular     │
                 │       Dataset        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Data Preprocessing   │
                 └──────────┬───────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ Exploratory Data Analysis  │
              └─────────────┬──────────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
      ┌─────────────┐ ┌────────────┐ ┌──────────────┐
      │ Statistics  │ │Correlation │ │Visualization │
      └──────┬──────┘ └─────┬──────┘ └──────────────┘
             │              │
             └──────────────┼──────────────┐
                            ▼              ▼
                   ┌──────────────┐ ┌──────────────┐
                   │ Regression   │ │  Clustering  │
                   └──────┬───────┘ └──────┬───────┘
                          │                │
                          └───────┬────────┘
                                  ▼
                         ┌──────────────────┐
                         │ Machine Learning │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Patient          │
                         │ Prediction       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ PDF Report       │
                         └──────────────────┘
```

---

## 🛠️ Technology Stack

| Technology       | Purpose                         |
| ---------------- | ------------------------------- |
| **Python 3.12**  | Core programming language       |
| **Streamlit**    | Interactive web application     |
| **Pandas**       | Data manipulation and analysis  |
| **NumPy**        | Numerical computation           |
| **Matplotlib**   | Data visualization              |
| **Seaborn**      | Statistical visualization       |
| **SciPy**        | Statistical analysis            |
| **Scikit-learn** | Machine learning and clustering |
| **ReportLab**    | PDF report generation           |

---

## 📂 Project Structure

```text
CardioSense/
│
├── app.py
├── cardio_train.csv
├── requirements.txt
│
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── models.py
    ├── inference.py
    ├── visualization.py
    └── statistics.py
```

### Main Files

**`app.py`**

The main Streamlit application that connects the different modules and provides the interactive user interface.

**`cardio_train.csv`**

The cardiovascular disease dataset used for analysis and machine learning.

**`src/preprocessing.py`**

Contains data preprocessing and preparation functions.

**`src/models.py`**

Contains machine learning model-related functionality.

**`src/inference.py`**

Handles statistical inference and related analytical operations.

**`src/visualization.py`**

Contains functions used for generating data visualizations.

**`src/statistics.py`**

Contains statistical analysis functions.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd CardioSense
```

---

### 2. Create a Virtual Environment

Python 3.12 is recommended for this project.

```bash
python -m venv venv
```

---

### 3. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Install ReportLab for PDF report generation:

```bash
pip install reportlab
```

---

## ▶️ Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

After starting the application, Streamlit will provide a local URL:

```text
http://localhost:8501
```

Open the URL in your web browser to access CardioSense.

---

## 📊 Application Modules

The application provides several analytical modules through the Streamlit interface:

```text
CardioSense
│
├── 🏠 Dashboard
│
├── 📁 Dataset Explorer
│
├── 🧹 Data Preprocessing
│
├── 📈 Descriptive Statistics
│
├── 🔍 Exploratory Data Analysis
│
├── 📊 Probability Distributions
│
├── 🔗 Correlation Analysis
│
├── 🧪 Statistical Inference
│
├── 📉 Linear Regression
│
├── 🤖 Machine Learning
│
├── 🔵 K-Means Clustering
│
├── 🎯 Patient Prediction
│
└── ℹ️ About
```

---

## 🎯 Objective

The main objective of CardioSense is to provide an interactive platform for studying cardiovascular disease data using **data science, statistics, visualization, and machine learning**.

Instead of performing each analysis separately using different tools, CardioSense combines these operations into a single application.

The project demonstrates how a complete data analytics workflow can be applied to a healthcare dataset, including:

```text
Data
  ↓
Preprocessing
  ↓
Analysis
  ↓
Visualization
  ↓
Statistical Inference
  ↓
Machine Learning
  ↓
Prediction
  ↓
Report Generation
```

---

## 🔬 Machine Learning Perspective

CardioSense uses machine learning techniques to analyze cardiovascular health data and perform disease-status classification.

The workflow consists of:

1. Loading the cardiovascular dataset.
2. Preprocessing the input data.
3. Selecting relevant features.
4. Preparing data for machine learning.
5. Training classification models.
6. Evaluating model predictions.
7. Accepting patient information through the application.
8. Generating a cardiovascular disease prediction.

The platform also includes **K-Means clustering** to explore groups of patients with similar characteristics without relying on predefined labels.

---

## 📄 Report Generation

CardioSense provides PDF report generation for selected analysis and prediction results.

Reports can be used to document:

* Patient prediction results
* Analysis outcomes
* Statistical information
* Relevant application results

---

## 🖥️ Interface

CardioSense is built using **Streamlit**, allowing users to interact with the analysis modules through a web-based interface without requiring them to write Python code.

The application provides an interactive environment for exploring the dataset, performing analysis, training models, and obtaining predictions.

---

## 🔮 Future Improvements

Potential future extensions include:

* Additional machine learning algorithms
* Hyperparameter optimization
* Model comparison dashboards
* Advanced feature selection
* Explainable AI techniques
* Interactive model performance visualization
* Improved patient report generation
* Model persistence and deployment
* Integration with additional healthcare datasets
* Cloud deployment

---

## ⚠️ Disclaimer

CardioSense is an **educational and analytical project** developed for exploring cardiovascular health data and machine learning techniques.

The predictions generated by this application **should not be considered a medical diagnosis** and should not replace consultation with a qualified healthcare professional.
