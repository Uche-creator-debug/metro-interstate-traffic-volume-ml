# 🚦 Metro Interstate Traffic Volume Prediction

An end-to-end Machine Learning project focused on predicting interstate traffic volume using weather conditions, temporal traffic patterns, and engineered cyclical features.

The project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Model development & tuning
- Explainable AI (SHAP)
- Streamlit deployment

---

# 📌 Project Overview

Traffic congestion is a major challenge in urban transportation systems.  
This project leverages historical interstate traffic and weather data to build a predictive machine learning system capable of estimating traffic volume under varying environmental and temporal conditions.

The final deployed application allows users to interactively predict traffic volume using:
- Time-based inputs
- Weather conditions
- Traffic-related features

---

# 📂 Dataset

Dataset: **Metro Interstate Traffic Volume Dataset**

The dataset contains:
- Weather conditions
- Temperature
- Rainfall
- Cloud coverage
- Holiday information
- Date & time
- Traffic volume observations

Target Variable:
- `traffic_volume`

---

# ⚙️ Project Workflow

## 1️⃣ Exploratory Data Analysis (EDA)

The dataset was thoroughly explored to:
- Understand feature distributions
- Detect anomalies and outliers
- Identify temporal traffic patterns
- Analyze relationships between traffic and weather variables

### Key Findings
- Traffic volume follows strong cyclical daily patterns
- Weekdays exhibit significantly higher traffic than weekends
- Weather variables have weaker influence compared to temporal features
- Several anomalies and duplicate rows were identified and handled

---

## 2️⃣ Data Preprocessing

Preprocessing steps included:
- Duplicate removal
- Handling suspicious observations
- Missing value treatment
- Datetime conversion
- Data cleaning and formatting

### Missing Values
The `holiday` column contained numerous missing values representing the absence of holidays rather than missing information.  
These values were imputed using:
```python
"NoHoliday"
```

---

## 3️⃣ Feature Engineering

Extensive feature engineering was performed to improve model learning.

### Engineered Features
- `hour`
- `month`
- `weekday`
- `is_weekend`
- `rush_hour`
- `season`

### Cyclical Encoding
To properly represent cyclical temporal behavior, sine and cosine transformations were applied:

```math
hour_{sin} = \sin(2\pi \cdot hour / 24)
```

```math
hour_{cos} = \cos(2\pi \cdot hour / 24)
```

This significantly improved model performance by preserving temporal cyclicity.

---

# 🤖 Machine Learning Models

The following baseline models were trained and evaluated:

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 713.72 | 924.79 | 0.7833 |
| Random Forest | 267.29 | 456.72 | 0.9471 |
| XGBoost | 269.64 | 444.07 | 0.9500 |

---

# 🏆 Final Model

The final deployed model was:
- **XGBoost Regressor**
- Tuned using **Optuna**

### Final Test Performance

| Metric | Score |
|---|---|
| MAE | 262.69 |
| RMSE | 478.28 |
| R² Score | 0.9412 |

The slight reduction from validation performance suggests strong generalization capability and minimal overfitting.

---

# 📊 Model Diagnostics

## Residual Analysis
Residual plots showed:
- Predictions were generally unbiased
- Residuals were randomly distributed around zero
- Slight heteroscedasticity at higher traffic volumes

This indicates:
- strong predictive capability
- reduced consistency under extreme traffic conditions

---

# 🔍 Explainable AI (SHAP)

SHAP analysis revealed that:
- Temporal cyclical features (`hour_cos`, `hour_sin`) were the most influential predictors
- Weekend behavior strongly impacted traffic patterns
- Weather-related variables contributed comparatively less

This confirms that commuter behavior and temporal patterns dominate interstate traffic dynamics.

---

# 🚀 Streamlit Application

An interactive Streamlit dashboard was developed for real-time traffic prediction.

## Features
- Interactive sidebar controls
- Real-time traffic prediction
- Traffic intensity gauge visualization
- Traffic category interpretation
- User-friendly interface

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Optuna
- SHAP
- Streamlit
- Plotly
- Matplotlib
- Seaborn

---

# 📁 Project Structure

```bash
metro-interstate-traffic-volume-ml/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv
│
├── model/
│   ├── xgb_model.joblib
│   └── xgb_model_columns.joblib
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Modeling.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── __init__.py
│
├── requirements.txt
│
└── README.md
```

---

# ▶️ Running the Project

## Clone Repository

```bash
git clone <your-repository-link>
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

# 📈 Future Improvements

Potential future enhancements include:
- Time-series cross validation
- Deep learning architectures
- Real-time traffic API integration
- Advanced ensemble methods
- Live deployment and monitoring

---

# 🙌 Acknowledgements

Dataset source:
- UCI Machine Learning Repository

Libraries and tools:
- Scikit-learn
- XGBoost
- Streamlit
- SHAP
- Optuna

---

# 📬 Contact

If you found this project interesting or would like to collaborate, feel free to connect.

---