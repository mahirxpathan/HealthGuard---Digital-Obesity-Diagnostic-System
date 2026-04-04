# 🏥 Obesity Level Prediction & Diagnostic Dashboard
**Researcher & Developer: Pathan MohammadMahir I.**

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://healthguard-digital-obesity-diagnostic-system.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?style=for-the-badge)](https://www.python.org/downloads/release/python-3120/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge)](https://scikit-learn.org/)

## 📝 Project Abstract
This repository contains a complete clinical data science pipeline for predicting obesity levels in individuals from **Mexico, Peru, and Colombia**. The project leverages a research dataset (Palechor & Manotas, 2019) to classify individuals into 7 distinct weight categories using advanced Machine Learning algorithms.

The final deliverable is a **Professional Multi-Page Streamlit Dashboard** that provides real-time health assessments with over 95% accuracy.

---

## 🔗 Live Demonstration
**Access the live diagnostic dashboard here:**  
👉 [HealthGuard AI Dashboard](https://healthguard-digital-obesity-diagnostic-system.streamlit.app/)

---

## 🎯 Target Health Categories
- **Insufficient Weight**
- **Normal Weight**
- **Overweight Level I & II**
- **Obesity Type I, II & III**

---

## 🏗️ Technical Architecture & Structure
The project is organized into a modular structure for easy reproduction and deployment:

```
obesityPrediction/
├── app.py                         # Professional Streamlit Dashboard (Main Entry)
├── ObesityLevelPrediction.ipynb   # Comprehensive Research & Training Notebook
├── ObesityDataSet.csv             # Clinical Research Dataset (2111 Records)
├── requirements.txt               # Project Dependencies for Deployment
├── rf_model.pkl                   # Optimized Random Forest Artifact
├── xgb_model.pkl                  # High-Performance XGBoost Artifact
├── target_encoder.pkl             # Label Encoding Mapping
├── scaler.pkl                     # Standardized Feature Scaler
└── LICENSE                        # MIT Project License
```

---

## 🚀 Deployment & Usage Guide

### 1. Launching the Interactive Dashboard
To run the diagnostic dashboard locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

### 2. Exploring the Research (Jupyter)
To explore the Exploratory Data Analysis (EDA) and Model Training:
```bash
# Launch the notebook
jupyter notebook ObesityLevelPrediction.ipynb
```

---

## 🔥 Final Model Performance
Our models achieved state-of-the-art performance on the validation dataset:

| Model Performance Metric | Random Forest (Bagging) | XGBoost (Boosting) |
| :--- | :---: | :---: |
| **Accuracy Score** | **95.27%** | **95.27%** |
| **Recall (Weighted)** | **95.27%** | **95.27%** |
| **F1-Score (Macro)** | **95.12%** | **95.18%** |
| **Precision (Weighted)** | **95.30%** | **95.34%** |

### 📋 Scientific Findings
*   **Feature Importance**: Weight, Height, and Age are the three "Anchor Features" for the AI models.
*   **Lifestyle Impact**: Frequent consumption of high-caloric food (FAVC) and lack of physical activity (FAF) show direct correlations with progressing obesity tiers.
*   **Balance & Reliability**: The integration of **SMOTE** oversampling ensured that the AI is equally sensitive to all 7 obesity stages, avoiding common "Normal Weight" bias.

---

## 🛠️ Technology Stack
- **Dashboard**: Streamlit (Multi-page Architecture)
- **Data Engineering**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Intelligence**: Scikit-Learn (Random Forest), XGBoost
- **Serialization**: Joblib

---

## 📜 Acknowledgements & License
- **Dataset**: UCI Machine Learning Repository (Obesity Levels).
- **License**: [MIT License](LICENSE)

**Developed with ❤️ by Mahiri.**
