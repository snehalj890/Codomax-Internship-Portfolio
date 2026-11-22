# EduPulse AI — Student Academic Risk & Performance Platform

## Overview
EduPulse AI is a student academic risk and performance analytics platform developed as part of the Codomax AI Internship Portfolio (Module 06)[cite: 1, 2, 16]. The application applies supervised machine learning to student academic and engagement indicators to classify academic standing (`High Performer`, `Steady`, `At Risk`), support interactive what-if scenario testing, and assist with cohort-level triage[cite: 1, 2, 8].

## Theoretical & Machine Learning Foundation
- **Problem Formulation (Multi-Class Classification):** Student performance monitoring is framed as a supervised multi-class classification problem where each student profile is mapped to a discrete academic risk tier (`High Performer`, `Steady`, or `At Risk`) based on historical/input features[cite: 8].
- **Data Preprocessing & Feature Pipelines:** Real-world academic datasets often contain missing values, mixed feature scales, and categorical/numerical attributes. Using a `scikit-learn` `Pipeline` (with imputation and feature transformation steps) ensures that identical preprocessing logic is applied consistently during both model training and live inference without data leakage.
- **Model Training, Selection & Evaluation:** Candidate classification models are trained and evaluated on held-out test data using metrics such as **Accuracy** and **Macro-F1** (Macro-F1 is especially helpful in multi-class academic risk settings because it weights classes equally rather than letting a majority class mask underperformance on minority risk tiers)[cite: 8]. Metadata from training runs—including selected model details, class labels, dataset splits, and evaluation scores—is stored in `model_metadata.json`[cite: 8].
- **What-If Simulation & Decision Support:** Rather than acting as a static score lookup, interactive what-if simulation lets educators or advisors vary input features and observe how predicted risk classifications respond, supporting proactive intervention and cohort triage[cite: 1].
- **Environment & Serialization Consistency:** Serialized `scikit-learn` estimators/pipelines (`model.pkl`) depend on compatible `scikit-learn` and Python runtime versions between training and deployment[cite: 4, 11, 17]. Pinning dependencies in `requirements.txt` (such as `scikit-learn==1.6.1` and `joblib`) helps prevent deserialization or attribute mismatches across local and cloud environments[cite: 4, 11, 17].

## Live App & Repository
- **Streamlit App:** https://codomax-internship-portfolio-miniprojectt.streamlit.app/[cite: 6, 17]
- **GitHub Repository:** https://github.com/snehalj890/Codomax-Internship-Portfolio[cite: 16]

## Key Features
- **Academic Risk Prediction:** Predicts `High Performer`, `Steady`, or `At Risk` outcomes using the trained pipeline in `model.pkl`[cite: 8, 17].
- **What-If Simulation:** Enables interactive parameter adjustments to evaluate sensitivity and potential risk shifts[cite: 1].
- **Cohort Triage & Analytics:** Displays cohort samples (`sample_cohort.csv`), model evaluation metadata (`model_metadata.json`), and interactive `plotly` visualizations[cite: 8, 17].

## Project File Structure
```text
Codomax-Internship-Portfolio/
├── app.py                  # Main Streamlit web application, UI, simulation, and analytics[cite: 16, 17]
├── features.py             # Feature definitions, validation, and preprocessing helpers[cite: 17]
├── model.pkl               # Serialized scikit-learn classification pipeline[cite: 17]
├── model_metadata.json     # Training timestamp, class labels, split sizes, and metrics[cite: 8, 17]
├── sample_cohort.csv       # Sample student dataset for cohort inspection and testing[cite: 17]
├── requirements.txt        # Pinned Python packages for local & Streamlit Cloud runs[cite: 17]
├── jp.gif                  # Animated visual asset used in the interface[cite: 17]
├── assets/                 # Static visual assets (such as background imagery)[cite: 16, 17]
└── README.md               # Project documentation, theory, architecture, and setup guide
