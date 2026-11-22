# EduPulse AI — Student Academic Risk & Performance Platform

## Overview
EduPulse AI is a student academic risk and performance analytics platform developed as part of the Codomax AI Internship Portfolio (Module 06). The application applies supervised machine learning to student academic and engagement indicators to classify academic standing (High Performer, Steady, At Risk), support interactive what-if scenario testing, and assist with cohort-level triage.

## Theoretical & Machine Learning Foundation
- **Problem Formulation (Multi-Class Classification):** Student performance monitoring is framed as a supervised multi-class classification problem where each student profile is mapped to a discrete academic risk tier (High Performer, Steady, or At Risk) based on input features.
- **Data Preprocessing & Feature Pipelines:** Real-world academic data can contain missing values, varying scales, and mixed attributes. Using a scikit-learn Pipeline ensures consistent preprocessing across training and inference without data leakage.
- **Model Training, Selection & Evaluation:** Candidate classifiers are trained and evaluated on a holdout split using **Accuracy** and **Macro-F1** (which weights classes equally to avoid letting a majority class hide performance differences across risk tiers). Run details, class labels, split sizes, and evaluation scores are recorded in model_metadata.json.
- **What-If Simulation & Decision Support:** Interactive what-if simulation lets users adjust input features and observe how predicted classifications change to support proactive cohort triage.
- **Environment & Serialization Consistency:** Pickled scikit-learn pipelines (model.pkl) rely on compatible scikit-learn and Python versions across training and deployment. Pinning dependencies in equirements.txt (such as scikit-learn==1.6.1 and joblib) helps prevent version mismatch errors on Streamlit Community Cloud.

## Live App & Repository
- **Streamlit App:** https://codomax-internship-portfolio-miniprojectt.streamlit.app/
- **GitHub Repository:** https://github.com/snehalj890/Codomax-Internship-Portfolio

## Key Features
- **Academic Risk Prediction:** Classifies students into High Performer, Steady, or At Risk using model.pkl.
- **What-If Simulation:** Lets users vary inputs interactively and inspect prediction changes.
- **Cohort Triage & Analytics:** Provides cohort sample inspection (sample_cohort.csv), model metadata (model_metadata.json), and interactive plotly charts.

## Project File Structure
~~~text
Codomax Internship Portfolio/
├── app.py                  # Main Streamlit web application
├── features.py             # Feature preparation and input helpers
├── model.pkl               # Trained scikit-learn pipeline
├── model_metadata.json     # Model metrics, classes, and training metadata
├── sample_cohort.csv       # Sample student cohort data
├── requirements.txt        # Pinned Python dependencies
├── jp.gif                  # Animated UI asset
├── assets/                 # Visual assets (e.g., bg.jpg)
└── README.md               # Project documentation
~~~

## Local Setup & Execution
1. Clone the repository:
   ~~~bash
   git clone https://github.com/snehalj890/Codomax-Internship-Portfolio.git
   cd Codomax-Internship-Portfolio
   ~~~
2. Create and activate a virtual environment (Windows PowerShell):
   ~~~powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ~~~
3. Install dependencies:
   ~~~bash
   pip install -r requirements.txt
   ~~~
4. Run the Streamlit app:
   ~~~bash
   streamlit run app.py
   ~~~
