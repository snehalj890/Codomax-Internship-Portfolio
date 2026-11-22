"""
Shared Domain Feature Engineering for Module 06: Student Academic Risk & Performance Platform.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_NUMERIC_COLS = [
    "study_hours",
    "attendance_pct",
    "previous_gpa",
    "assignments_completed",
    "quiz_avg",
    "late_submissions",
]

RAW_CATEGORICAL_COLS = ["track"]

ENGINEERED_NUMERIC_COLS = [
    "study_efficiency",
    "engagement_index",
    "assessment_momentum",
    "risk_pressure_score",
]

ALL_MODEL_FEATURES = RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS + ENGINEERED_NUMERIC_COLS


def enrich_student_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes domain-specific academic indicators from raw student metrics."""
    out = df.copy()

    out["track"] = out["track"].fillna("AI & ML").astype(str)
    for col in RAW_NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    # GPA gained per unit of daily study investment
    out["study_efficiency"] = np.round(
        out["previous_gpa"] / (out["study_hours"] + 0.5), 4
    )

    # Weighted active participation across attendance and coursework completion
    out["engagement_index"] = np.round(
        0.55 * out["attendance_pct"] + 0.45 * (out["assignments_completed"] * 10.0),
        4,
    )

    # Combined exam/quiz readiness blended with GPA scaled to 100
    out["assessment_momentum"] = np.round(
        0.50 * out["quiz_avg"] + 0.50 * (out["previous_gpa"] * 10.0),
        4,
    )

    # Composite early-warning pressure indicator (higher means greater risk)
    out["risk_pressure_score"] = np.round(
        np.maximum(0.0, 80.0 - out["attendance_pct"]) * 0.45
        + np.maximum(0.0, 75.0 - out["quiz_avg"]) * 0.35
        + out["late_submissions"] * 4.5,
        4,
    )

    return out


class AcademicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for student academic feature engineering."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return enrich_student_features(pd.DataFrame(X).copy())