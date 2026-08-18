"""
Risk Utilities - Normalization, classification, and risk calculations.

Purpose:
  - Convert raw scores to 0-100 percentile risk scale
  - Classify risk levels
  - Clamp scores
  - Utility functions for fusion logic
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"


def setup_logging(name: str = "RiskUtils") -> logging.Logger:
    """Setup logging for the module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def percentile_risk(scores: pd.Series, method: str = "average") -> pd.Series:
    """
    Convert scores to 0-100 percentile risk scale.

    Args:
        scores: Series of raw anomaly/fraud scores
        method: Ranking method ('average', 'min', 'max', 'first', 'dense')

    Returns:
        Series of risk scores on 0-100 scale
    """
    risk = scores.rank(pct=True, method=method) * 100.0
    return risk.clip(0, 100).round(4)


def clamp_risk(score: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamp risk score to [min_val, max_val]."""
    return float(np.clip(score, min_val, max_val))


def classify_risk_level(score: float) -> str:
    """
    Classify risk score into risk level.

    Risk Levels:
      0-30     Low
      31-60    Medium
      61-80    High
      81-100   Critical
    """
    if score < 0 or score > 100:
        return "Unknown"
    elif score <= 30:
        return "Low"
    elif score <= 60:
        return "Medium"
    elif score <= 80:
        return "High"
    else:
        return "Critical"


def load_kaggle_data(path: Path | None = None) -> pd.DataFrame:
    """Load Kaggle provider dataset."""
    if path is None:
        path = BASE_DIR / "processed" / "PROVIDER_ML_READY_KAGGLE.csv"

    if not path.exists():
        raise FileNotFoundError(f"Kaggle dataset not found: {path}")

    df = pd.read_csv(path)
    return df


def safe_divide(numerator, denominator, fill_value: float = 0.0) -> pd.Series | np.ndarray:
    """Safely divide with fill value for division by zero."""
    numerator = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    denominator = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator

    result = numerator.divide(denominator).replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    return result


def normalize_fraud_proba_to_risk(proba: pd.Series) -> pd.Series:
    """Convert fraud probability [0-1] to risk [0-100]."""
    return (proba.clip(0, 1) * 100.0).round(4)


def normalize_score_to_risk(scores: pd.Series, use_percentile: bool = True) -> pd.Series:
    """
    Normalize scores to risk [0-100].

    Args:
        scores: Raw scores
        use_percentile: If True, use percentile ranking; else scale [0, max]

    Returns:
        Risk scores on 0-100 scale
    """
    if use_percentile:
        return percentile_risk(scores)
    else:
        max_score = scores.max()
        if max_score <= 0:
            return pd.Series([0.0] * len(scores), index=scores.index)
        return (scores / max_score * 100.0).clip(0, 100).round(4)


def validate_dataframe(
    df: pd.DataFrame, expected_columns: list, logger: logging.Logger | None = None
) -> bool:
    """
    Validate dataframe has required columns.

    Args:
        df: DataFrame to validate
        expected_columns: List of required column names
        logger: Optional logger for messages

    Returns:
        True if all columns present, False otherwise
    """
    missing = [col for col in expected_columns if col not in df.columns]

    if missing:
        msg = f"Missing columns: {missing}"
        if logger:
            logger.error(msg)
        else:
            print(f"ERROR: {msg}")
        return False

    return True


def apply_risk_level(df: pd.DataFrame, score_col: str = "Final_Risk_Score", output_col: str = "Risk_Level") -> pd.DataFrame:
    """Apply risk level classification to dataframe."""
    df[output_col] = df[score_col].apply(classify_risk_level)
    return df


def apply_priority_score(
    df: pd.DataFrame,
    risk_col: str = "Final_Risk_Score",
    reimbursement_col: str = "Total_Reimbursement",
    output_col: str = "Priority_Score",
) -> pd.DataFrame:
    """
    Calculate priority score as risk * log(1 + reimbursement).

    Higher risk + higher reimbursement = higher priority for investigation.

    Args:
        df: DataFrame with risk scores and reimbursement
        risk_col: Name of risk score column
        reimbursement_col: Name of reimbursement column
        output_col: Name of output priority score column

    Returns:
        DataFrame with priority scores added
    """
    if reimbursement_col not in df.columns:
        df[output_col] = df[risk_col].round(0).astype(int)
    else:
        df[output_col] = (
            df[risk_col] * np.log1p(df[reimbursement_col])
        ).round(0).astype(int)
        
    df["Priority_Rank"] = (
    df[output_col]
    .rank(method="dense", ascending=False)
    .astype(int)
)

    return df


def print_risk_distribution(df: pd.DataFrame, risk_col: str = "Final_Risk_Score") -> None:
    """Print distribution of risk scores."""
    print()
    print("Risk Score Distribution:")
    print(f"  Min:    {df[risk_col].min():.2f}")
    print(f"  Q1:     {df[risk_col].quantile(0.25):.2f}")
    print(f"  Median: {df[risk_col].median():.2f}")
    print(f"  Q3:     {df[risk_col].quantile(0.75):.2f}")
    print(f"  Max:    {df[risk_col].max():.2f}")
    print(f"  Mean:   {df[risk_col].mean():.2f}")
    print(f"  Std:    {df[risk_col].std():.2f}")


def print_risk_level_counts(df: pd.DataFrame, risk_level_col: str = "Risk_Level") -> None:
    """Print counts of providers in each risk level."""
    print()
    print("Providers by Risk Level:")
    counts = df[risk_level_col].value_counts().sort_index()
    for level, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"  {level:10s}: {count:5d} ({pct:5.1f}%)")


def print_top_providers(df: pd.DataFrame, n: int = 20, risk_col: str = "Final_Risk_Score") -> None:
    """Print top N highest-risk providers."""
    print()
    print(f"Top {n} Highest-Risk Providers:")
    print()

    top = df.nlargest(n, risk_col)[["Provider", risk_col, "Risk_Level"]].copy()

    for idx, row in top.iterrows():
        provider = row["Provider"]
        risk = row[risk_col]
        level = row["Risk_Level"]
        print(f"  {idx + 1:2d}. Provider {provider:10s} → Risk {risk:6.2f} [{level}]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = setup_logging()

    # Test normalize functions
    test_scores = pd.Series([10, 20, 30, 40, 50])
    risk = percentile_risk(test_scores)
    print(f"Test percentile_risk: {risk.values}")

    # Test risk classification
    test_risk_scores = [15, 45, 70, 90]
    for score in test_risk_scores:
        level = classify_risk_level(score)
        print(f"Risk {score} → {level}")
