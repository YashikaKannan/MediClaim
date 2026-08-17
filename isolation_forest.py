#!/usr/bin/env python3
"""Production-ready Isolation Forest pipeline for Medicare carrier claim anomaly detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, RobustScaler


RANDOM_STATE = 42
CONTAMINATION_LEVELS = [0.01, 0.03, 0.05]
DATA_FILE = "carrier_clean_final_perfect.csv"

REQUIRED_COLUMNS = [
    "CLM_ID",
    "BENE_ID",
    "PRVDR_SPCLTY",
    "CLM_PMT_AMT",
    "NCH_CARR_CLM_SBMTD_CHRG_AMT",
    "NCH_CARR_CLM_ALOWD_AMT",
    "LINE_SBMTD_CHRG_AMT",
    "LINE_ALOWD_CHRG_AMT",
    "LINE_SRVC_CNT",
    "DIAGNOSIS_COUNT",
    "LINE_NCH_PMT_AMT",
]

NUMERIC_COLUMNS = [
    "CLM_PMT_AMT",
    "NCH_CARR_CLM_SBMTD_CHRG_AMT",
    "NCH_CARR_CLM_ALOWD_AMT",
    "LINE_SBMTD_CHRG_AMT",
    "LINE_ALOWD_CHRG_AMT",
    "LINE_SRVC_CNT",
    "DIAGNOSIS_COUNT",
    "LINE_NCH_PMT_AMT",
]

LOG_TRANSFORM_COLUMNS = [
    "CLM_PMT_AMT",
    "NCH_CARR_CLM_SBMTD_CHRG_AMT",
    "NCH_CARR_CLM_ALOWD_AMT",
    "LINE_SBMTD_CHRG_AMT",
    "LINE_ALOWD_CHRG_AMT",
]

MODEL_FEATURES = [
    "CLM_PMT_AMT",
    "NCH_CARR_CLM_SBMTD_CHRG_AMT",
    "NCH_CARR_CLM_ALOWD_AMT",
    "LINE_SBMTD_CHRG_AMT",
    "LINE_ALOWD_CHRG_AMT",
    "LINE_SRVC_CNT",
    "DIAGNOSIS_COUNT",
    "CHARGE_ALLOWED_RATIO",
    "CHARGE_PAYMENT_RATIO",
    "SERVICE_INTENSITY",
    "PEER_BILLING_RATIO",
    "EXCESS_BILLING_RATIO",
]


def setup_logging() -> None:
    """Configure application-wide logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directories(base_dir: Path) -> Dict[str, Path]:
    """Create output directories used for models, reports, and plots."""
    outputs_dir = base_dir / "outputs"
    models_dir = base_dir / "models"
    plots_dir = base_dir / "plots"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Ensured output directories exist.")
    return {"outputs": outputs_dir, "models": models_dir, "plots": plots_dir}


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the CSV dataset and print a quick diagnostic summary."""
    logging.info("Loading dataset from: %s", data_path)
    df = pd.read_csv(data_path, low_memory=False)

    logging.info("Dataset loaded successfully.")
    logging.info("Shape: %s", df.shape)
    logging.info("Columns (%d): %s", len(df.columns), list(df.columns))

    missing_values = df.isna().sum()
    logging.info("Missing values by column:\n%s", missing_values.to_string())

    basic_stats = df.describe(include="all").transpose()
    logging.info("Basic statistics:\n%s", basic_stats.to_string())

    return df


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Verify required columns exist and fail fast when critical data is missing."""
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logging.error("Missing required columns: %s", missing_columns)
        raise ValueError(
            "Critical required columns are missing. "
            f"Cannot continue pipeline. Missing: {missing_columns}"
        )

    logging.info("All required columns are present.")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two series and replace non-finite outputs with zero."""
    result = numerator.astype("float64") / denominator.astype("float64")
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and impute core fields used by fraud features and model inputs."""
    logging.info("Starting data cleaning.")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    df["PRVDR_SPCLTY"] = df["PRVDR_SPCLTY"].fillna("UNKNOWN")

    logging.info("Data cleaning complete.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create fraud-oriented engineered features from claims and line-level amounts."""
    logging.info("Starting feature engineering.")

    # Measures how aggressively a line item is billed versus what is allowed.
    df["CHARGE_ALLOWED_RATIO"] = safe_divide(
        df["LINE_SBMTD_CHRG_AMT"], df["LINE_ALOWD_CHRG_AMT"] + 1.0
    )

    # Measures billed amount versus paid amount for potential overcharging pressure.
    df["CHARGE_PAYMENT_RATIO"] = safe_divide(
        df["LINE_SBMTD_CHRG_AMT"], df["LINE_NCH_PMT_AMT"] + 1.0
    )

    # Measures service volume relative to diagnosis complexity.
    df["SERVICE_INTENSITY"] = safe_divide(
        df["LINE_SRVC_CNT"], df["DIAGNOSIS_COUNT"] + 1.0
    )

    # Measures claim-level billed excess over allowed claim-level amount.
    df["EXCESS_BILLING_RATIO"] = safe_divide(
        df["NCH_CARR_CLM_SBMTD_CHRG_AMT"], df["NCH_CARR_CLM_ALOWD_AMT"] + 1.0
    )

    logging.info("Feature engineering complete.")
    return df


def add_peer_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize billing behavior against specialty peer medians."""
    logging.info("Starting peer normalization by provider specialty.")

    specialty_median = df.groupby("PRVDR_SPCLTY")["LINE_SBMTD_CHRG_AMT"].transform("median")
    specialty_median = specialty_median.replace(0, np.nan)

    # Compares each provider line charge to the typical charge in the same specialty.
    df["PEER_BILLING_RATIO"] = safe_divide(df["LINE_SBMTD_CHRG_AMT"], specialty_median)

    logging.info("Peer normalization complete.")
    return df


def apply_log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply log1p transform to reduce skew in high-range monetary variables."""
    logging.info("Applying log1p transformation on monetary columns.")

    for col in columns:
        df[col] = np.log1p(np.clip(df[col], a_min=0, a_max=None))

    logging.info("Log transformation complete.")
    return df


def prepare_model_matrix(df: pd.DataFrame, model_features: List[str]) -> Tuple[np.ndarray, RobustScaler]:
    """Build scaled feature matrix for Isolation Forest training and scoring."""
    logging.info("Preparing model feature matrix and applying RobustScaler.")

    missing_model_features = [col for col in model_features if col not in df.columns]
    if missing_model_features:
        raise ValueError(f"Missing model features after engineering: {missing_model_features}")

    scaler = RobustScaler()
    X = scaler.fit_transform(df[model_features])

    logging.info("Feature matrix prepared. Shape: %s", X.shape)
    return X, scaler


def train_isolation_forest(X: np.ndarray, contamination: float) -> IsolationForest:
    """Train Isolation Forest model with production-friendly settings."""
    logging.info("Training Isolation Forest with contamination=%s", contamination)

    model = IsolationForest(
        n_estimators=300,
        max_samples=100000,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)

    logging.info("Model training complete for contamination=%s", contamination)
    return model


def score_claims(model: IsolationForest, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate anomaly labels, decision scores, and normalized risk scores."""
    logging.info("Scoring claims with trained model.")

    predictions = model.predict(X)
    decision_scores = model.decision_function(X)

    raw_score = -decision_scores.reshape(-1, 1)
    risk_scaler = MinMaxScaler(feature_range=(0, 100))
    risk_scores = risk_scaler.fit_transform(raw_score).ravel()

    logging.info("Scoring complete.")
    return predictions, decision_scores, risk_scores


def assign_risk_category(risk_scores: np.ndarray) -> pd.Series:
    """Bucket risk scores into analyst-friendly categories."""
    bins = [-0.001, 25, 50, 75, 100.0]
    labels = ["Low", "Medium", "High", "Critical"]

    categories = pd.cut(
        risk_scores,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    return categories.astype(str)


def save_model(model: IsolationForest, model_path: Path) -> None:
    """Persist trained model artifact to disk."""
    joblib.dump(model, model_path)
    logging.info("Saved model to: %s", model_path)


def build_result_frame(
    df: pd.DataFrame,
    predictions: np.ndarray,
    risk_scores: np.ndarray,
) -> pd.DataFrame:
    """Build result frame with anomaly outputs for downstream triage."""
    result_df = pd.DataFrame(
        {
            "CLM_ID": df["CLM_ID"],
            "BENE_ID": df["BENE_ID"],
            "PRVDR_SPCLTY": df["PRVDR_SPCLTY"],
            "RISK_SCORE": risk_scores,
            "RISK_CATEGORY": assign_risk_category(risk_scores),
            "ANOMALY_FLAG": predictions,
        }
    )

    return result_df.sort_values("RISK_SCORE", ascending=False)


def save_suspicious_claims(result_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Save suspicious claims only, sorted by descending risk score."""
    suspicious_df = result_df[result_df["ANOMALY_FLAG"] == -1].copy()
    suspicious_df.to_csv(output_path, index=False)

    logging.info("Saved suspicious claims to: %s | rows=%d", output_path, len(suspicious_df))
    return suspicious_df


def summarize_results(result_df: pd.DataFrame) -> Dict[str, float]:
    """Compute run-level KPI summary for model governance."""
    total_claims = len(result_df)
    suspicious_claims = int((result_df["ANOMALY_FLAG"] == -1).sum())
    normal_claims = total_claims - suspicious_claims

    suspicious_pct = (suspicious_claims / total_claims) * 100 if total_claims else 0.0

    return {
        "Total Claims": total_claims,
        "Normal Claims": normal_claims,
        "Suspicious Claims": suspicious_claims,
        "Suspicious Percentage": suspicious_pct,
        "Average Risk Score": float(result_df["RISK_SCORE"].mean()),
        "Maximum Risk Score": float(result_df["RISK_SCORE"].max()),
    }


def print_summary(contamination: float, summary: Dict[str, float]) -> None:
    """Print model summary in the required reporting format."""
    print("\n" + "=" * 72)
    print(f"Summary Report | contamination={contamination:.2%}")
    print("=" * 72)
    print(f"Total Claims         : {summary['Total Claims']:,}")
    print(f"Normal Claims        : {summary['Normal Claims']:,}")
    print(f"Suspicious Claims    : {summary['Suspicious Claims']:,}")
    print(f"Suspicious Percentage: {summary['Suspicious Percentage']:.2f}%")
    print(f"Average Risk Score   : {summary['Average Risk Score']:.2f}")
    print(f"Maximum Risk Score   : {summary['Maximum Risk Score']:.2f}")


def create_visualizations(
    contamination_results: Dict[float, pd.DataFrame],
    output_plots_dir: Path,
) -> None:
    """Generate and save risk analytics visualizations."""
    logging.info("Creating visualizations.")

    sns.set_style("whitegrid")

    # A) Risk score distribution histogram across contamination experiments.
    plt.figure(figsize=(11, 6))
    for contamination, result_df in contamination_results.items():
        sns.histplot(
            result_df["RISK_SCORE"],
            bins=50,
            stat="density",
            alpha=0.25,
            label=f"contamination={contamination:.0%}",
        )
    plt.title("Risk Score Distribution")
    plt.xlabel("Risk Score (0-100)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plots_dir / "risk_distribution.png", dpi=300)
    plt.close()

    # B) Top 20 highest-risk suspicious claims (using 3% contamination reference model).
    reference_contamination = 0.03 if 0.03 in contamination_results else list(contamination_results.keys())[0]
    reference_df = contamination_results[reference_contamination]
    top20 = reference_df[reference_df["ANOMALY_FLAG"] == -1].head(20).copy()

    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=top20,
        x="RISK_SCORE",
        y=top20["CLM_ID"].astype(str),
        hue="RISK_CATEGORY",
        dodge=False,
        palette="Reds",
    )
    plt.title(f"Top 20 Highest Risk Suspicious Claims (contamination={reference_contamination:.0%})")
    plt.xlabel("Risk Score")
    plt.ylabel("CLM_ID")
    plt.tight_layout()
    plt.savefig(output_plots_dir / "top20_suspicious_claims.png", dpi=300)
    plt.close()

    # C) Distribution of risk categories by contamination setting.
    category_rows = []
    ordered_categories = ["Low", "Medium", "High", "Critical"]
    for contamination, result_df in contamination_results.items():
        counts = (
            result_df["RISK_CATEGORY"]
            .value_counts()
            .reindex(ordered_categories, fill_value=0)
            .rename_axis("RISK_CATEGORY")
            .reset_index(name="COUNT")
        )
        counts["CONTAMINATION"] = f"{contamination:.0%}"
        category_rows.append(counts)

    category_df = pd.concat(category_rows, ignore_index=True)

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=category_df,
        x="RISK_CATEGORY",
        y="COUNT",
        hue="CONTAMINATION",
        palette="Set2",
    )
    plt.title("Risk Category Distribution")
    plt.xlabel("Risk Category")
    plt.ylabel("Claim Count")
    plt.tight_layout()
    plt.savefig(output_plots_dir / "risk_category_distribution.png", dpi=300)
    plt.close()

    logging.info("Visualizations saved in: %s", output_plots_dir)


def run_pipeline(base_dir: Path) -> None:
    """Execute the complete fraud anomaly detection workflow."""
    paths = ensure_directories(base_dir)
    data_path = base_dir / DATA_FILE

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    df = load_dataset(data_path)
    validate_required_columns(df, REQUIRED_COLUMNS)

    logging.info("Reducing dataset to required columns for memory efficiency.")
    df = df[REQUIRED_COLUMNS].copy()

    # Base payment and charge features retain direct claim and line level financial signals.
    # Engineered ratios capture overbilling and service intensity risk patterns.
    df = clean_data(df)
    df = engineer_features(df)
    df = add_peer_normalization(df)
    df = apply_log_transform(df, LOG_TRANSFORM_COLUMNS)

    X, robust_scaler = prepare_model_matrix(df, MODEL_FEATURES)
    joblib.dump(robust_scaler, paths["models"] / "robust_scaler.pkl")
    logging.info("Saved RobustScaler to: %s", paths["models"] / "robust_scaler.pkl")

    contamination_results: Dict[float, pd.DataFrame] = {}

    for contamination in CONTAMINATION_LEVELS:
        model = train_isolation_forest(X, contamination)
        predictions, decision_scores, risk_scores = score_claims(model, X)

        # Keep decision scores if analysts need explainability diagnostics later.
        _ = decision_scores

        result_df = build_result_frame(df, predictions, risk_scores)
        contamination_results[contamination] = result_df

        contamination_tag = f"{int(contamination * 100)}pct"

        output_file = paths["outputs"] / f"suspicious_claims_{contamination_tag}.csv"
        save_suspicious_claims(result_df, output_file)

        model_file = paths["models"] / f"isolation_forest_{contamination_tag}.pkl"
        save_model(model, model_file)

        summary = summarize_results(result_df)
        print_summary(contamination, summary)

    create_visualizations(contamination_results, paths["plots"])

    logging.info("Pipeline execution completed successfully.")


def main() -> None:
    """Program entrypoint with defensive exception handling."""
    setup_logging()

    try:
        base_dir = Path(__file__).resolve().parent
        run_pipeline(base_dir)
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
