"""
Generate Provider Explanations - Explainability adapter layer.

Purpose:
  - Load final provider risk scores
  - Generate risk explanation narratives for each provider
  - Classify into reason categories
  - Output to provider_explanations.csv

Explanation includes:
  - Risk score and level
  - Which models flagged the provider
  - Primary risk factors
  - Recommendation for investigation
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from risk_fusion.risk_utils import setup_logging

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"


class ExplanationEngine:
    """Generate explanations for provider risk scores."""

    def __init__(self):
        self.logger = setup_logging("ExplanationEngine")

    def _get_risk_description(self, risk_score: float) -> str:
        """Convert risk score to description."""
        if risk_score >= 81:
            return "CRITICAL"
        elif risk_score >= 61:
            return "HIGH"
        elif risk_score >= 31:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_model_flags(self, row: pd.Series) -> list[str]:
        """Identify which models flagged this provider."""
        flags = []
        threshold = 60.0  # Models flagging at 60+ percentile

        if pd.notna(row.get("IsolationForest_Risk")) and row["IsolationForest_Risk"] >= threshold:
            flags.append("Isolation Forest (anomaly detection)")

        if pd.notna(row.get("OCSVM_Risk")) and row["OCSVM_Risk"] >= threshold:
            flags.append("One-Class SVM (novelty detection)")

        if pd.notna(row.get("ClaimAutoencoder_Risk")) and row["ClaimAutoencoder_Risk"] >= threshold:
            flags.append("Claim Autoencoder (provider-level reconstruction anomaly)")

        if pd.notna(row.get("Autoencoder_Risk")) and row["Autoencoder_Risk"] >= threshold:
            flags.append("Autoencoder (reconstruction anomaly)")

        if pd.notna(row.get("CatBoost_Risk")) and row["CatBoost_Risk"] >= threshold:
            flags.append("CatBoost (fraud probability)")

        return flags

    def _get_highest_models(self, row: pd.Series, n: int = 3) -> list[tuple[str, float]]:
        """Get top N models by risk score."""
        models = {
            "Isolation Forest": row.get("IsolationForest_Risk", 0),
            "One-Class SVM": row.get("OCSVM_Risk", 0),
            "Autoencoder": row.get("Autoencoder_Risk", 0),
            "CatBoost": row.get("CatBoost_Risk", 0),
            "Claim Autoencoder": row.get("ClaimAutoencoder_Risk", 0),
        }
        sorted_models = sorted(models.items(), key=lambda x: x[1], reverse=True)
        return sorted_models[:n]

    def generate_explanation(self, row: pd.Series) -> str:
        """Generate narrative explanation for a provider's risk."""
        provider = row["Provider"]
        risk_score = row["Final_Risk_Score"]
        risk_level = row.get("Risk_Level", "UNKNOWN")

        # Start with basic info
        explanation = f"Provider {provider} has a FINAL RISK SCORE of {risk_score:.1f}/100 ({risk_level}).\n"

        # Get model flags
        flags = self._get_model_flags(row)
        if flags:
            explanation += f"Flagged by: {', '.join(flags)}. "
        else:
            explanation += "Flagged by: Ensemble combination of anomaly detection models. "

        # Get top contributing models
        top_models = self._get_highest_models(row, n=3)
        if top_models:
            explanation += f"Top risk factors: "
            factors = [f"{model} ({score:.0f}/100)" for model, score in top_models if score > 0]
            explanation += ", ".join(factors) + ". "

        # Add context
        if risk_score >= 81:
            explanation += "RECOMMENDATION: IMMEDIATE investigation required. High confidence of fraudulent activity."
        elif risk_score >= 61:
            explanation += "RECOMMENDATION: Priority investigation. Significant anomalies detected across multiple models."
        elif risk_score >= 31:
            explanation += "RECOMMENDATION: Standard review. Moderate risk indicators present."
        else:
            explanation += "RECOMMENDATION: Monitor for changes. Low risk profile."

        return explanation

    def generate_explanations(self, input_path: Path | None = None, output_path: Path | None = None) -> pd.DataFrame:
        """
        Load risk scores and generate explanations.

        Args:
            input_path: Path to final_provider_risk_scores.csv (auto-discovered if None)
            output_path: Path to save provider_explanations.csv (auto-discovered if None)

        Returns:
            DataFrame with explanations
        """
        if input_path is None:
            input_path = OUTPUT_DIR / "final_provider_risk_scores.csv"

        if output_path is None:
            output_path = OUTPUT_DIR / "provider_explanations.csv"

        # Load final scores
        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            return pd.DataFrame()

        self.logger.info(f"Loading final scores from {input_path}")
        df = pd.read_csv(input_path)
        self.logger.info(f"Loaded {len(df)} providers")

        # Generate explanations
        self.logger.info("Generating risk explanations...")
        df["Explanation"] = df.apply(self.generate_explanation, axis=1)

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_cols = ["Provider", "Final_Risk_Score", "Risk_Level", "Explanation"]
        output_cols = [col for col in output_cols if col in df.columns]

        df_out = df[output_cols].sort_values("Final_Risk_Score", ascending=False).reset_index(drop=True)
        df_out.to_csv(output_path, index=False)

        self.logger.info(f"Saved explanations to {output_path}")
        self.logger.info(f"Sample explanations (top 3 risk providers):")

        for idx, row in df_out.head(3).iterrows():
            print(f"\n  Provider {row['Provider']} (Risk: {row['Final_Risk_Score']:.1f}):")
            print(f"  {row['Explanation']}")

        return df_out


def generate_provider_explanations() -> int:
    """Entry point for explanation generation."""
    logger = setup_logging("ExplanationGenerator")

    try:
        logger.info("=" * 100)
        logger.info("GENERATING PROVIDER RISK EXPLANATIONS")
        logger.info("=" * 100)

        engine = ExplanationEngine()
        df_explanations = engine.generate_explanations()

        logger.info("=" * 100)
        logger.info(f"SUCCESS: Generated explanations for {len(df_explanations)} providers")
        logger.info("=" * 100)

        return 0

    except Exception as e:
        logger.error(f"FAILED to generate explanations: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    exit_code = generate_provider_explanations()
    exit(exit_code)
