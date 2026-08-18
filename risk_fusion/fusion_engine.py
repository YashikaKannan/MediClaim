"""
Fusion Engine - Combines multiple model risk scores into final provider risk.

Purpose:
  - Load all model output CSVs
  - Normalize scores to 0-100 risk scale using percentile ranking
  - Apply weighted ensemble formula
  - Generate final provider risk scores
  - Calculate investigation priority scores
  - Output to CSV
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from risk_fusion.model_registry import ModelRegistry
from risk_fusion.risk_utils import (
    apply_priority_score,
    apply_risk_level,
    clamp_risk,
    normalize_fraud_proba_to_risk,
    percentile_risk,
    print_risk_distribution,
    print_risk_level_counts,
    print_top_providers,
    setup_logging,
)

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"


class FusionEngine:
    """Ensemble fusion of multiple fraud detection models."""

    # Weights for ensemble fusion
    WEIGHTS = {
        "IsolationForest_Risk": 0.25,
        "OCSVM_Risk": 0.20,
        "Autoencoder_Risk": 0.20,
        "CatBoost_Risk": 0.25,
        "ClaimAutoencoder_Risk": 0.10,
    }

    def __init__(self, registry: ModelRegistry | None = None):
        self.logger = setup_logging("FusionEngine")
        self.registry = registry or ModelRegistry()
        self.df_final = None
        self.models_used: list[str] = []
        self.models_skipped: list[str] = []

    def load_isolation_forest_scores(self) -> pd.DataFrame | None:
        """Load Isolation Forest scores."""
        path = self.registry.get_output_csv("Isolation Forest")
        if path is None or not path.exists():
            self.logger.warning("Isolation Forest scores not found")
            return None

        df = pd.read_csv(path)
        self.logger.info(f"Loaded Isolation Forest scores: {df.shape[0]} providers")

        if "IsolationForest_Risk" not in df.columns:
            if "IForest_Risk" in df.columns:
                df["IsolationForest_Risk"] = df["IForest_Risk"]
            elif "IForest_Score" in df.columns:
                df["IsolationForest_Risk"] = percentile_risk(df["IForest_Score"])
            else:
                self.logger.warning("Could not find Isolation Forest risk score column")
                return None

        return df[["Provider", "IsolationForest_Risk"]].copy()

    def load_ocsvm_scores(self) -> pd.DataFrame | None:
        """Load One-Class SVM scores."""
        path = self.registry.get_output_csv("One-Class SVM")
        if path is None or not path.exists():
            self.logger.warning("One-Class SVM scores not found")
            return None

        df = pd.read_csv(path)
        self.logger.info(f"Loaded OCSVM scores: {df.shape[0]} providers")

        if "OCSVM_Risk" not in df.columns and "OCSVM_Score" in df.columns:
            df["OCSVM_Risk"] = percentile_risk(df["OCSVM_Score"])

        return df[["Provider", "OCSVM_Risk"]].copy()

    def load_claim_autoencoder_scores(self) -> pd.DataFrame | None:
        """Load provider-level claim autoencoder risk scores."""
        path = self.registry.get_output_csv("Claim Autoencoder")
        if path is None or not path.exists():
            self.logger.warning("Claim Autoencoder scores not found")
            return None

        df = pd.read_csv(path)
        self.logger.info(f"Loaded Claim Autoencoder scores: {df.shape[0]} providers")

        if "ClaimAutoencoder_Risk" not in df.columns:
            if "AverageRiskScore" in df.columns:
                df["ClaimAutoencoder_Risk"] = df["AverageRiskScore"].clip(0, 100)
            elif "ProviderRiskScore" in df.columns:
                df["ClaimAutoencoder_Risk"] = df["ProviderRiskScore"].clip(0, 100)
            elif "RiskScore" in df.columns:
                df["ClaimAutoencoder_Risk"] = df["RiskScore"].clip(0, 100)
            else:
                self.logger.warning("Could not find Claim Autoencoder risk score column")
                return None

        return df[["Provider", "ClaimAutoencoder_Risk"]].copy()

    def load_autoencoder_scores(self) -> pd.DataFrame | None:
        """Load Autoencoder scores."""
        path = self.registry.get_output_csv("Autoencoder")
        if path is None or not path.exists():
            self.logger.info("Autoencoder scores not found (optional)")
            return None

        df = pd.read_csv(path)
        self.logger.info(f"Loaded Autoencoder scores: {df.shape[0]} providers")

        if "Autoencoder_Risk" not in df.columns:
            if "ProviderRiskScore" in df.columns:
                df["Autoencoder_Risk"] = percentile_risk(df["ProviderRiskScore"])
            elif "AverageRiskScore" in df.columns:
                df["Autoencoder_Risk"] = percentile_risk(df["AverageRiskScore"])
            else:
                self.logger.warning("Could not find autoencoder risk score column")
                return None

        return df[["Provider", "Autoencoder_Risk"]].copy()

    def load_catboost_scores(self) -> pd.DataFrame | None:
        """Load CatBoost fraud probabilities and convert to risk."""
        path = self.registry.get_output_csv("CatBoost")
        if path is None or not path.exists():
            self.logger.warning("CatBoost scores not found")
            return None

        df = pd.read_csv(path)
        self.logger.info(f"Loaded CatBoost scores: {df.shape[0]} providers")

        # Use fraud_proba as risk if available, otherwise Risk_Score
        if "fraud_proba" in df.columns:
            df["CatBoost_Risk"] = normalize_fraud_proba_to_risk(df["fraud_proba"])
        elif "Risk_Score" in df.columns:
            df["CatBoost_Risk"] = df["Risk_Score"].clip(0, 100)
        else:
            self.logger.warning("Could not find CatBoost fraud probability or risk score")
            return None

        return df[["Provider", "CatBoost_Risk"]].copy()

    def load_peer_benchmark_scores(self) -> pd.DataFrame | None:
        """Peer Benchmarking is disabled in production because peer groups are invalid."""
        path = self.registry.get_output_csv("Peer Benchmarking")
        if path is not None and path.exists():
            self.logger.warning("Peer Benchmarking disabled due to invalid peer groups")
        else:
            self.logger.warning("Peer Benchmarking disabled due to invalid peer groups")
        return None

    def load_fraud_labels(self) -> pd.DataFrame | None:
        """Load fraud labels from any model output."""
        # Try to get from Isolation Forest first
        path = self.registry.get_output_csv("Isolation Forest")
        if path and path.exists():
            df = pd.read_csv(path)
            if "Fraud_Label" in df.columns:
                return df[["Provider", "Fraud_Label"]].copy()

        # Try One-Class SVM
        path = self.registry.get_output_csv("One-Class SVM")
        if path and path.exists():
            df = pd.read_csv(path)
            if "Fraud_Label" in df.columns:
                return df[["Provider", "Fraud_Label"]].copy()

        return None

    def fuse_scores(self) -> pd.DataFrame:
        """
        Fuse all model scores into final risk.

        Steps:
          1. Load all model scores
          2. Merge on Provider
          3. Fill missing values with 0
          4. Apply weighted ensemble
          5. Clamp to [0, 100]
          6. Classify risk levels
          7. Calculate priority scores
        """
        self.logger.info("=" * 100)
        self.logger.info("FUSION ENGINE: STARTING ENSEMBLE FUSION")
        self.logger.info("=" * 100)

        # Load all scores
        iforest = self.load_isolation_forest_scores()
        ocsvm = self.load_ocsvm_scores()
        autoenc = self.load_autoencoder_scores()
        catboost = self.load_catboost_scores()
        claim_autoencoder = self.load_claim_autoencoder_scores()
        peer_benchmark = self.load_peer_benchmark_scores()
        labels = self.load_fraud_labels()

        if iforest is None:
            raise ValueError("Isolation Forest scores are required for fusion")

        self.models_used = []
        self.models_skipped = []

        model_loaders = [
            ("IsolationForest_Risk", iforest, "Isolation Forest"),
            ("OCSVM_Risk", ocsvm, "One-Class SVM"),
            ("Autoencoder_Risk", autoenc, "Autoencoder"),
            ("CatBoost_Risk", catboost, "CatBoost"),
            ("ClaimAutoencoder_Risk", claim_autoencoder, "Claim Autoencoder"),
        ]

        # Start with Isolation Forest
        df = iforest.copy()
        self.models_used.append("Isolation Forest")

        # Merge all other scores
        for risk_col, score_df, model_name in model_loaders[1:]:
            if score_df is not None:
                df = df.merge(score_df, on="Provider", how="left")
                self.models_used.append(model_name)
            else:
                self.models_skipped.append(model_name)
                self.logger.warning(f"Missing model output for {model_name}; redistributing weights proportionally")
                df[risk_col] = 0.0

        if labels is not None:
            df = df.merge(labels, on="Provider", how="left")
        else:
            df["Fraud_Label"] = 0

        # Fill missing values
        for risk_col in self.WEIGHTS.keys():
            if risk_col not in df.columns:
                df[risk_col] = 0.0
            df[risk_col] = df[risk_col].fillna(0.0)

        present_weight_names = [risk_col for risk_col in self.WEIGHTS if df[risk_col].notna().any() and (df[risk_col] != 0).any()]
        total_present_weight = sum(self.WEIGHTS[risk_col] for risk_col in present_weight_names)

        if total_present_weight <= 0:
            raise ValueError("No valid model scores available for fusion")

        if len(present_weight_names) < len(self.WEIGHTS):
            self.logger.warning(
                "Redistributing weights proportionally because some model outputs are missing: "
                + ", ".join(sorted(set(self.WEIGHTS) - set(present_weight_names)))
            )

        # Apply weighted ensemble with proportional redistribution
        self.logger.info("Applying weighted ensemble formula...")
        df["Final_Risk_Score"] = 0.0

        for risk_col, weight in self.WEIGHTS.items():
            if risk_col in df.columns and risk_col in present_weight_names:
                adjusted_weight = weight / total_present_weight
                df["Final_Risk_Score"] += adjusted_weight * df[risk_col]
                self.logger.info(f"  {risk_col:25s} weight={adjusted_weight:.4f} (base={weight:.2f})")

        # Clamp to [0, 100]
        df["Final_Risk_Score"] = df["Final_Risk_Score"].apply(clamp_risk)
        df["Final_Risk_Score"] = df["Final_Risk_Score"].round(2)

        # Classify risk levels
        df = apply_risk_level(df, score_col="Final_Risk_Score", output_col="Risk_Level")

        # Calculate priority scores
        df = self._add_priority_scores(df)

        self.df_final = df
        self.logger.info(f"Fusion complete: {df.shape[0]} providers scored")

        return df

    def _add_priority_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add priority scores for investigation prioritization."""
        # Try to load reimbursement data
        try:
            kaggle_path = BASE_DIR / "processed" / "PROVIDER_ML_READY_KAGGLE.csv"
            if kaggle_path.exists():
                kaggle = pd.read_csv(kaggle_path)
                if "Provider" in kaggle.columns and "Total_Reimbursement" in kaggle.columns:
                    df = df.merge(kaggle[["Provider", "Total_Reimbursement"]], on="Provider", how="left")
                    df = apply_priority_score(df, risk_col="Final_Risk_Score", reimbursement_col="Total_Reimbursement")
                    self.logger.info("Priority scores calculated using reimbursement data")
                else:
                    df["Priority_Score"] = df["Final_Risk_Score"]
            else:
                df["Priority_Score"] = df["Final_Risk_Score"]
        except Exception as e:
            self.logger.warning(f"Could not add reimbursement data: {e}")
            df["Priority_Score"] = df["Final_Risk_Score"]
    
        return df

    def save_results(self, output_path: Path | None = None) -> Path:
        """Save final scores to CSV."""
        if self.df_final is None:
            raise ValueError("No fused data available. Call fuse_scores() first.")

        if output_path is None:
            output_path = OUTPUT_DIR / "final_provider_risk_scores.csv"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Select output columns
        output_cols = [
            "Provider",
            "Fraud_Label",
            "IsolationForest_Risk",
            "OCSVM_Risk",
            "Autoencoder_Risk",
            "CatBoost_Risk",
            "ClaimAutoencoder_Risk",
            "Final_Risk_Score",
            "Risk_Level",
            "Priority_Score",
        ]

        # Only include columns that exist
        output_cols = [col for col in output_cols if col in self.df_final.columns]

        df_out = self.df_final[output_cols].sort_values("Final_Risk_Score", ascending=False).reset_index(drop=True)

        df_out.to_csv(output_path, index=False)
        self.logger.info(f"Saved results to {output_path}")

        return output_path

    def validate_results(self) -> None:
        """Validate and print results summary."""
        if self.df_final is None:
            self.logger.error("No fused data available")
            return

        print()
        print("=" * 100)
        print("FUSION RESULTS SUMMARY")
        print("=" * 100)

        print(f"\nModels Used: {', '.join(self.models_used) if self.models_used else 'None'}")
        print(f"Models Skipped: {', '.join(self.models_skipped) if self.models_skipped else 'None'}")
        print(f"Providers Scored: {len(self.df_final)}")
        print(f"Final Provider Count: {len(self.df_final)}")
        print(f"Missing Fraud Labels: {self.df_final['Fraud_Label'].isna().sum()}")

        print_risk_distribution(self.df_final, risk_col="Final_Risk_Score")
        print_risk_level_counts(self.df_final, risk_level_col="Risk_Level")
        print_top_providers(self.df_final, n=20, risk_col="Final_Risk_Score")

        print()
        print("=" * 100)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    registry = ModelRegistry()
    engine = FusionEngine(registry)
    df_results = engine.fuse_scores()
    engine.save_results()
    engine.validate_results()
