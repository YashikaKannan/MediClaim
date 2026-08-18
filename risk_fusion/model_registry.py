"""
Model Registry - Discovers and validates model outputs for fusion layer.

Purpose:
  - Automatically discovers model output files
  - Validates file existence
  - Logs missing models gracefully
  - Continues pipeline even if some models missing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"


class ModelRegistry:
    """Registry for all fraud detection models."""

    def __init__(self):
        self.logger = self._setup_logging()
        self.models: Dict[str, Dict] = {}
        self._register_all_models()

    @staticmethod
    def _setup_logging() -> logging.Logger:
        logger = logging.getLogger("ModelRegistry")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _register_all_models(self) -> None:
        """Register all known models."""
        # Isolation Forest
        self._register_model(
            name="Isolation Forest",
            source_dir=BASE_DIR / "isolation-forest",
            model_file="models/isolation_forest_model.pkl",
            output_csv="kaggle_iforest_scores.csv",
            output_location=OUTPUT_DIR,
            required=True,
        )

        # One-Class SVM
        self._register_model(
            name="One-Class SVM",
            source_dir=OUTPUT_DIR,
            model_file="kaggle_ocsvm_model.joblib",
            output_csv="kaggle_ocsvm_scores.csv",
            output_location=OUTPUT_DIR,
            required=False,
        )

        # Autoencoder (Provider-level)
        self._register_model(
            name="Autoencoder",
            source_dir=BASE_DIR / "autoencoder",
            model_file="model/claim_autoencoder.keras",
            output_csv="autoencoder_provider_scores.csv",
            output_location=OUTPUT_DIR,
            required=False,
        )

        # Claim Autoencoder (must be aggregated)
        self._register_model(
            name="Claim Autoencoder",
            source_dir=BASE_DIR / "claim_autoencoder",
            model_file="model/claim_autoencoder.keras",
            output_csv="provider_investigation_queue.csv",
            output_location=BASE_DIR / "claim_autoencoder" / "results",
            required=False,
        )

        # CatBoost
        self._register_model(
            name="CatBoost",
            source_dir=BASE_DIR / "catboost_model",
            model_file="output/provider_risk_ranked.csv",
            output_csv="provider_risk_ranked.csv",
            output_location=BASE_DIR / "catboost_model" / "output",
            required=False,
        )

        # Peer Benchmarking (disabled: Experimental / Phase-2)
        self._register_model(
            name="Peer Benchmarking",
            source_dir=BASE_DIR / "peer-benchmarking",
            model_file="peer_benchmarking.py",
            output_csv="peer_benchmark_scores.csv",
            output_location=BASE_DIR / "peer-benchmarking" / "output",
            required=False,
        )

    def _register_model(
        self,
        name: str,
        source_dir: Path,
        model_file: str,
        output_csv: str,
        output_location: Path,
        required: bool = False,
    ) -> None:
        """Register a model and validate its files."""
        model_path = source_dir / model_file
        output_path = output_location / output_csv

        exists_model = model_path.exists()
        exists_output = output_path.exists()

        if name == "Peer Benchmarking":
            status = "DISABLED (Experimental / Phase-2)"
        else:
            status = "OK" if exists_output else "MISSING"

        self.models[name] = {
            "model_file": model_path,
            "output_csv": output_path,
            "exists_model": exists_model,
            "exists_output": exists_output,
            "required": required,
            "status": status,
        }

        if exists_output:
            self.logger.info(f"✓ {name:25s} → Output found: {output_csv}")
        elif required:
            self.logger.warning(f"⚠ {name:25s} → REQUIRED output missing: {output_csv}")
        else:
            self.logger.info(f"○ {name:25s} → Output not found (optional): {output_csv}")

    def get_output_csv(self, model_name: str) -> Optional[Path]:
        """Get output CSV path for a model."""
        if model_name not in self.models:
            self.logger.warning(f"Model not found in registry: {model_name}")
            return None

        model = self.models[model_name]
        if model["exists_output"]:
            return model["output_csv"]

        return None

    def all_outputs_found(self) -> bool:
        """Check if all required outputs exist."""
        missing = [name for name, info in self.models.items() if info["required"] and not info["exists_output"]]
        return len(missing) == 0

    def get_summary(self) -> Dict:
        """Get registry summary."""
        total = len(self.models)
        found = sum(1 for info in self.models.values() if info["exists_output"])
        required_missing = sum(1 for info in self.models.values() if info["required"] and not info["exists_output"])

        return {
            "total_models": total,
            "outputs_found": found,
            "outputs_missing": total - found,
            "required_missing": required_missing,
            "models": self.models,
        }

    def print_summary(self) -> None:
        """Print registry summary."""
        summary = self.get_summary()
        print()
        print("=" * 100)
        print("MODEL REGISTRY SUMMARY")
        print("=" * 100)
        print(f"\nTotal Models:        {summary['total_models']}")
        print(f"Outputs Found:       {summary['outputs_found']}")
        print(f"Outputs Missing:     {summary['outputs_missing']}")
        print(f"Required Missing:    {summary['required_missing']}")
        print()

        for name, info in summary["models"].items():
            status_marker = "✓" if info["exists_output"] else "✗"
            print(f"  {status_marker} {name:25s} → {info['status']}")

        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    registry = ModelRegistry()
    registry.print_summary()
