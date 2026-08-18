"""
Run Fusion Pipeline - Entry point for complete healthcare fraud risk fusion.

Usage:
    python risk_fusion/run_fusion_pipeline.py

This script:
  1. Discovers and validates model outputs
  2. Loads scores from all models
  3. Normalizes to 0-100 risk scale
  4. Applies weighted ensemble fusion
  5. Generates final risk scores and explanations
  6. Produces validation report

Output Files:
  - outputs/final_provider_risk_scores.csv    (Main output with all risk scores)
  - outputs/provider_explanations.csv          (Risk explanation narratives)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from risk_fusion.fusion_engine import FusionEngine
from risk_fusion.generate_explanations import ExplanationEngine
from risk_fusion.model_registry import ModelRegistry
from risk_fusion.risk_utils import setup_logging

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"


def run_pipeline() -> int:
    """
    Run complete fusion pipeline.

    Returns:
        0 on success, 1 on failure
    """
    logger = setup_logging("FusionPipeline")

    logger.info("=" * 120)
    logger.info("HEALTHCARE FRAUD ANALYTICS FUSION LAYER")
    logger.info("=" * 120)

    try:
        # Step 1: Initialize model registry
        logger.info("\n[1/5] DISCOVERING MODELS...")
        registry = ModelRegistry()
        registry.print_summary()

        # Step 2: Initialize fusion engine
        logger.info("\n[2/5] INITIALIZING FUSION ENGINE...")
        engine = FusionEngine(registry)

        # Step 3: Perform fusion
        logger.info("\n[3/5] FUSING RISK SCORES...")
        df_results = engine.fuse_scores()

        # Step 4: Save results
        logger.info("\n[4/5] SAVING RESULTS...")
        output_path = engine.save_results()
        logger.info(f"✓ Saved {len(df_results)} provider scores to {output_path}")

        explanation_engine = ExplanationEngine()
        explanations_df = explanation_engine.generate_explanations(input_path=output_path, output_path=OUTPUT_DIR / "provider_explanations.csv")
        logger.info(f"✓ Saved {len(explanations_df)} provider explanations")

        # Step 5: Validate and report
        logger.info("\n[5/5] GENERATING VALIDATION REPORT...")
        engine.validate_results()

        logger.info("\n" + "=" * 120)
        logger.info("FUSION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 120)

        return 0

    except Exception as e:
        logger.error(f"\nFUSION PIPELINE FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    exit_code = run_pipeline()
    sys.exit(exit_code)
