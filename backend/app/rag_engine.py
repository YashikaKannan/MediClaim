#!/usr/bin/env python3
"""FAISS + sentence-transformers retrieval layer for fraud evidence."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


KNOWLEDGE_BASE_FILE = "fraud_knowledge_base.json"
OUTPUT_DIR = "outputs/explanations"
RAG_EVIDENCE_FILE = "rag_retrieved_patterns.csv"


@dataclass
class RetrievedPattern:
    """Represents one retrieved fraud pattern match."""

    pattern_name: str
    description: str
    business_reason: str
    severity: str
    score: float


class FraudRAGEngine:
    """Embeds fraud pattern knowledge and retrieves best matches for suspicious claims."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.patterns: List[Dict[str, str]] = []
        self.index: faiss.IndexFlatIP | None = None
        self.pattern_embeddings: np.ndarray | None = None

    def load_knowledge_base(self, kb_path: Path) -> None:
        """Load fraud knowledge patterns from JSON file."""
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

        with kb_path.open("r", encoding="utf-8") as f:
            patterns = json.load(f)

        if not isinstance(patterns, list) or len(patterns) == 0:
            raise ValueError("Knowledge base must contain a non-empty list of patterns.")

        required_fields = {"pattern_name", "description", "business_reason", "severity"}
        for idx, item in enumerate(patterns):
            missing = required_fields.difference(item.keys())
            if missing:
                raise ValueError(f"Pattern at index {idx} is missing fields: {sorted(missing)}")

        self.patterns = patterns
        logging.info("Loaded %d fraud patterns from knowledge base.", len(self.patterns))

    @staticmethod
    def pattern_to_text(pattern: Dict[str, str]) -> str:
        """Create semantic text payload for embedding search."""
        return (
            f"Pattern: {pattern['pattern_name']}. "
            f"Description: {pattern['description']} "
            f"Business reason: {pattern['business_reason']} "
            f"Severity: {pattern['severity']}"
        )

    def build_index(self) -> None:
        """Build FAISS cosine-similarity index from pattern embeddings."""
        if not self.patterns:
            raise ValueError("Knowledge base patterns are empty. Load knowledge base first.")

        logging.info("Loading sentence-transformer model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)

        corpus = [self.pattern_to_text(p) for p in self.patterns]
        logging.info("Generating embeddings for %d fraud patterns.", len(corpus))

        embeddings = self.model.encode(
            corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.pattern_embeddings = embeddings

        logging.info("FAISS index built. Embedding dimension: %d", dim)

    def retrieve_top_patterns(self, summary: str, top_k: int = 3) -> List[RetrievedPattern]:
        """Retrieve top-k pattern matches for one anomaly summary."""
        if self.model is None or self.index is None:
            raise RuntimeError("RAG engine is not initialized. Call build_index first.")

        query_vec = self.model.encode(
            [summary],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        scores, indices = self.index.search(query_vec, top_k)

        results: List[RetrievedPattern] = []
        for score, idx in zip(scores[0], indices[0]):
            pattern = self.patterns[int(idx)]
            results.append(
                RetrievedPattern(
                    pattern_name=pattern["pattern_name"],
                    description=pattern["description"],
                    business_reason=pattern["business_reason"],
                    severity=pattern["severity"],
                    score=float(score),
                )
            )
        return results

    def retrieve_top_patterns_batch(
        self,
        summaries: List[str],
        top_k: int = 3,
        batch_size: int = 512,
    ) -> List[List[RetrievedPattern]]:
        """Retrieve top-k pattern matches for many summaries efficiently."""
        if self.model is None or self.index is None:
            raise RuntimeError("RAG engine is not initialized. Call build_index first.")

        all_results: List[List[RetrievedPattern]] = []

        for start in tqdm(range(0, len(summaries), batch_size), desc="Encoding summaries"):
            end = min(start + batch_size, len(summaries))
            batch = summaries[start:end]

            query_vecs = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
            ).astype("float32")

            scores, indices = self.index.search(query_vecs, top_k)

            for row_scores, row_indices in zip(scores, indices):
                one_result: List[RetrievedPattern] = []
                for score, idx in zip(row_scores, row_indices):
                    pattern = self.patterns[int(idx)]
                    one_result.append(
                        RetrievedPattern(
                            pattern_name=pattern["pattern_name"],
                            description=pattern["description"],
                            business_reason=pattern["business_reason"],
                            severity=pattern["severity"],
                            score=float(score),
                        )
                    )
                all_results.append(one_result)

        return all_results


def build_rag_evidence(
    claims_df: pd.DataFrame,
    base_dir: Path,
    kb_file: str = KNOWLEDGE_BASE_FILE,
    output_dir: str = OUTPUT_DIR,
    top_k: int = 3,
) -> pd.DataFrame:
    """Run RAG retrieval for suspicious claims and save evidence output."""
    output_path = base_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    engine = FraudRAGEngine()
    engine.load_knowledge_base(base_dir / kb_file)
    engine.build_index()

    required_cols = {"CLAIM_UID", "CLM_ID", "BENE_ID", "ANOMALY_SUMMARY"}
    missing = sorted([c for c in required_cols if c not in claims_df.columns])
    if missing:
        raise ValueError(f"Claims DataFrame missing required columns: {missing}")

    records: List[Dict[str, object]] = []

    claim_rows = [row._asdict() for row in claims_df.itertuples(index=False)]
    summaries = [str(row.get("ANOMALY_SUMMARY", "")) for row in claim_rows]
    all_results = engine.retrieve_top_patterns_batch(summaries=summaries, top_k=top_k)

    for row_dict, results in tqdm(
        zip(claim_rows, all_results),
        total=len(claim_rows),
        desc="Assembling retrieval output",
    ):

        record: Dict[str, object] = {
            "CLAIM_UID": int(row_dict.get("CLAIM_UID")),
            "CLM_ID": row_dict.get("CLM_ID"),
            "BENE_ID": row_dict.get("BENE_ID"),
        }

        for i in range(top_k):
            if i < len(results):
                rec = results[i]
                rank = i + 1
                record[f"RETRIEVED_PATTERN_{rank}"] = rec.pattern_name
                record[f"RETRIEVED_PATTERN_{rank}_DESC"] = rec.description
                record[f"RETRIEVED_PATTERN_{rank}_SEVERITY"] = rec.severity
                record[f"RETRIEVED_PATTERN_{rank}_SCORE"] = round(rec.score, 6)
            else:
                rank = i + 1
                record[f"RETRIEVED_PATTERN_{rank}"] = ""
                record[f"RETRIEVED_PATTERN_{rank}_DESC"] = ""
                record[f"RETRIEVED_PATTERN_{rank}_SEVERITY"] = ""
                record[f"RETRIEVED_PATTERN_{rank}_SCORE"] = np.nan

        records.append(record)

    rag_df = pd.DataFrame(records)
    rag_path = output_path / RAG_EVIDENCE_FILE
    rag_df.to_csv(rag_path, index=False)

    logging.info("Saved RAG retrieval output to: %s", rag_path)
    return rag_df


def setup_logging() -> None:
    """Configure logging for standalone usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Standalone test harness that expects explanation-engine output."""
    setup_logging()
    try:
        base_dir = Path(__file__).resolve().parent
        explanation_path = base_dir / OUTPUT_DIR / "suspicious_claims_with_metrics.csv"

        if not explanation_path.exists():
            raise FileNotFoundError(
                f"Expected explanation-engine intermediate file at: {explanation_path}"
            )

        claims_df = pd.read_csv(explanation_path, low_memory=False)
        _ = build_rag_evidence(claims_df=claims_df, base_dir=base_dir)
        logging.info("RAG engine completed successfully.")
    except Exception as exc:
        logging.exception("RAG engine failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
