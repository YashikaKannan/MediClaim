from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "processed" / "PROVIDER_ML_READY_KAGGLE.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class PeerBenchmarkingEngine:
    """Independent provider peer benchmarking model."""

    def __init__(self, dataset_path: Path | str | None = None):
        self.dataset_path = Path(dataset_path) if dataset_path is not None else DATASET_PATH
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_mad(values: pd.Series) -> float:
        if values.empty:
            return 0.0
        median = values.median()
        deviations = (values - median).abs()
        mad = deviations.median()
        return float(mad if pd.notna(mad) else 0.0)

    @staticmethod
    def _robust_z_score(value: float, median: float, mad: float) -> float:
        if pd.isna(value) or pd.isna(median):
            return 0.0
        if mad == 0 or pd.isna(mad):
            return 0.0
        return float((value - median) / (1.4826 * mad))

    def _detect_peer_group_column(self, df: pd.DataFrame) -> str:
        for candidate in ["Rndrng_Prvdr_Type", "Provider_Type", "Specialty"]:
            if candidate in df.columns:
                return candidate
        return "Provider"

    def _detect_feature_columns(self, df: pd.DataFrame) -> list[str]:
        preferred = [
            "Total_Reimbursement",
            "Total_Claim_Count",
            "Reimbursement_Per_Claim",
            "Total_Unique_Beneficiaries",
            "Claims_Per_Beneficiary",
            "Average_Beneficiary_Age",
            "Diagnosis_Diversity",
            "Procedure_Diversity",
            "Average_Length_Of_Stay",
        ]
        available = [col for col in preferred if col in df.columns]
        if available:
            return available

        numeric_cols = [
            col for col in df.select_dtypes(include="number").columns
            if col not in {"Provider", "Fraud_Label", "Unnamed: 0"}
        ]
        return numeric_cols

    def build_peer_scores(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_path)
        if "Provider" not in df.columns:
            raise ValueError("Provider column required for peer benchmarking")

        peer_group_col = self._detect_peer_group_column(df)
        feature_cols = self._detect_feature_columns(df)
        if not feature_cols:
            raise ValueError("No usable provider-level feature columns found for peer benchmarking")

        all_rows: list[dict] = []

        for group_name, group_df in df.groupby(peer_group_col, dropna=False):
            group_df = group_df.copy()
            peer_count = len(group_df)
            risk_columns = []

            for feature in feature_cols:
                if feature not in group_df.columns:
                    continue
                series = pd.to_numeric(group_df[feature], errors='coerce')
                median = series.median()
                mad = self._safe_mad(series)
                robust_z = group_df[feature].apply(
                    lambda value: self._robust_z_score(float(value), median, mad) if pd.notna(value) else 0.0
                )
                feature_risk = robust_z.abs().rank(pct=True, method="average") * 100.0
                risk_columns.append(f"{feature}_risk")
                group_df[f"{feature}_risk"] = feature_risk

            if not risk_columns:
                continue

            group_df["Peer_Risk_Score"] = group_df[risk_columns].mean(axis=1).clip(0, 100)
            group_df["Peer_Percentile"] = group_df["Peer_Risk_Score"].rank(pct=True, method="average") * 100.0
            group_df["Peer_Rank"] = group_df["Peer_Risk_Score"].rank(method="min", ascending=False).astype(int)
            group_df["Peer_Count"] = peer_count
            group_df["Peer_Group"] = group_name

            for _, row in group_df.iterrows():
                all_rows.append(
                    {
                        "Provider": row["Provider"],
                        "Peer_Group": row["Peer_Group"],
                        "Peer_Risk_Score": float(row["Peer_Risk_Score"]),
                        "Peer_Percentile": float(row["Peer_Percentile"]),
                        "Peer_Count": int(row["Peer_Count"]),
                        "Peer_Rank": int(row["Peer_Rank"]),
                    }
                )

        if not all_rows:
            return pd.DataFrame(columns=["Provider", "Peer_Group", "Peer_Risk_Score", "Peer_Percentile", "Peer_Count", "Peer_Rank"])

        peer_df = pd.DataFrame(all_rows)
        peer_df["Peer_Risk_Score"] = pd.to_numeric(peer_df["Peer_Risk_Score"], errors="coerce").clip(0, 100)
        peer_df["Peer_Percentile"] = pd.to_numeric(peer_df["Peer_Percentile"], errors="coerce").clip(0, 100)
        peer_df["Peer_Count"] = pd.to_numeric(peer_df["Peer_Count"], errors="coerce").fillna(0).astype(int)
        peer_df["Peer_Rank"] = pd.to_numeric(peer_df["Peer_Rank"], errors="coerce").fillna(0).astype(int)
        peer_df = peer_df.sort_values(["Peer_Group", "Peer_Risk_Score"], ascending=[True, False]).reset_index(drop=True)
        return peer_df[["Provider", "Peer_Group", "Peer_Risk_Score", "Peer_Percentile", "Peer_Count", "Peer_Rank"]].copy()

    def build_top_outliers(self, peer_df: pd.DataFrame | None = None, n: int = 100) -> pd.DataFrame:
        if peer_df is None:
            peer_df = self.build_peer_scores()

        top = peer_df.sort_values("Peer_Risk_Score", ascending=False).head(n).copy().reset_index(drop=True)
        return top[["Provider", "Peer_Group", "Peer_Risk_Score", "Peer_Percentile", "Peer_Count", "Peer_Rank"]].copy()

    def build_explanations(self, peer_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if peer_df is None:
            peer_df = self.build_peer_scores()

        records = []
        for _, row in peer_df.iterrows():
            provider = row["Provider"]
            group = row["Peer_Group"]
            score = float(row["Peer_Risk_Score"])
            percentile = float(row["Peer_Percentile"])
            rank = int(row["Peer_Rank"])
            text = (
                f"Provider {provider} is an outlier versus {group} peers on the selected utilization and reimbursement metrics. "
                f"Peer_Risk_Score = {score:.1f}; percentile rank = {percentile:.0f}; peer rank = {rank} of {int(row['Peer_Count'])}."
            )
            records.append(
                {
                    "Provider": provider,
                    "Peer_Group": group,
                    "Peer_Risk_Score": score,
                    "Peer_Percentile": percentile,
                    "Peer_Rank": rank,
                    "Explanation": text,
                }
            )

        return pd.DataFrame(records)

    def save_outputs(self, peer_df: pd.DataFrame | None = None) -> tuple[Path, Path, Path]:
        if peer_df is None:
            peer_df = self.build_peer_scores()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        score_path = self.output_dir / "peer_benchmark_scores.csv"
        top_path = self.output_dir / "top_peer_outliers.csv"
        explanations_path = self.output_dir / "peer_explanations.csv"

        peer_df.to_csv(score_path, index=False)
        self.build_top_outliers(peer_df).to_csv(top_path, index=False)
        self.build_explanations(peer_df).to_csv(explanations_path, index=False)

        return score_path, top_path, explanations_path

    def validate(self) -> None:
        peer_df = self.build_peer_scores()
        print("Total Providers:", len(peer_df))
        print("Total Peer Groups:", peer_df["Peer_Group"].nunique())
        print("Risk Distribution:")
        print(peer_df["Peer_Risk_Score"].describe().to_string())
        print("Top 20 Peer Outliers:")
        print(self.build_top_outliers(peer_df, 20).to_string(index=False))


def main() -> None:
    engine = PeerBenchmarkingEngine()
    peer_df = engine.build_peer_scores()
    engine.save_outputs(peer_df)
    engine.validate()


if __name__ == "__main__":
    main()
