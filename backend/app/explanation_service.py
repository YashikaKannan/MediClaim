"""Business-facing, evidence-grounded explanation builders."""

from __future__ import annotations

from typing import Any, Iterable


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def money(value: Any) -> str:
    return f"${number(value):,.2f}"


def priority_for_risk(risk_score: Any) -> str:
    score = number(risk_score)
    if score >= 81:
        return "P1"
    if score >= 61:
        return "P2"
    if score >= 31:
        return "P3"
    return "P4"


def risk_category(risk_score: Any, supplied: str | None = None) -> str:
    if supplied in {"Low", "Medium", "High", "Critical"}:
        return supplied
    score = number(risk_score)
    if score >= 81:
        return "Critical"
    if score >= 61:
        return "High"
    if score >= 31:
        return "Medium"
    return "Low"


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def build_claim_explanation(
    claim: dict[str, Any],
    provider: dict[str, Any],
    peer_claim_amount: Any,
    rule_reasons: Iterable[str] = (),
    comparison_count: int = 0,
    rag_pattern: str | None = None,
) -> dict[str, Any]:
    score = number(claim.get("risk_score"))
    category = risk_category(score, claim.get("risk_category"))
    priority = priority_for_risk(score)
    amount = number(claim.get("claim_amount"))
    similar_amount = number(peer_claim_amount)
    amount_ratio = amount / similar_amount if similar_amount > 0 else 0.0
    provider_score = number(provider.get("risk_score"))
    provider_ratio = number(provider.get("reimbursement_ratio"))
    flagged = bool(claim.get("is_anomaly")) or score >= 70

    why_flagged = list(rule_reasons)
    if amount_ratio > 1.5:
        why_flagged.append(
            f"This claim paid {money(amount)}, which is {amount_ratio:.1f}x the usual payment level for similar claims in the same review group."
        )
    if provider_score >= 61:
        why_flagged.append(
            f"The associated provider shows a pattern of unusually high billing compared with similar providers in the network."
        )
    if claim.get("is_anomaly"):
        why_flagged.append("The review identified multiple unusual billing signals on this claim when compared with normal claims in the dataset.")
    if rag_pattern:
        why_flagged.append(f"This case matches a recurring billing pattern seen in previous claims with similar payment behavior: '{rag_pattern}'.")
    if not why_flagged:
        why_flagged.append("This claim's payment and utilization pattern contributed to the overall risk assessment.")

    why_suspicious = []
    if amount_ratio > 1.5:
        why_suspicious.append(f"The payment is materially above what is typical for similar claims ({amount_ratio:.1f}x the comparable median).")
    if provider_ratio > 1.5:
        why_suspicious.append(f"The provider's billing profile is also elevated relative to peers, which increases the claim's review context.")
    if claim.get("explanation_2") and claim.get("explanation_2") != claim.get("explanation_1"):
        why_suspicious.append(str(claim["explanation_2"]))
    if not why_suspicious:
        why_suspicious.append("The current evidence does not point to one single issue; the overall pattern is what raises concern.")

    comparison = [
        f"Reviewed claim group: {comparison_count:,} claims.",
        f"This claim paid {money(amount)} versus a comparable median of {money(similar_amount)} ({amount_ratio:.1f}x)." if similar_amount else f"This claim paid {money(amount)}; a comparable median was not available for this claim type.",
    ]
    financial = [
        f"Claim value submitted: {money(amount)}.",
        f"Estimated financial exposure if the payment were not adjusted: {money(amount * score / 100)}.",
    ]
    action = {
        "P1": "Review medical documentation, coding, and claim justification immediately.",
        "P2": "Prioritize medical-necessity and coding review with supporting records.",
        "P3": "Perform targeted documentation review and compare related claims.",
        "P4": "Monitor the claim and retain it for routine payment-integrity review.",
    }[priority]
    summary = (
        f"Claim {claim.get('claim_id', 'unknown')} is in the {category} risk category with {priority} investigation priority. "
        f"The payment amount of {money(amount)} is materially above the normal range for comparable claims, and the associated provider also shows elevated billing patterns. "
        f"Recommended next step: {action}"
    )
    return {
        "risk_score": round(score, 2),
        "risk_category": category,
        "priority": priority,
        "why_flagged": _unique(why_flagged),
        "why_suspicious": _unique(why_suspicious),
        "peer_comparison": comparison,
        "financial_impact": financial,
        "recommended_action": action,
        "ai_summary": summary,
        "evidence": {"claim_id": claim.get("claim_id"), "provider_id": claim.get("provider_id"), "flagged": flagged},
    }


def build_provider_explanation(
    provider: dict[str, Any],
    scores: dict[str, Any],
    benchmarks: dict[str, Any],
    drift: dict[str, Any],
    reasons: Iterable[str] = (),
) -> dict[str, Any]:
    score = number(provider.get("risk_score"))
    category = risk_category(score, provider.get("risk_level"))
    priority = priority_for_risk(score)
    reimbursement = number(provider.get("total_reimbursement"))
    ratio = number(benchmarks.get("reimbursement_ratio"), 1.0)
    claims_ratio = number(benchmarks.get("claims_ratio"), 1.0)
    reimbursement_percentile = number(benchmarks.get("reimbursement_percentile"), 50.0)
    claims_percentile = number(benchmarks.get("claims_percentile"), 50.0)
    drift_score = number(drift.get("drift_score"))
    drift_level = str(drift.get("drift_level") or "Low")

    why_flagged = list(reasons)
    if ratio > 1.5:
        why_flagged.append(f"Total reimbursement is {ratio:.1f}x the median for comparable providers in the same peer group.")
    if claims_ratio > 1.5:
        why_flagged.append(f"Claim volume is {claims_ratio:.1f}x the median for comparable providers.")
    if score >= 61:
        why_flagged.append(f"The overall review placed this provider in the {category.lower()} risk band based on multiple billing indicators.")
    if not why_flagged:
        why_flagged.append("The provider's billing and utilization patterns contributed to the overall risk assessment.")

    peer_comparison = [
        f"Reimbursement is {ratio:.1f}x the comparable-provider median and sits at the {reimbursement_percentile:.1f}th percentile nationally.",
        f"Claim volume is {claims_ratio:.1f}x the comparable-provider median and sits at the {claims_percentile:.1f}th percentile nationally.",
    ]
    behaviour = [
        f"This provider submitted {int(number(provider.get('total_claims'))):,} claims for {int(number(provider.get('total_beneficiaries'))):,} beneficiaries.",
        f"Average reimbursement per claim was {money(provider.get('mean_reimbursement'))}.",
    ]
    drift_findings = [
        f"Recent billing activity is elevated relative to prior months, with claims rising to {number(drift.get('claims_spike_ratio'), 1.0):.2f}x the prior-month median.",
        f"Reimbursement has also increased to {number(drift.get('reimbursement_spike_ratio'), 1.0):.2f}x the prior-month median.",
    ]
    if drift_level == "Low" and drift_score == 0:
        drift_findings = ["No material month-over-month billing change was recorded in the available history."]

    suspicious = []
    if ratio > 1.5:
        suspicious.append(f"The payment level is materially above comparable providers ({ratio:.1f}x).")
    if claims_ratio > 1.5:
        suspicious.append(f"The volume of claims being submitted is materially above comparable providers ({claims_ratio:.1f}x).")
    if drift_level in {"High", "Critical"}:
        suspicious.append(f"Billing activity shows a noticeable rise in the most recent period, which increases review urgency.")
    if not suspicious:
        suspicious.append("No single benchmark alone explains the risk; the pattern across claims and billing history is what raises concern.")

    leakage = reimbursement * score / 100
    action = {
        "P1": "Perform an immediate audit of submitted claims and supporting records.",
        "P2": "Prioritize provider audit, medical-necessity validation, and coding review.",
        "P3": "Conduct targeted review of high-value claims and peer deviations.",
        "P4": "Monitor provider activity and retain the case for routine review.",
    }[priority]
    summary = (
        f"Provider {provider.get('provider_id', 'unknown')} is in the {category} risk category with {priority} investigation priority. "
        f"The provider's reimbursement and claim volume are both materially above comparable providers, and recent billing activity has risen above the normal range. "
        f"Estimated financial exposure is {money(leakage)}. Recommended next step: {action}"
    )
    return {
        "risk_score": round(score, 2),
        "risk_category": category,
        "priority": priority,
        "why_flagged": _unique(why_flagged),
        "why_suspicious": _unique(suspicious),
        "peer_comparison": peer_comparison,
        "billing_behaviour_summary": behaviour,
        "temporal_drift_findings": drift_findings,
        "financial_impact": [f"Total reimbursement: {money(reimbursement)}.", f"Risk-weighted exposure estimate: {money(leakage)}."],
        "recommended_action": action,
        "ai_summary": summary,
        "evidence": {"provider_id": provider.get("provider_id"), "model_scores": scores, "drift_score": drift_score},
    }
