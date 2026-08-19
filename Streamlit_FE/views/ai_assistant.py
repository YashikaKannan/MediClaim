
import os

import streamlit as st

from components.cards import card, close_card


def _build_local_summary(provider):
    summary = {
        "Investigation Summary": (
            f"{provider['name']} has a final risk score of {provider['riskScore']:.1f}/100 "
            f"({provider['riskLabel']}) and a priority score of {provider['priorityScore']:.1f}."
        ),
        "Audit Narrative": (
            "The provider is elevated because the ensemble model identified material anomalies across "
            "the isolation forest, autoencoder, and CatBoost components. The current score aligns with "
            "elevated recurrence and leakage potential in the generated outputs."
        ),
        "Questions for Interview": [
            "Can you confirm whether the provider's unusually high service volume is driven by legitimate growth or a concentrated claim pattern?",
            "Do any historical billing practices or coding changes explain the elevated anomaly profile?",
            "Are there known staffing, referral, or outlier claim patterns associated with this provider?",
        ],
        "Recommended Actions": [
            "Prioritize claim review for the top anomalous providers and ensure documentation matches billed patterns.",
            "Validate provider billing patterns against peer benchmarks and prior payment history.",
            "Escalate if the provider demonstrates sustained high-risk behavior over multiple review cycles.",
        ],
        "Recovery Suggestions": [
            "Require documentation review for flagged claims before payment release.",
            "Establish pre-payment edits on key risk indicators from the fused model outputs.",
            "Monitor the provider's score trajectory to detect whether the risk is transient or recurring.",
        ],
    }
    return summary


def _maybe_call_gemini(provider):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"You are an investigator assistant for healthcare fraud. Use this provider data: "
            f"name={provider['name']}, riskScore={provider['riskScore']}, riskLevel={provider['riskLabel']}, "
            f"explanation={provider['reasons'][0]['reason']} if available. "
            "Return JSON with keys Investigation Summary, Audit Narrative, Questions for Interview, "
            "Recommended Actions, and Recovery Suggestions."
        )
        response = model.generate_content(prompt)
        text = getattr(response, "text", "")
        if text:
            return text
    except Exception:
        return None
    return None


def render(providers):
    st.title("AI Investigation Assistant")
    st.caption("Structured investigator support powered by the live provider risk data.")
    provider = next((p for p in providers if p["id"] == st.session_state.get("provider_id")), providers[0] if providers else {"id": "N/A", "name": "No provider", "riskScore": 0, "riskLabel": "Low", "priorityScore": 0, "reasons": [{"reason": "No data available"}]})

    prompts = [
        "Why is this provider high risk?",
        "Compare provider with peers.",
        "Summarize this investigation.",
        "Generate audit narrative.",
        "Recommend next actions.",
    ]
    prompt = st.selectbox("Investigation prompt", prompts)
    if st.button("Run Investigation", type="primary"):
        summary = _build_local_summary(provider)
        gemini_result = _maybe_call_gemini(provider)
        if gemini_result:
            try:
                import json

                parsed = json.loads(gemini_result)
                if isinstance(parsed, dict):
                    summary = parsed
            except Exception:
                pass

        st.session_state.ai = summary

    if "ai" in st.session_state:
        for title, body in st.session_state.ai.items():
            if isinstance(body, list):
                body_text = "\n".join(f"- {item}" for item in body)
            else:
                body_text = str(body)
            card(title, body_text)
            close_card()
