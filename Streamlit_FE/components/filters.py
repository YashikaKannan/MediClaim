
import streamlit as st

def apply_investigation_filters(df):
    st.markdown("### Investigator-Centric Search & Filtering")
    a,b,c = st.columns(3)

    with a:
        query = st.text_input(
            "Search Provider ID / NPI / Claim ID / Name",
            placeholder="e.g. PRV-000124"
        )
        risk = st.multiselect(
            "Risk Level",
            ["Low Risk","Medium Risk","High Risk","Critical Risk"]
        )
        state = st.multiselect(
            "State",
            sorted(df["state"].dropna().unique().tolist())
        )

    with b:
        status = st.multiselect(
            "Status",
            sorted(df["status"].dropna().unique().tolist())
        )
        reason = st.multiselect(
            "Detection Reason",
            sorted(df["detection_reason"].dropna().unique().tolist())
        )
        investigator = st.multiselect(
            "Investigator",
            sorted(df["assigned_investigator"].dropna().unique().tolist())
        )

    with c:
        score = st.slider("Risk Score", 0, 100, (0,100))
        amount = st.number_input(
            "Minimum Claim Amount ($)", min_value=0.0, value=0.0, step=100.0
        )
        provider_type = st.multiselect(
            "Provider Type",
            sorted(df["provider_type"].dropna().unique().tolist())
        )

    out = df.copy()
    if query:
        q = query.lower()
        mask = out.astype(str).apply(
            lambda col: col.str.lower().str.contains(q, na=False)
        ).any(axis=1)
        out = out[mask]
    if risk: out = out[out["risk_label"].isin(risk)]
    if state: out = out[out["state"].isin(state)]
    if status: out = out[out["status"].isin(status)]
    if reason: out = out[out["detection_reason"].isin(reason)]
    if investigator: out = out[out["assigned_investigator"].isin(investigator)]
    if provider_type: out = out[out["provider_type"].isin(provider_type)]
    out = out[(out["risk_score"] >= score[0]) & (out["risk_score"] <= score[1])]
    out = out[out["claim_amount"] >= amount]

    st.caption(f"{len(out):,} matching records")
    return out
