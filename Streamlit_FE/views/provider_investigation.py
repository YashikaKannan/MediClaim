
import streamlit as st
import pandas as pd
from components.cards import card, close_card
from components.risk_badges import risk_badge, risk_text
from models.model_adapter import generate_shap_explanation

def render(providers, claims):
    p = next(
        (x for x in providers if x["id"] == st.session_state.get("provider_id")),
        providers[0]
    )

    st.title("Provider Investigation Profile")
    st.caption("Evidence-led provider assessment, peer comparison, explainability, and financial impact.")

    st.markdown(
        f'<div class="hero"><h2>{p["name"]}</h2>'
        f'<div>NPI: {p["npi"]} &nbsp; • &nbsp; {p["location"]}</div>'
        f'<div style="margin-top:12px">{risk_badge(p["riskScore"])}</div></div>',
        unsafe_allow_html=True
    )

    for col,(lab,val) in zip(
        st.columns(5),
        [
            ("Provider ID",p["id"]),
            ("Total Claims",f'{p["totalClaims"]:,}'),
            ("Total Paid",f'${p["totalPaid"]:,.0f}'),
            ("Potential Excess Payment",f'${p.get("potentialExcessPayment",468980):,.0f}'),
            ("Risk Score",p["riskScore"])
        ]
    ):
        col.metric(lab,val)

    card("Why Was This Provider Flagged", "Every risk driver includes source and operational status.")
    for r in p["reasons"]:
        a,b,c = st.columns([5,2,1])
        a.write(f'**{r["reason"]}**')
        b.write(f'Source: {r["source"]}')
        c.write(r["status"])
    close_card()

    card("Peer Analysis", "Comparison against a relevant provider peer group using raw values.")
    rows = []
    for x in p.get("peerMetrics",[]):
        rows.append({
            "Metric":x["metric"],
            "Provider":x.get("providerDisplay",x.get("provider")),
            "Peer Median":x.get("peerDisplay",x.get("peerMedian")),
            "Peer Percentile":x.get("peerPercentile","94th"),
            "Difference":x.get("difference","+")
        })
    if rows:
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

    pg = p.get("peerGroup", {
        "region":p["location"].split(",")[-1].strip(),
        "peerCount":4102,
        "threshold":"Met",
        "confidence":"High"
    })
    for col,(lab,val) in zip(
        st.columns(4),
        [
            ("Region",pg.get("region","—")),
            ("Peer Count",f'{pg.get("peerCount",0):,}'),
            ("Threshold",pg.get("threshold","Met")),
            ("Confidence",pg.get("confidence","High"))
        ]
    ):
        col.metric(lab,val)
    close_card()

    card(
        "Financial Impact",
        f'<h2>${p.get("potentialExcessPayment",468980):,.0f}</h2>'
        '<div class="small">Source: POTENTIAL_EXCESS_AMOUNT</div>'
    )
    close_card()

    card("Provider Score Breakdown", "Provider-specific contribution. This is different from GLOBAL MODEL WEIGHTS.")
    sb = pd.DataFrame([
        {"Feature":x["label"],"Contribution":x["value"]}
        for x in p.get("scoreBreakdown",[]) if x["label"]!="Final Score"
    ])
    if not sb.empty:
        st.bar_chart(sb.set_index("Feature")["Contribution"])
    st.markdown(f'**Final Score: {p["riskScore"]} — {p.get("riskLabel","High Risk")}**')
    close_card()

    card("SHAP-Based Explainability", "Top drivers shown as explanation data. Connect the real explainer before production use.")
    st.markdown('<span class="badge blue">DEMO EXPLANATION DATA</span>',unsafe_allow_html=True)
    st.dataframe(generate_shap_explanation(p),hide_index=True,use_container_width=True)
    close_card()

    associated = [c for c in claims if c["id"] in p.get("associatedClaims",[])]
    card("Associated Claims", "Drill from provider evidence to claim-level investigation.")
    if associated:
        cdf = pd.DataFrame(associated)
        cdf["Risk"] = cdf["riskScore"].map(risk_text)
        st.dataframe(
            cdf[["id","serviceDate","claimAmount","paidAmount","Risk"]].rename(
                columns={"id":"Claim ID","serviceDate":"Service Date",
                         "claimAmount":"Claim Amount","paidAmount":"Paid Amount"}
            ),
            hide_index=True,use_container_width=True
        )
        selected = st.selectbox("Open Claim",[c["id"] for c in associated])
        if st.button("Open Claim",type="primary",icon=":material/visibility:"):
            st.session_state.claim_id=selected
            st.session_state.page="Claim Investigations"
            st.rerun()
    else:
        st.info("No associated mock claims.")
    close_card()
