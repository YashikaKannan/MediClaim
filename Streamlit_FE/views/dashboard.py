
import streamlit as st
import pandas as pd
from components.cards import kpi, card, close_card
from components.charts import risk_donut, bar, model_donut
from components.risk_badges import risk_text

def render(dashboard_data, queue_df):
    st.title("Dashboard")
    st.caption("Executive overview of payment integrity risk, financial impact, and investigation priority.")

    values = [
        ("Providers Analyzed", f'{dashboard_data["providersAnalyzed"]:,}', "#1976D2"),
        ("Claims Analyzed", f'{dashboard_data["claimsAnalyzed"]:,}', "#1257A6"),
        ("Flagged Providers", f'{dashboard_data["flaggedProviders"]:,}', "#F97316"),
        ("Potential Financial Leakage", f'${dashboard_data["potentialFinancialLeakage"]/1e6:.1f}M', "#00A6D6"),
        ("High Risk Claims", "3,421", "#DC2626"),
        ("Average Risk Score", "54.8", "#062B5C")
    ]
    for col,(label,value,accent) in zip(st.columns(6), values):
        with col:
            kpi(label,value,accent)

    st.markdown('<div class="section-kicker">Risk intelligence</div>', unsafe_allow_html=True)
    a,b = st.columns([1,1.15])

    with a:
        card("Risk Distribution", "Portfolio segmentation with count and percentage.")
        st.plotly_chart(
            risk_donut(dashboard_data["riskDistribution"]),
            use_container_width=True,
            config={"displayModeBar":False}
        )
        close_card()

    with b:
        card("Financial Leakage Overview", "Financial exposure by operational dimension.")
        tabs = st.tabs(["By Provider Type","By State","By Risk Tier","By Detection Reason"])
        datasets = [
            dashboard_data.get("leakageByProviderType", []),
            dashboard_data.get("leakageByState", []),
            dashboard_data.get("riskDistribution", []),
            [{"label":"Peer anomaly","value":82},
             {"label":"Duplicate billing","value":64},
             {"label":"Exclusion match","value":38}]
        ]
        for tab,data in zip(tabs,datasets):
            with tab:
                st.plotly_chart(
                    bar(pd.DataFrame(data),"label","value",""),
                    use_container_width=True,
                    config={"displayModeBar":False}
                )
        close_card()

    st.markdown('<div class="section-kicker">Investigation priority</div>', unsafe_allow_html=True)
    a,b = st.columns([1.5,.75])

    with a:
        card("Top Priority Investigations", "Cases ranked by risk severity and potential financial impact.")
        q = queue_df.sort_values("priority_score", ascending=False).head(5).copy()
        q["Risk"] = q["risk_score"].map(risk_text)
        q["Potential Leakage"] = q["potential_leakage"].map(lambda x:f"${x:,.0f}")
        st.dataframe(
            q[["priority_rank","provider_name","provider_id","Risk","risk_score",
               "Potential Leakage","status"]].rename(
                columns={"risk_score":"Risk Score","status":"Status"}
            ),
            hide_index=True, use_container_width=True
        )
        st.markdown(
            '<div class="note">Priority Score = Risk Score × Potential Leakage.</div>',
            unsafe_allow_html=True
        )
        close_card()

    with b:
        card("Global Model Weights", "System-level model composition.")
        st.plotly_chart(
            model_donut(pd.DataFrame({
                "Model":["Peer Analysis","Isolation Forest","LOF","Rule Engine"],
                "Weight":[35,25,20,20]
            })),
            use_container_width=True,
            config={"displayModeBar":False}
        )
        close_card()

    st.markdown('<div class="section-kicker">Operational health</div>', unsafe_allow_html=True)
    card("System Health", "Pipeline readiness indicators.")
    for col,label in zip(
        st.columns(5),
        ["Provider Pipeline","Claim Pipeline","Risk Engine","Reports","AI Assistant"]
    ):
        with col:
            st.markdown(
                f'<div style="text-align:center"><div style="font-size:1.35rem;color:#16A34A">●</div>'
                f'<b>{label}</b><div class="small" style="color:#16A34A">Healthy</div></div>',
                unsafe_allow_html=True
            )
    close_card()
