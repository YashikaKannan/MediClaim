
import streamlit as st
import pandas as pd
from components.cards import card, close_card
from components.charts import model_donut
from components.risk_badges import status_badge, status_text

def render(g):
    st.title("Model Governance")
    st.caption("Transparency and oversight for payment integrity scoring, model operations, and data quality.")

    st.markdown('<div class="section-kicker">1 · Risk Scoring Framework</div>',unsafe_allow_html=True)
    card("Risk Tier Definitions","Read-only risk standardization.")
    st.dataframe(pd.DataFrame(g["risk_tiers"]),hide_index=True,use_container_width=True)
    st.markdown('<div class="note">These thresholds are applied consistently across dashboard, investigation queue, provider profiles, claim investigations, and exported reports.</div>',unsafe_allow_html=True)
    close_card()

    st.markdown('<div class="section-kicker">2 · Active Models</div>',unsafe_allow_html=True)
    card("Model Inventory","Read-only inventory of active and planned analytical models.")
    mdf=pd.DataFrame(g["models"])
    mdf["Status"]=mdf["Status"].map(status_text)
    st.dataframe(mdf,hide_index=True,use_container_width=True)
    close_card()

    st.markdown('<div class="section-kicker">3 · Model Weights</div>',unsafe_allow_html=True)
    a,b=st.columns([1,1])
    with a:
        card("GLOBAL MODEL WEIGHTS","System-level weights, not provider-level contribution.")
        w=pd.DataFrame(g["weights"])
        st.plotly_chart(model_donut(w),use_container_width=True)
        close_card()
    with b:
        card("Risk Score Composition")
        st.dataframe(pd.DataFrame(g["weights"]),hide_index=True,use_container_width=True)
        st.markdown("**Final Risk Score = 35% Peer Analysis + 25% Isolation Forest + 20% LOF + 20% Rule Engine**")
        close_card()

    st.markdown('<div class="section-kicker">4 · Data Sources</div>',unsafe_allow_html=True)
    cols=st.columns(3)
    for i,s in enumerate(g["data_sources"]):
        with cols[i%3]:
            card(s["Source Name"])
            st.metric("Records",f'{s["Records Count"]:,}')
            st.write(f'Last refresh: {s["Last Refresh Date"]}')
            st.markdown(status_badge(s["Status"]),unsafe_allow_html=True)
            close_card()

    st.markdown('<div class="section-kicker">5 · Data Quality Monitoring</div>',unsafe_allow_html=True)
    card("Data Quality Overview","Pipeline completeness and validation health.")
    q=g["quality"]
    cols=st.columns(5)
    for col,(label,value) in zip(cols,q.items()):
        with col:
            if label=="Data Quality Score":
                st.metric(label,f"{value}%")
                st.progress(value/100)
            else:
                st.metric(label,f"{value:,}")
    st.progress(q["Data Quality Score"]/100,text=f'Data Quality Score · {q["Data Quality Score"]}%')
    close_card()

    st.markdown('<div class="section-kicker">6 · Validation & Business Impact</div>',unsafe_allow_html=True)
    card("Business Validation Metrics","Operational validation indicators, not classification metrics.")
    b=g["business_validation"]
    cols=st.columns(5)
    for col,(label,value) in zip(cols,b.items()):
        with col:
            if "Amount" in label: col.metric(label,f'${value:,.0f}')
            elif "Rate" in label: col.metric(label,f'{value}%')
            else: col.metric(label,f'{value:,}')
    st.markdown('<div class="warning-note">LEIE overlap is used for business validation only and is not a formal precision, recall, or accuracy metric.</div>',unsafe_allow_html=True)
    close_card()

    st.markdown('<div class="section-kicker">7 · System Health</div>',unsafe_allow_html=True)
    card("Operational Status","Read-only operational health.")
    for row in g["system_health"]:
        a,b=st.columns([6,1])
        a.write(f'**{row["Component"]}**')
        b.markdown(f'<span class="health">● {row["Status"]}</span>',unsafe_allow_html=True)
    close_card()

    st.markdown('<div class="section-kicker">8 · Audit Log</div>',unsafe_allow_html=True)
    card("Recent System Activity","Recent governance-relevant system activity.")
    st.dataframe(pd.DataFrame(g["audit_log"]),hide_index=True,use_container_width=True)
    close_card()
