
import streamlit as st
from components.filters import apply_investigation_filters
from components.risk_badges import risk_text

def render(queue_df):
    st.title("Investigation Queue")
    st.caption("Narrow thousands of claims and providers into a focused set of potentially suspicious cases.")

    df = apply_investigation_filters(queue_df)
    if df.empty:
        st.warning("No records match the current search and filters.")
        return

    view = df.sort_values("priority_score", ascending=False).copy()
    view["Risk"] = view["risk_score"].map(risk_text)
    view["Potential Leakage"] = view["potential_leakage"].map(lambda x:f"${x:,.0f}")

    st.dataframe(
        view[[
            "priority_rank","provider_name","provider_id","npi","provider_type",
            "Risk","risk_score","Potential Leakage","detection_reason",
            "status","assigned_investigator"
        ]].rename(columns={
            "priority_rank":"Rank","provider_name":"Provider Name","provider_id":"Provider ID",
            "npi":"NPI","provider_type":"Provider Type","risk_score":"Risk Score",
            "detection_reason":"Detection Reason","status":"Status",
            "assigned_investigator":"Investigator"
        }),
        hide_index=True, use_container_width=True
    )

    choices = {
        f'{r["provider_name"]} — {r["provider_id"]}':r["provider_id"]
        for _,r in df.iterrows()
    }
    selected = st.selectbox("Open Provider Investigation", list(choices))
    if st.button(
        "Open Investigation",
        type="primary",
        icon=":material/visibility:"
    ):
        st.session_state.provider_id = choices[selected]
        st.session_state.page = "Provider Investigations"
        st.rerun()
