
import streamlit as st
import pandas as pd
from components.cards import card, close_card
from components.risk_badges import risk_badge, status_badge
from models.model_adapter import generate_shap_explanation

def render(claims):
    c = next((x for x in claims if x["id"] == st.session_state.get("claim_id")), claims[0])
    st.title("Claim Investigation Profile")
    st.caption("Claim-level evidence, procedure breakdown, risk explanation, and financial impact.")

    cols=st.columns(7)
    vals=[("Claim ID",c["id"]),("Provider",c["providerName"]),("Beneficiary",c["beneficiaryId"]),
          ("Claim Amount",f'${c["claimAmount"]:,.0f}'),("Paid Amount",f'${c["paidAmount"]:,.0f}'),
          ("Service Date",c["serviceDate"]),("Risk Score",str(c["riskScore"]))]
    for col,(lab,val) in zip(cols,vals): col.metric(lab,val)
    st.markdown(risk_badge(c["riskScore"]),unsafe_allow_html=True)

    card("Why Was This Claim Flagged","Evidence is linked to its detection source.")
    for r in c["reasons"]:
        a,b,d=st.columns([5,3,1])
        a.write(f'**{r["reason"]}**')
        b.write(f'Source: {r["source"]}')
        d.markdown(status_badge(r["status"]),unsafe_allow_html=True)
    close_card()

    card("SHAP-Based Claim Explanation","Top contributing claim-level features.")
    st.markdown('<span class="badge blue">DEMO EXPLANATION DATA</span>',unsafe_allow_html=True)
    st.dataframe(generate_shap_explanation(c),hide_index=True,use_container_width=True)
    close_card()

    card("Procedure Breakdown","HCPCS-level payment evidence.")
    pdf=pd.DataFrame(c.get("procedureBreakdown",[]))
    if not pdf.empty:
        pdf["Paid Amount"]=pdf["paidAmount"].map(lambda x:f'${x:,.0f}')
        st.dataframe(pdf.rename(columns={"hcpcs":"HCPCS","description":"Description","units":"Units","flag":"Flag"}),hide_index=True,use_container_width=True)
    close_card()

    card("Potential Excess Payment",f'<h2>${c.get("potentialExcessPayment",0):,.0f}</h2>')
    close_card()
