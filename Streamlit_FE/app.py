import streamlit as st
from components.styles import CSS
from components.auth import login_page
from utils.data_utils import load_all, enrich_queue
from views import dashboard, investigation_queue, provider_investigation, claim_investigation, reports, ai_assistant, model_governance, profile

st.set_page_config(page_title="MEDICLAIM | Payment Integrity", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

dashboard_data, providers, queue, claims, reports_data, governance, users = load_all()
queue_df = enrich_queue(queue, providers, claims)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "provider_id" not in st.session_state and providers:
    st.session_state.provider_id = providers[0]["id"]
if "claim_id" not in st.session_state and claims:
    st.session_state.claim_id = claims[0]["id"]

if not st.session_state.authenticated:
    login_page(users)
    st.stop()

NAV = [
    ("Dashboard","Dashboard","📊"),
    ("Investigation Queue","Investigation Queue","🔎"),
    ("Provider Investigations","Provider Investigations","👤"),
    ("Claim Investigations","Claim Investigations","📄"),
    ("Reports","Reports","📈"),
    ("AI Investigation Assistant","AI Investigation Assistant","✨"),
    ("Model Governance","Model Governance","🛡️"),
]
u = st.session_state.user

with st.sidebar:
    st.markdown(
        '<div class="brand-box"><span class="brand-mark"></span>'
        '<span class="brand-name">MEDICLAIM</span>'
        '<div class="brand-sub">Payment Integrity</div></div>',
        unsafe_allow_html=True
    )
    for label,key,icon in NAV:
        if st.button(
            f"{icon}  {label}",
            use_container_width=True,
            type="primary" if st.session_state.page == key else "secondary",
            key=f"nav_{key}"
        ):
            st.session_state.page = key
            st.rerun()

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">ACCOUNT</div>', unsafe_allow_html=True)

    if st.button(
        f"👤  {u['name']} · Profile",
        use_container_width=True,
        type="primary" if st.session_state.page == "Profile" else "secondary",
        key="open_profile"
    ):
        st.session_state.page = "Profile"
        st.rerun()

    if st.button("↪  Logout", use_container_width=True, key="logout"):
        st.session_state.authenticated = False
        st.session_state.page = "Dashboard"
        st.rerun()

page = st.session_state.page
if page == "Dashboard":
    dashboard.render(dashboard_data, queue_df)
elif page == "Investigation Queue":
    investigation_queue.render(queue_df)
elif page == "Provider Investigations":
    provider_investigation.render(providers, claims)
elif page == "Claim Investigations":
    claim_investigation.render(claims)
elif page == "Reports":
    reports.render(reports_data, queue_df)
elif page == "AI Investigation Assistant":
    ai_assistant.render(providers)
elif page == "Model Governance":
    model_governance.render(governance)
elif page == "Profile":
    profile.render(st.session_state.user)
