
import streamlit as st

ROLES = [
    "Insurance Company Fraud Investigation Team",
    "Medicare Payment Integrity Team",
    "Claims Audit Team",
    "SIU Investigator",
    "Fraud Analyst",
]


def login_page(users):
    # This login intentionally uses Streamlit-native content for the visible UI.
    # No HTML markup is used for the hero or form, so HTML source can never appear
    # as visible text on the login screen.

    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #F4F8FC !important;
    }

    [data-testid="stAppViewContainer"] .main {
        overflow: hidden !important;
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1180px !important;
        padding: 8px 16px 6px !important;
        margin: 0 auto !important;
    }

    [data-testid="stVerticalBlock"] {
        gap: 0.25rem !important;
    }

    /* Desktop: both panels occupy the same viewport. */
    .st-key-login_left {
        height: calc(100vh - 28px) !important;
        max-height: calc(100vh - 28px) !important;
        min-height: 560px !important;
        box-sizing: border-box !important;
        padding: 34px 38px !important;
        border-radius: 20px !important;
        color: white !important;
        overflow: hidden !important;
        background:
            radial-gradient(circle at 88% 10%, rgba(55,205,255,.38), transparent 27%),
            radial-gradient(circle at 7% 92%, rgba(80,150,255,.25), transparent 30%),
            linear-gradient(145deg, #041E43 0%, #073D7D 56%, #0B72B5 100%);
        box-shadow: 0 14px 38px rgba(6,43,92,.14) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }

    .st-key-login_right {
        height: calc(100vh - 28px) !important;
        max-height: calc(100vh - 28px);
        min-height: 560px !important;
        box-sizing: border-box !important;
        padding: 32px 38px !important;
        border: 1px solid #D9E7F5;
        border-radius: 20px;
        background: white;
        box-shadow: 0 14px 38px rgba(6,43,92,.08);
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
    }

    /* Style the native Streamlit text used by the hero. */
    .st-key-login_left h1 {
        color: #FFFFFF !important;
        font-size: 2.35rem !important;
        line-height: 1.08 !important;
        margin: 8px 0 12px !important;
        font-weight: 850 !important;
    }

    .hero-panel h3 {
        color: white !important;
        margin: 0 0 26px !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
    }

    .hero-panel p {
        color: rgba(255,255,255,.80) !important;
        font-size: .80rem !important;
        line-height: 1.55 !important;
    }

    .hero-panel [data-testid="stMarkdownContainer"] {
        color: white;
    }

    .st-key-login_right h1,
    .st-key-login_right h2,
    .st-key-login_right h3 {
        color: #062B5C !important;
    }

    .st-key-login_right [data-testid="stTextInput"] label,
    .st-key-login_right [data-testid="stSelectbox"] label {
        color: #36536F !important;
        font-size: .68rem !important;
        font-weight: 750 !important;
        margin-bottom: 2px !important;
    }

    .st-key-login_right [data-testid="stTextInput"] input {
        height: 38px !important;
        min-height: 38px !important;
        font-size: .76rem !important;
    }

    .st-key-login_right [data-testid="stSelectbox"] div[data-baseweb="select"] {
        min-height: 38px !important;
        height: 38px !important;
        font-size: .74rem !important;
    }

    .st-key-login_right [data-testid="stButton"] button {
        height: 38px !important;
        min-height: 38px !important;
        border-radius: 8px !important;
        font-size: .72rem !important;
        font-weight: 750 !important;
    }

    /* Clear MEDICLAIM wordmark — matched to the supplied reference image. */
    .st-key-login_left h2 {
        color: #FFFFFF !important;
        font-size: 1.30rem !important;
        line-height: 1 !important;
        font-weight: 850 !important;
        letter-spacing: .01em !important;
        margin: 0 !important;
    }

    .st-key-login_left h3 {
        color: #8FDCFF !important;
        font-size: 1.95rem !important;
        line-height: 1 !important;
        margin: 0 !important;
        font-weight: 900 !important;
    }

    .st-key-login_left [data-testid="stCaptionContainer"] {
        color: rgba(255,255,255,.82) !important;
        font-size: .66rem !important;
        font-weight: 650 !important;
        margin-top: -2px !important;
    }

    .st-key-login_left [data-testid="stCaptionContainer"] {
        color: rgba(255,255,255,.80) !important;
    }

    .st-key-login_left [data-testid="stMarkdownContainer"] {
        color: #FFFFFF !important;
    }

    .st-key-login_left [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
    }

    .st-key-login_right [data-testid="stAlert"] {
        padding: 7px 10px !important;
        margin: 5px 0 8px !important;
        font-size: .65rem !important;
    }

    .st-key-login_right [data-testid="stCaptionContainer"] {
        color: #52718E !important;
        font-size: .60rem !important;
    }

    @media (min-width: 851px) and (max-height: 760px) {
        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 4px !important;
            padding-bottom: 3px !important;
        }

        .st-key-login_left, .st-key-login_right {
            height: calc(100vh - 18px) !important;
            max-height: calc(100vh - 18px) !important;
        }

        .st-key-login_left {
            min-height: 500px;
            padding: 25px 30px;
        }

        .st-key-login_right {
            min-height: 500px !important;
            padding: 23px 30px !important;
        }

        .hero-panel h1 {
            font-size: 2rem !important;
        }
    }

    @media (max-width: 850px) {
        [data-testid="stAppViewContainer"] .main {
            overflow: visible !important;
        }

        .st-key-login_left, .st-key-login_right {
            height: auto !important;
            max-height: none !important;
            min-height: 0 !important;
            overflow: visible !important;
        }

        .st-key-login_left {
            min-height: 390px;
            padding: 30px;
        }

        .st-key-login_right {
            padding: 28px 24px 30px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="small")

    with left:
        # Native Streamlit container: NO visible HTML markup.
        with st.container(key="login_left", border=False):
            brand_icon, brand_name = st.columns([0.18, 0.82], gap="small")
            with brand_icon:
                st.markdown("### ⚡")
            with brand_name:
                st.markdown("## MEDICLAIM")
                st.caption("Payment Integrity")

            st.markdown("### Payment Integrity Intelligence")
            st.markdown("# Investigate smarter. Protect every payment.")
            st.write(
                "A secure investigator workspace for identifying anomalous providers, "
                "prioritizing suspicious claims, understanding risk drivers, and "
                "supporting payment-integrity decisions."
            )

            f1, f2 = st.columns(2, gap="small")
            with f1:
                st.markdown("◈ Risk Prioritization")
                st.markdown("◉ Explainable AI")
            with f2:
                st.markdown("⌁ Peer Analysis")
                st.markdown("✓ Audit Ready")
    with right:
        with st.container(key="login_right", border=False):
            st.markdown("# Welcome back")
            st.caption("Sign in to your MEDICLAIM investigator workspace.")

            st.text_input(
                "User ID / Email",
                value="hannah.smith@mediclaim.demo",
                placeholder="name@organization.com",
                key="login_email",
            )

            st.text_input(
                "Password",
                type="password",
                value="demo",
                placeholder="Enter password",
                key="login_password",
            )

            role = st.selectbox(
                "Investigator Role",
                ROLES,
                key="login_role",
            )

            st.info(f"ACCESS PROFILE  ·  {role}")

            b1, b2 = st.columns(2, gap="small")
            with b1:
                sign_in = st.button(
                    "Sign In  →",
                    type="primary",
                    use_container_width=True,
                    key="login_sign_in",
                )
            with b2:
                demo = st.button(
                    "Demo Access",
                    use_container_width=True,
                    key="login_demo",
                )

            if sign_in or demo:
                match = next((u for u in users if u["role"] == role), users[0])
                st.session_state.authenticated = True
                st.session_state.user = match
                st.session_state.page = "Dashboard"
                st.rerun()

            st.caption(
                "🔒 Prototype security — Demo authentication is enabled for the "
                "hackathon prototype. Production deployment should connect this "
                "interface to the client's enterprise identity provider and "
                "access-control system."
            )
