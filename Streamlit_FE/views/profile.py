
import streamlit as st

PROFILE_DETAILS = {
    "Hannah Smith": {"title":"Senior SIU Investigator","department":"Special Investigations Unit","organization":"Payment Integrity Operations","location":"Chicago, IL","work_email":"hannah.smith@mediclaim.demo","experience":"8 years","focus":"Provider & claim investigations","access":"Investigator"},
    "Michael Chen": {"title":"Payment Integrity Analyst","department":"Medicare Payment Integrity","organization":"Payment Integrity Operations","location":"Washington, DC","work_email":"michael.chen@mediclaim.demo","experience":"6 years","focus":"Payment integrity analytics","access":"Investigator"},
    "Alicia Stone": {"title":"Claims Audit Lead","department":"Claims Audit","organization":"Audit & Compliance","location":"Dallas, TX","work_email":"alicia.stone@mediclaim.demo","experience":"7 years","focus":"Claims review & audit","access":"Auditor"},
    "Daniel Brooks": {"title":"Fraud Intelligence Analyst","department":"Fraud Analytics","organization":"Payment Integrity Operations","location":"New York, NY","work_email":"daniel.brooks@mediclaim.demo","experience":"5 years","focus":"Fraud analytics & anomaly detection","access":"Analyst"},
    "Priya Rao": {"title":"Fraud Investigation Manager","department":"Insurance Fraud Investigation","organization":"Fraud & Payment Integrity","location":"Boston, MA","work_email":"priya.rao@mediclaim.demo","experience":"9 years","focus":"Fraud investigation oversight","access":"Investigator"},
}

def render(user):
    d = PROFILE_DETAILS.get(user["name"], {
        "title":"Payment Integrity Professional","department":user.get("team",""),
        "organization":"MEDICLAIM","location":"United States",
        "work_email":f'{user["name"].lower().replace(" ",".")}@mediclaim.demo',
        "experience":"—","focus":"Payment integrity","access":"Investigator"
    })

    st.title("My Profile")
    st.caption("Your investigator identity, professional context, and access profile.")

    st.markdown(f"""
    <div class="profile-hero">
      <div class="profile-avatar">{user["name"][0]}</div>
      <div>
        <div class="profile-name">{user["name"]}</div>
        <div class="profile-title">{d["title"]}</div>
        <div class="profile-team">{user["team"]}</div>
      </div>
      <div class="profile-live"><span>●</span> Active session</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Professional profile</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.metric("User ID", user["user_id"])
    b.metric("Access Level", d["access"])
    c.metric("Experience", d["experience"])

    # Information is intentionally rendered directly on the page,
    # without the large surrounding white containers from the previous version.
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="profile-section-title">Contact & Organization</div>', unsafe_allow_html=True)
        for label,value in [
            ("Work Email", d["work_email"]),
            ("Department", d["department"]),
            ("Organization", d["organization"]),
            ("Location", d["location"])
        ]:
            st.markdown(
                f'<div class="profile-row-clean"><span>{label}</span><b>{value}</b></div>',
                unsafe_allow_html=True
            )

    with right:
        st.markdown('<div class="profile-section-title">Professional Context</div>', unsafe_allow_html=True)
        for label,value in [
            ("Role", user["role"]),
            ("Team", user["team"]),
            ("Primary Focus", d["focus"]),
            ("Session Status", "Authenticated")
        ]:
            st.markdown(
                f'<div class="profile-row-clean"><span>{label}</span><b>{value}</b></div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-kicker">Access & workspace</div>', unsafe_allow_html=True)
    x,y,z = st.columns(3, gap="medium")
    with x:
        st.markdown('<div class="profile-mini"><div>🔎</div><b>Investigation Access</b><span>Provider and claim evidence</span></div>', unsafe_allow_html=True)
    with y:
        st.markdown('<div class="profile-mini"><div>🛡️</div><b>Governance Access</b><span>Read-only model transparency</span></div>', unsafe_allow_html=True)
    with z:
        st.markdown('<div class="profile-mini"><div>📄</div><b>Report Access</b><span>Investigation exports</span></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="profile-note">Profile details are read-only in this prototype. '
        'Production identity, authorization, and profile data should be sourced from the client enterprise identity platform.</div>',
        unsafe_allow_html=True
    )
