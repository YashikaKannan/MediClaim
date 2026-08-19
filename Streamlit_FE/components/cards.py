
import streamlit as st

def card(title, subtitle=None):
    extra = f'<div class="card-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="card"><div class="card-title">{title}</div>{extra}', unsafe_allow_html=True)

def close_card():
    st.markdown('</div>', unsafe_allow_html=True)

def kpi(label, value, accent="#1976D2", help_text=None):
    help_html = f'<div class="small">{help_text}</div>' if help_text else ""
    st.markdown(
        f'<div class="kpi"><div class="kpi-accent" style="background:{accent}"></div>'
        f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{help_html}</div>',
        unsafe_allow_html=True
    )
