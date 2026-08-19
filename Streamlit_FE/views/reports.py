
import streamlit as st
from utils.export_utils import dataframe_csv, dataframe_excel, simple_pdf

def render(reports_data, queue_df):
    # Reports-only presentation update.
    # Existing report data and export functionality are unchanged.
    st.title("Reports")
    st.caption("Generate and download investigation reports from the MEDICLAIM workspace.")

    report_names = [r["name"] for r in reports_data]

    if not report_names:
        st.info("No report templates are available.")
        return

    if "selected_report_name" not in st.session_state:
        st.session_state.selected_report_name = report_names[0]
    if st.session_state.selected_report_name not in report_names:
        st.session_state.selected_report_name = report_names[0]

    st.markdown("""
    <style>
    /* Reports page: intentionally no surrounding white boxes */
    .reports-label{
        color:#062B5C;
        font-size:.68rem;
        font-weight:800;
        letter-spacing:.12em;
        text-transform:uppercase;
        margin:34px 0 8px;
    }

    .reports-heading{
        color:#102A43;
        font-size:1.05rem;
        font-weight:750;
        margin:0 0 12px;
    }

    .report-description{
        background:#F5F9FE;
        border:1px solid #E1EAF3;
        border-radius:10px;
        padding:11px 13px;
        color:#526D88;
        font-size:.78rem;
        margin-top:12px;
    }

    /* Remove Streamlit radio's default enclosing visual treatment */
    div[data-testid="stRadio"] > div{
        gap:8px;
    }

    div[data-testid="stRadio"] label{
        background:#F8FAFD;
        border:1px solid #E1EAF3;
        border-radius:10px;
        padding:10px 12px !important;
        min-height:42px;
        transition:all .15s ease;
    }

    div[data-testid="stRadio"] label:hover{
        background:#EFF6FF;
        border-color:#BFD8F2;
    }

    div[data-testid="stRadio"] label:has(input:checked){
        background:#0B1226;
        border-color:#0B1226;
        box-shadow:0 4px 10px rgba(11,18,38,.14);
    }

    div[data-testid="stRadio"] label:has(input:checked) p{
        color:#FFFFFF !important;
        font-weight:700;
    }

    div[data-testid="stRadio"] label > div:first-child{
        display:none;
    }

    .download-title{
        color:#102A43;
        font-size:1.05rem;
        font-weight:750;
        margin:0 0 12px;
    }

    /* Keep export buttons compact like the reference design */
    div[data-testid="stDownloadButton"] button{
        border-radius:9px;
        border:1px solid #F1D9B4;
        background:#FFF8ED;
        color:#9A5B08;
        font-weight:700;
        min-height:42px;
    }

    div[data-testid="stDownloadButton"] button:hover{
        border-color:#E9B96E;
        background:#FFF1D9;
        color:#7C4700;
    }
    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.35, .78], gap="large")

    with left:
        st.markdown('<div class="reports-label">Templates</div>', unsafe_allow_html=True)
        st.markdown('<div class="reports-heading">Investigation Reports</div>', unsafe_allow_html=True)

        selected = st.radio(
            "Investigation Reports",
            report_names,
            index=report_names.index(st.session_state.selected_report_name),
            key="reports_template_selector",
            label_visibility="collapsed",
        )
        st.session_state.selected_report_name = selected

        selected_report = next(
            r for r in reports_data if r["name"] == selected
        )

        st.markdown(
            f'<div class="report-description">{selected_report["description"]}</div>',
            unsafe_allow_html=True
        )

    with right:
        st.markdown('<div class="reports-label">Export</div>', unsafe_allow_html=True)
        st.markdown('<div class="download-title">Download Package</div>', unsafe_allow_html=True)

        df = queue_df.copy()

        csv = dataframe_csv(df)
        excel = dataframe_excel(df, selected_report["name"])
        pdf = simple_pdf(
            selected_report["name"],
            [
                "Generated for MEDICLAIM investigation workspace.",
                f"Records represented: {len(df)}",
                "Report uses the same visible queue data; no disconnected values are generated."
            ]
        )

        e1, e2, e3 = st.columns(3, gap="small")

        with e1:
            st.download_button(
                "PDF",
                pdf,
                file_name=f'{selected_report["id"]}.pdf',
                mime="application/pdf",
                use_container_width=True,
                key="reports_pdf"
            )

        with e2:
            st.download_button(
                "Excel",
                excel,
                file_name=f'{selected_report["id"]}.xlsx',
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="reports_excel"
            )

        with e3:
            st.download_button(
                "CSV",
                csv,
                file_name=f'{selected_report["id"]}.csv',
                mime="text/csv",
                use_container_width=True,
                key="reports_csv"
            )
