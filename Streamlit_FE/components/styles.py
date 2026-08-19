CSS = r'''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --navy:#062B5C; --blue:#1257A6; --bright:#1976D2; --cyan:#00A6D6;
  --light:#EAF4FF; --bg:#F4F9FF; --white:#FFFFFF; --text:#102A43;
  --muted:#607D9B; --border:#D9E7F5; --green:#16A34A; --amber:#F59E0B;
  --orange:#F97316; --red:#DC2626;
}
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { max-width: 1500px; padding: 1.2rem 2rem 2.5rem; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #062B5C 0%, #0B3A78 55%, #1257A6 100%);
  border-right: 0;
}
section[data-testid="stSidebar"] * { color: white; }
section[data-testid="stSidebar"] .stButton button {
  border: 1px solid rgba(255,255,255,.12);
  background: transparent;
  color: white;
  text-align: left;
  border-radius: 9px;
  min-height: 42px;
}
section[data-testid="stSidebar"] .stButton button:hover {
  background: rgba(255,255,255,.10);
  border-color: rgba(255,255,255,.20);
}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {
  background: #1976D2;
  border-color: #4AA4FF;
  box-shadow: 0 5px 16px rgba(0,0,0,.18);
}
h1,h2,h3 { color: var(--navy) !important; letter-spacing:-.035em; }
h1 { font-size: 2rem !important; }
h2 { font-size: 1.25rem !important; }
h3 { font-size: 1rem !important; }
.card {
  background: white; border:1px solid var(--border); border-radius:14px;
  padding:18px 20px; margin:0 0 14px 0;
  box-shadow:0 2px 10px rgba(15,62,110,.04);
}
.card-title {font-size:1.02rem;font-weight:750;color:var(--navy);margin-bottom:5px;}
.card-subtitle {font-size:.82rem;color:var(--muted);}
.brand-box {
  padding:8px 4px 14px; margin-bottom:10px; border-bottom:1px solid rgba(255,255,255,.18);
}
.brand-mark {
  width:38px;height:38px;border-radius:10px;display:inline-block;vertical-align:middle;
  margin-right:9px;background:linear-gradient(135deg,#FFFFFF 0%,#9DDAFF 45%,#00A6D6 100%);
  clip-path:polygon(18% 0,100% 0,72% 30%,100% 30%,66% 100%,0 100%,31% 54%,0 54%);
}
.brand-name {font-size:1.22rem;font-weight:800;vertical-align:middle;}
.brand-sub {font-size:.72rem;opacity:.78;margin-left:48px;margin-top:-5px;}
.badge {display:inline-block;padding:4px 9px;border-radius:999px;font-size:.70rem;font-weight:750;}
.green {background:#DCFCE7;color:#166534}.amber{background:#FEF3C7;color:#92400E}
.orange{background:#FFEDD5;color:#9A3412}.red{background:#FEE2E2;color:#991B1B}
.gray{background:#F1F5F9;color:#64748B}.blue{background:#EAF4FF;color:#0B3A78}
.navy{background:#E2E8F0;color:#0F172A}
.note {background:#EFF8FF;border-left:4px solid var(--bright);padding:11px 14px;border-radius:7px;color:#35536F;font-size:.82rem;margin:8px 0 14px;}
.warning-note {background:#FFFBEB;border-left:4px solid var(--amber);padding:11px 14px;border-radius:7px;color:#6B4D0C;font-size:.82rem;margin:8px 0 14px;}
.kpi {background:#fff;border:1px solid var(--border);border-radius:14px;padding:15px 16px;min-height:105px;}
.kpi-label {font-size:.77rem;color:var(--muted);font-weight:650}.kpi-value{font-size:1.65rem;font-weight:800;color:var(--navy);margin-top:7px}
.kpi-accent {width:34px;height:4px;border-radius:8px;background:var(--bright);margin-bottom:12px;}
.topbar {background:white;border:1px solid var(--border);border-radius:14px;padding:8px 12px;margin-bottom:15px;}
.section-kicker {color:#1976D2;font-size:.72rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-top:14px;}
.hero {background:linear-gradient(120deg,#062B5C,#1257A6 58%,#00A6D6);color:white;border-radius:16px;padding:22px;margin-bottom:16px;box-shadow:0 12px 30px rgba(6,43,92,.16);}
.hero h1,.hero h2,.hero h3 {color:white !important;}
.health {color:#16A34A;font-weight:750}.live{color:#16A34A;font-weight:700}.planned{color:#64748B;font-weight:700}
div[data-testid="stMetric"] {background:white;border:1px solid var(--border);border-radius:14px;padding:12px;}
div[data-testid="stDataFrame"] {border:1px solid var(--border);border-radius:10px;}
.stButton button {border-radius:9px;font-weight:650;}
.small {font-size:.78rem;color:var(--muted);}
.profile-pill {background:#EAF4FF;border:1px solid #CFE5FA;border-radius:12px;padding:8px 12px;color:#0B3A78;font-weight:700;}
section[data-testid="stSidebar"] .profile-pill {color:#0B3A78 !important;background:#EAF4FF !important;}
section[data-testid="stSidebar"] .profile-pill .small {color:#35536F !important;}
.nav-divider{height:1px;background:rgba(255,255,255,.14);margin:6px 0 10px}
.quick-chip{display:inline-block;background:#EFF6FF;border:1px solid #D6E8FA;color:#0B3A78;border-radius:999px;padding:5px 9px;font-size:.70rem;font-weight:750;margin-right:5px}

.profile-hero{background:linear-gradient(120deg,#062B5C,#0B4F98 65%,#0B72B5);border-radius:18px;padding:24px 26px;color:#fff;display:flex;align-items:center;gap:16px;box-shadow:0 12px 30px rgba(6,43,92,.14);position:relative;overflow:hidden}
.profile-avatar{width:64px;height:64px;border-radius:16px;background:linear-gradient(135deg,#fff,#9DDAFF);color:#0B3A78;display:flex;align-items:center;justify-content:center;font-size:1.65rem;font-weight:850;flex:none}
.profile-name{font-size:1.35rem;font-weight:850}.profile-title{font-size:.83rem;opacity:.9;margin-top:2px}.profile-team{font-size:.73rem;opacity:.72;margin-top:5px}
.profile-live{margin-left:auto;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);border-radius:999px;padding:7px 11px;font-size:.70rem;font-weight:750}.profile-live span{color:#72F1A4;margin-right:5px}
.profile-card{background:transparent;border:0;border-radius:0;padding:0;box-shadow:none;min-height:0}
.profile-card-title{color:#062B5C;font-size:1rem;font-weight:800;margin-bottom:8px}
.profile-row{display:flex;justify-content:space-between;gap:20px;padding:11px 0;border-bottom:1px solid #EDF3F8;font-size:.77rem}.profile-row:last-child{border-bottom:0}.profile-row span{color:#607D9B}.profile-row b{color:#173A5E;text-align:right}
.profile-mini{background:#fff;border:1px solid #D9E7F5;border-radius:14px;padding:16px;min-height:105px;box-shadow:0 5px 18px rgba(15,62,110,.04)}.profile-mini div{font-size:1.25rem;margin-bottom:7px}.profile-mini b{display:block;color:#062B5C;font-size:.82rem}.profile-mini span{display:block;color:#607D9B;font-size:.70rem;margin-top:4px}
.profile-note{background:#EFF8FF;border-left:4px solid #1976D2;padding:11px 14px;border-radius:8px;color:#35536F;font-size:.76rem;margin-top:15px}
.sidebar-section-label{font-size:.62rem;letter-spacing:.13em;font-weight:800;color:rgba(255,255,255,.52);margin:2px 0 7px}

.profile-section-title{color:#062B5C;font-size:.92rem;font-weight:800;margin:4px 0 5px}
.profile-row-clean{display:flex;justify-content:space-between;gap:20px;padding:10px 0;border-bottom:1px solid #E5EEF6;font-size:.75rem}
.profile-row-clean:last-child{border-bottom:0}
.profile-row-clean span{color:#607D9B}
.profile-row-clean b{color:#173A5E;text-align:right}
</style>


'''
