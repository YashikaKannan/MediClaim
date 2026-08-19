
def risk_band(score):
    score = float(score)
    if score >= 81: return "Critical", "red"
    if score >= 61: return "High", "orange"
    if score >= 31: return "Medium", "amber"
    return "Low", "green"

def risk_text(score):
    label, _ = risk_band(score)
    icons = {"Critical":"🔴", "High":"🟠", "Medium":"🟡", "Low":"🟢"}
    return f"{icons[label]} {label} · {int(score)}"

def risk_badge(score):
    label, cls = risk_band(score)
    return f'<span class="badge {cls}">{label} · {int(score)}</span>'

def status_text(status):
    icons = {"Active":"🟢", "Planned":"⚪", "Connected":"🟢", "Healthy":"🟢",
             "Success":"🟢", "Assigned":"🟡", "Open":"🔵", "Investigating":"🟣"}
    return f'{icons.get(str(status),"🔵")} {status}'

def status_badge(status):
    cls = {"Active":"green","Connected":"green","Healthy":"green","Success":"green",
           "LIVE":"green","Planned":"gray","Assigned":"amber","Open":"blue",
           "Investigating":"blue"}.get(str(status),"blue")
    return f'<span class="badge {cls}">{status}</span>'
