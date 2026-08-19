
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

RISK_COLORS = {"Low Risk":"#16A34A","Medium Risk":"#F59E0B","High Risk":"#F97316","Critical Risk":"#DC2626"}

def risk_donut(rows, title="Risk Distribution"):
    df = pd.DataFrame(rows)
    fig = px.pie(df, names="label", values="value", hole=.62,
                 color="label", color_discrete_map=RISK_COLORS)
    fig.update_traces(textinfo="label+percent", hovertemplate="%{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>")
    fig.update_layout(title=title, height=370, margin=dict(l=0,r=0,t=45,b=0), legend_title=None)
    return fig

def model_donut(df):
    blue = ["#062B5C","#1257A6","#1976D2","#00A6D6"]
    fig = go.Figure(go.Pie(labels=df["Model"], values=df["Weight"], hole=.62,
                           marker=dict(colors=blue), textinfo="label+percent",
                           hovertemplate="%{label}<br>%{value}%<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=0,r=0,t=30,b=0), legend_title=None)
    return fig

def bar(df, x, y, title, color="#1976D2", percent=False):
    fig = px.bar(df, x=x, y=y, title=title)
    fig.update_traces(marker_color=color, hovertemplate="%{x}<br>%{y}<extra></extra>")
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=45,b=0), xaxis_title=None, yaxis_title=None)
    return fig

def horizontal_contributions(df):
    fig = px.bar(df, x="Contribution", y="Feature", orientation="h",
                 color="Direction",
                 color_discrete_map={"Increases Risk":"#DC2626","Decreases Risk":"#1976D2"})
    fig.update_layout(height=330, margin=dict(l=0,r=0,t=10,b=0), legend_title=None)
    return fig
