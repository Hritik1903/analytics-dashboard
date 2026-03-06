"""
╔══════════════════════════════════════════════════════════════╗
║         COMBINED MULTI-PHASE ANALYTICS DASHBOARD            ║
║  Phase 1 · Phase 2 · Phase 3 & 4 · Phase 5 (Sentiment)      ║
║  Run:  python combined_dashboard.py                          ║
║  Open: http://127.0.0.1:8050                                 ║
╚══════════════════════════════════════════════════════════════╝

Install dependencies (one-time):
  pip install dash==2.14.2 jupyter-dash==0.4.2 pandas plotly
              dash-bootstrap-components scikit-learn scipy numpy
              nltk transformers torch wordcloud
"""

# ══════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════
import warnings, json, io, os, re, base64
warnings.filterwarnings("ignore")
print(">>> [1] base imports OK", flush=True)

import numpy as np
import pandas as pd
print(">>> [2] numpy/pandas OK", flush=True)
from datetime import date
from scipy import stats as scipy_stats

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

print(">>> [3] plotly OK", flush=True)
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)

print(">>> [4] dash/sklearn OK", flush=True)
# NLP (Phase 5) — gracefully degrade if missing
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from wordcloud import WordCloud
    nltk.download("vader_lexicon", quiet=True)
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# ══════════════════════════════════════════════════════════════
#  GLOBAL COLOUR PALETTE  (shared across all phases)
# ══════════════════════════════════════════════════════════════
print(">>> [5] all imports OK", flush=True)
NAV_BG   = "#0a0e1a"
DARK_BG  = "#0d1117"
CARD_BG  = "#161b22"
PANEL_BG = "#1c2128"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
ORANGE   = "#f0883e"
PURPLE   = "#bc8cff"
WARN     = "#f78166"
TEXT     = "#c9d1d9"
MUTED    = "#8b949e"
BORDER   = "#30363d"

# ══════════════════════════════════════════════════════════════
#  DATA LOADING  (all phases share df.csv where applicable)
# ══════════════════════════════════════════════════════════════
def _safe_read(path, **kw):
    """Read a CSV if it exists, otherwise return an empty DataFrame."""
    if os.path.exists(path):
        return pd.read_csv(path, **kw)
    return pd.DataFrame()

DF_MAIN   = _safe_read("df.csv",           parse_dates=["Date"])
X_TRAIN   = _safe_read("x_train2_5.csv",   parse_dates=["Date"])
Y_TRAIN   = _safe_read("y_train2_5.csv",   parse_dates=["Date"])
X_TEST    = _safe_read("x_test2_5.csv",    parse_dates=["Date"])
Y_TEST    = _safe_read("y_test2_5.csv",    parse_dates=["Date"])

# Phase-1 – SOURCE_DF  (same as DF_MAIN)
SOURCE_DF = DF_MAIN.copy() if not DF_MAIN.empty else pd.DataFrame()

# Phase-2 – enriched df_raw
if not DF_MAIN.empty:
    df_raw = DF_MAIN.sort_values("Date").reset_index(drop=True).copy()
    df_raw["Year"]    = df_raw["Date"].dt.year
    df_raw["Quarter"] = df_raw["Date"].dt.quarter
    df_raw["Month"]   = df_raw["Date"].dt.month
else:
    df_raw = pd.DataFrame(columns=["Date","Year","Quarter","Month"])

INDICES = {
    "Nifty 50"  : {"close":"Close_^NSEI",  "open":"Open_^NSEI",  "ret":"returns_^NSEI"},
    "DAX"       : {"close":"Close_^GDAXI", "open":"Open_^GDAXI", "ret":"returns_^GDAX"},
    "DJI"       : {"close":"Close_^DJI",   "open":"Open_^DJI",   "ret":"returns_^DJI"},
    "Nikkei 225": {"close":"Close_^N225",  "open":"Open_^N225",  "ret":"returns_^N225"},
    "NASDAQ"    : {"close":"Close_^IXIC",  "open":"Open_^IXIC",  "ret":"returns_^IXIC"},
    "Hang Seng" : {"close":"Close_^HSI",   "open":"Open_^HSI",   "ret":"returns_^HSI"},
    "VIX"       : {"close":"Close_^VIX",   "open":"Open_^VIX",   "ret":"returns_^VIX"},
}
ALL_IDX = list(INDICES.keys())
IDX_COLORS = {
    "Nifty 50":"#6c8cff","DAX":"#ff6b9d","DJI":"#00e5c0",
    "Nikkei 225":"#ffd166","NASDAQ":"#a78bfa","Hang Seng":"#f97316","VIX":"#ef4444",
}

def RET(idx):
    col = INDICES[idx]["ret"]
    return df_raw[col] if col in df_raw.columns else pd.Series(dtype=float)

def CLO(idx):
    col = INDICES[idx]["close"]
    return df_raw[col] if col in df_raw.columns else pd.Series(dtype=float)

YEARS = sorted(df_raw["Year"].unique()) if not df_raw.empty else []

# Phase-3/4 – ML
FEATURE_COLS = [c for c in X_TRAIN.columns if c != "Date"] if not X_TRAIN.empty else []
ML_MODELS = {
    "Logistic Regression":          LogisticRegression(C=0.1, solver="liblinear", random_state=42, max_iter=1000),
    "Decision Tree":                DecisionTreeClassifier(random_state=42),
    "Gaussian Naive Bayes":         GaussianNB(),
    "K-Nearest Neighbor (KNN)":     KNeighborsClassifier(n_neighbors=5),
    "Random Forest":                RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":            GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Adaptive Boosting (AdaBoost)": AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME"),
    "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
}
MODEL_NAMES   = list(ML_MODELS.keys())
CUTOFF_VALUES = [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]

# Phase-5 – NLP
if NLP_AVAILABLE:
    sid_vader  = SentimentIntensityAnalyzer()
    _tok       = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    _finbert   = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
else:
    sid_vader = _tok = _finbert = None

positive_words = {"good","great","positive","gain","profits","rise","up","bull","bullish","strong","growth"}
negative_words = {"bad","loss","losses","fall","down","bear","bearish","weak","decline","crash","debt"}

df_sentiment_global   = pd.DataFrame()
wc_b64_global         = ""

# ══════════════════════════════════════════════════════════════
#  DASH APP  INIT
# ══════════════════════════════════════════════════════════════
print(">>> [6] building app layout", flush=True)
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Multi-Phase Analytics Dashboard",
)
server = app.server

# ── CSS injection ──────────────────────────────────────────────
GLOBAL_CSS = """
<style>
body { background:#0d1117 !important; }
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:#161b22}
::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#58a6ff}
.phase-nav { border-bottom: none !important; }
.phase-nav .tab {
    background: transparent !important; border: none !important;
    border-bottom: 3px solid transparent !important; color: #8b949e !important;
    font-weight: 600 !important; font-size: 13px !important; padding: 0 28px !important;
    height: 62px !important; display: flex !important; align-items: center !important;
    justify-content: center !important; transition: all 0.2s !important; letter-spacing: 0.2px;
}
.phase-nav .tab:hover { color: #c9d1d9 !important; background: rgba(88,166,255,0.04) !important; }
.phase-nav .tab--selected {
    color: #58a6ff !important; border-bottom: 3px solid #58a6ff !important;
    background: rgba(88,166,255,0.09) !important; font-weight: 700 !important;
}
.phase-nav .tab-container--top { border-bottom: none !important; }
.pill-multi  label:has(input[type=checkbox]:checked){background:#6c8cff!important;border-color:#6c8cff!important;color:#fff!important}
.pill-single label:has(input[type=radio]:checked){background:#ff6b9d!important;border-color:#ff6b9d!important;color:#fff!important}
.pill-multi  label:hover{border-color:#6c8cff!important;color:#6c8cff!important}
.pill-single label:hover{border-color:#ff6b9d!important;color:#ff6b9d!important}
.eda-tabs .tab{background:#141726!important;border:none!important;color:#7b84b0!important;font-weight:600;padding:12px 22px}
.eda-tabs .tab--selected{color:#6c8cff!important;border-bottom:3px solid #6c8cff!important}
.eda-tabs .tab-container--top{border-bottom:1px solid #2a2f4a!important}
.model-btn:hover{opacity:.85;box-shadow:0 0 8px rgba(88,166,255,0.4)}
.rc-slider-track{background-color:#58a6ff!important}
.rc-slider-handle{border-color:#58a6ff!important;background:#58a6ff!important}
.dark-date .SingleDatePickerInput{background:#21262d!important;border:1px solid #30363d!important;border-radius:6px!important}
.dark-date .DateInput_input{background:#21262d!important;color:#c9d1d9!important;font-size:13px!important}
</style>"""

app.index_string = app.index_string.replace("</head>", GLOBAL_CSS + "</head>", 1)

# ══════════════════════════════════════════════════════════════
#  SHARED STYLE HELPERS
# ══════════════════════════════════════════════════════════════
CARD_STYLE = {"background":CARD_BG,"border":f"1px solid {BORDER}","borderRadius":"10px","padding":"20px","marginBottom":"16px"}
LABEL_STYLE = {"color":ACCENT,"fontWeight":"600","fontSize":"13px","letterSpacing":"0.5px","marginBottom":"6px"}
BTN_BLUE = {"background":"linear-gradient(135deg,#1f6feb 0%,#58a6ff 100%)","border":"none","borderRadius":"8px","color":"#fff","fontWeight":"600","padding":"9px 20px","cursor":"pointer","fontSize":"13px","boxShadow":"0 3px 10px rgba(88,166,255,0.35)"}
BTN_GREEN = {**BTN_BLUE,"background":"linear-gradient(135deg,#196c2e 0%,#3fb950 100%)","boxShadow":"0 3px 10px rgba(63,185,80,0.35)"}

# ── Phase 3 metric box style ───────────────────────────────────
METRIC_BOX_P3 = {
    "background": CARD_BG,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "14px 16px",
    "textAlign": "center",
}

# ══════════════════════════════════════════════════════════════
#  PHASE 1 – LAYOUT & CALLBACKS
# ══════════════════════════════════════════════════════════════
P1_ACCENT1   = "#3b82f6"
P1_ACCENT2   = "#10b981"
P1_ACCENT3   = "#06b6d4"
P1_TEXT      = "#e2e8f0"
P1_MUTED     = "#94a3b8"
P1_BG_CARD   = "#1e293b"
P1_BG_HEADER = "#1e3a5f"
P1_BORDER    = "#334155"
MIN_DATE = date(2017, 1, 2)
MAX_DATE = date.today()

def p1_section_card(children, style=None):
    base = {"background":P1_BG_CARD,"border":f"1px solid {P1_BORDER}","borderRadius":"12px",
            "padding":"24px","marginBottom":"20px","boxShadow":"0 4px 24px rgba(0,0,0,0.4)"}
    if style: base.update(style)
    return html.Div(children, style=base)

def p1_action_btn(label, btn_id, color, icon=""):
    return dbc.Button(
        [html.Span(icon+" ", style={"fontSize":"1rem"}), label],
        id=btn_id,
        style={"background":color,"border":"none","borderRadius":"8px","padding":"10px 22px",
               "fontWeight":"600","fontSize":"0.9rem","color":"#fff",
               "boxShadow":f"0 2px 12px {color}66","cursor":"pointer","transition":"opacity 0.2s"},
    )

def phase1_layout():
    return html.Div(
        style={"background":"#0f172a","minHeight":"100vh","fontFamily":"'Inter', sans-serif"},
        children=[
            html.Div(style={"background":P1_BG_HEADER,"padding":"16px 36px",
                            "borderBottom":f"2px solid {P1_ACCENT1}","display":"flex",
                            "alignItems":"center","gap":"14px","boxShadow":"0 2px 16px rgba(0,0,0,0.5)"},
                children=[
                    html.Div("📊", style={"fontSize":"2rem"}),
                    html.Div([
                        html.H4("Phase 1 — Master Data Builder",
                                style={"margin":0,"color":P1_TEXT,"fontWeight":"700","letterSpacing":"0.5px"}),
                        html.Span("Financial indices dataset builder powered by Yahoo Finance",
                                  style={"color":P1_MUTED,"fontSize":"0.82rem"}),
                    ]),
                    html.Div(style={"flex":1}),
                    html.Span("● LIVE", style={"color":P1_ACCENT2,"fontWeight":"700","fontSize":"0.8rem",
                                               "border":f"1px solid {P1_ACCENT2}","borderRadius":"20px","padding":"4px 12px"}),
                ]),
            html.Div(style={"padding":"28px 36px"}, children=[
                p1_section_card([
                    html.H5("⚙️ Dataset Configuration", style={"color":P1_TEXT,"marginBottom":"20px","fontWeight":"700"}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Start Date", style={"color":P1_MUTED,"fontSize":"0.85rem","fontWeight":"600","marginBottom":"6px"}),
                            dcc.DatePickerSingle(id="p1-start-date", date=date(2018,1,1),
                                display_format="YYYY-MM-DD", min_date_allowed=MIN_DATE,
                                max_date_allowed=MAX_DATE, style={"width":"100%"}),
                        ], md=3),
                        dbc.Col([
                            html.Label("End Date", style={"color":P1_MUTED,"fontSize":"0.85rem","fontWeight":"600","marginBottom":"6px"}),
                            dcc.DatePickerSingle(id="p1-end-date", date=date(2024,12,31),
                                display_format="YYYY-MM-DD", min_date_allowed=MIN_DATE,
                                max_date_allowed=MAX_DATE, style={"width":"100%"}),
                        ], md=3),
                        dbc.Col([
                            html.Label(" ", style={"color":"transparent","fontSize":"0.85rem","marginBottom":"6px","display":"block"}),
                            html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap"}, children=[
                                p1_action_btn("Build Master Data","p1-btn-build",P1_ACCENT1,"🗂"),
                                p1_action_btn("Save master.csv",  "p1-btn-save",P1_ACCENT2,"💾"),
                                dcc.Download(id="p1-download-csv"),
                                p1_action_btn("Download master.csv","p1-btn-download",P1_ACCENT3,"⬇️"),
                            ]),
                        ], md=6),
                    ], align="end"),
                    html.Div(id="p1-status-msg", style={"marginTop":"14px","minHeight":"22px"}),
                ]),
                html.Div(id="p1-stats-row"),
                p1_section_card([
                    html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"16px"}, children=[
                        html.H5("📋 Preview — Master Data", style={"color":P1_TEXT,"margin":0,"fontWeight":"700"}),
                        html.Span(id="p1-row-count-label", style={"color":P1_MUTED,"fontSize":"0.83rem"}),
                    ]),
                    html.Div(id="p1-data-table-container", children=[
                        html.Div("👆  Select a date range and click  Build Master Data  to preview the dataset.",
                                 style={"color":P1_MUTED,"textAlign":"center","padding":"48px 0","fontSize":"0.95rem"})
                    ]),
                ]),
                dcc.Store(id="p1-built-data-store"),
                dcc.Store(id="p1-saved-flag", data=False),
            ]),
        ]
    )

# ══════════════════════════════════════════════════════════════
#  PHASE 2 – EDA
# ══════════════════════════════════════════════════════════════
P2_BG='#0d0f1a'; P2_SURF='#141726'; P2_SURF2='#1c2035'
P2_BORDER='#2a2f4a'; P2_TXT='#e8eaf6'; P2_MUTED='#7b84b0'
P2_ACCENT='#6c8cff'; P2_SUC='#06d6a0'; P2_DNG='#ef4444'; P2_GOLD='#ffd166'

LAYOUT_BASE = dict(
    paper_bgcolor=P2_BG, plot_bgcolor=P2_SURF,
    font=dict(color=P2_MUTED, size=11, family='Segoe UI, system-ui, sans-serif'),
    margin=dict(t=46, b=50, l=58, r=22),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=P2_MUTED, size=10)),
    hoverlabel=dict(bgcolor=P2_SURF2, bordercolor=P2_BORDER, font=dict(color=P2_TXT)),
    xaxis=dict(gridcolor=P2_SURF2, zerolinecolor=P2_BORDER, tickfont=dict(color=P2_MUTED)),
    yaxis=dict(gridcolor=P2_SURF2, zerolinecolor=P2_BORDER, tickfont=dict(color=P2_MUTED)),
)

def base_fig(title='', rows=1, cols=1, subplot_titles=None, **kw):
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=subplot_titles or [], **kw)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=P2_TXT), x=0.01),
        **LAYOUT_BASE)
    for r in range(1, rows+1):
        for c in range(1, cols+1):
            fig.update_xaxes(gridcolor=P2_SURF2, zerolinecolor=P2_BORDER,
                             tickfont=dict(color=P2_MUTED), row=r, col=c)
            fig.update_yaxes(gridcolor=P2_SURF2, zerolinecolor=P2_BORDER,
                             tickfont=dict(color=P2_MUTED), row=r, col=c)
    if subplot_titles:
        for ann in fig.layout.annotations:
            ann.font.color = P2_TXT
            ann.font.size  = 11
    return fig

def no_data_fig(msg='Select options and click Generate'):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5,
                       xref='paper', yref='paper',
                       showarrow=False, font=dict(color=P2_MUTED, size=14))
    no_data_layout = LAYOUT_BASE.copy()
    no_data_layout['margin'] = dict(t=20,b=20,l=20,r=20)
    fig.update_layout(**no_data_layout)
    return fig

def hex_rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

# ── Phase 2 plot functions ─────────────────────────────────────

def fig_boxplot(indices):
    fig = go.Figure()
    yr_strs = [str(y) for y in YEARS]
    for idx in indices:
        flat_x, flat_y = [], []
        for yr in YEARS:
            vals = RET(idx)[df_raw['Year'] == yr].dropna().tolist()
            flat_x.extend([str(yr)] * len(vals))
            flat_y.extend(vals)
        if not flat_y:
            continue
        fig.add_trace(go.Box(
            x=flat_x, y=flat_y, name=idx,
            marker_color=IDX_COLORS[idx], line_color=IDX_COLORS[idx],
            fillcolor=hex_rgba(IDX_COLORS[idx], 0.35),
            legendgroup=idx, boxmean=True, line_width=1.5, opacity=0.88,
        ))
    fig.update_layout(
        title=dict(text='Returns Distribution — Boxplot by Year',
                   font=dict(size=13, color=P2_TXT), x=0.01),
        paper_bgcolor=P2_BG, plot_bgcolor=P2_SURF,
        font=dict(color=P2_MUTED, size=11, family='Segoe UI, system-ui, sans-serif'),
        margin=dict(t=46, b=50, l=58, r=22),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=P2_MUTED, size=10)),
        hoverlabel=dict(bgcolor=P2_SURF2, bordercolor=P2_BORDER, font=dict(color=P2_TXT)),
        boxmode='group', height=500,
        xaxis=dict(title='Year', type='category', categoryorder='array',
                   categoryarray=yr_strs, gridcolor=P2_SURF2, zerolinecolor=P2_BORDER,
                   tickfont=dict(color=P2_MUTED)),
        yaxis=dict(title='Daily Return (%)', gridcolor=P2_SURF2, zerolinecolor=P2_BORDER,
                   tickfont=dict(color=P2_MUTED)),
    )
    return fig

def fig_yearly_stats(indices):
    fig = base_fig(rows=1, cols=2,
                   subplot_titles=['Mean Annual Returns (%)', 'Annual Std Dev (%)'])
    shown = set()
    for idx in indices:
        means = [RET(idx)[df_raw['Year']==y].dropna().mean() for y in YEARS]
        stds  = [RET(idx)[df_raw['Year']==y].dropna().std()  for y in YEARS]
        sl = idx not in shown; shown.add(idx)
        fig.add_trace(go.Bar(name=idx, x=[str(y) for y in YEARS], y=means,
                             marker_color=IDX_COLORS[idx], opacity=0.85,
                             showlegend=sl, legendgroup=idx), row=1, col=1)
        fig.add_trace(go.Bar(name=idx, x=[str(y) for y in YEARS], y=stds,
                             marker_color=IDX_COLORS[idx], opacity=0.85,
                             showlegend=False, legendgroup=idx), row=1, col=2)
    fig.update_layout(barmode='group', height=410)
    return fig

def tbl_yearly_stats(indices):
    rows = []
    for yr in YEARS:
        row = {'Year': str(yr)}
        for idx in indices:
            v = RET(idx)[df_raw['Year']==yr].dropna()
            row[f'{idx} N']    = len(v)
            row[f'{idx} Mean'] = f'{v.mean():.4f}' if len(v) else '-'
            row[f'{idx} Std']  = f'{v.std():.4f}'  if len(v) else '-'
        rows.append(row)
    col_ids = ['Year'] + [c for idx in indices
                          for c in (f'{idx} N', f'{idx} Mean', f'{idx} Std')]
    return rows, [{'name':c,'id':c} for c in col_ids]

def fig_median_bar(indices):
    fig = base_fig('Median Daily Returns by Year (%)')
    shown = set()
    for idx in indices:
        meds = [RET(idx)[df_raw['Year']==y].dropna().median() for y in YEARS]
        sl = idx not in shown; shown.add(idx)
        fig.add_trace(go.Bar(name=idx, x=[str(y) for y in YEARS], y=meds,
                             marker_color=IDX_COLORS[idx], opacity=0.85,
                             showlegend=sl, legendgroup=idx))
    fig.update_layout(barmode='group', xaxis_title='Year',
                      yaxis_title='Median Return (%)', height=410)
    return fig

def fig_heatmap_yq(idx):
    mat = [[RET(idx)[(df_raw['Year']==y)&(df_raw['Quarter']==q)].dropna().median()
            for q in [1,2,3,4]] for y in YEARS]
    fig = base_fig(f'{idx} - Median Returns: Year x Quarter')
    fig.add_trace(go.Heatmap(
        z=mat, x=['Q1','Q2','Q3','Q4'], y=[str(y) for y in YEARS],
        colorscale=[[0,P2_DNG],[0.5,P2_SURF2],[1,P2_SUC]], zmid=0,
        text=[[f'{v:.3f}' for v in row] for row in mat],
        texttemplate='%{text}', textfont={'size':11,'color':P2_TXT},
        colorbar=dict(tickfont=dict(color=P2_MUTED))))
    fig.update_layout(height=430, yaxis_autorange='reversed')
    return fig

def fig_multivariate_median():
    fig = base_fig('Multivariate Median Returns by Year - All 7 Indices')
    for idx in ALL_IDX:
        meds = [RET(idx)[df_raw['Year']==y].dropna().median() for y in YEARS]
        fig.add_trace(go.Scatter(x=[str(y) for y in YEARS], y=meds,
                                mode='lines+markers', name=idx,
                                line=dict(color=IDX_COLORS[idx], width=2),
                                marker=dict(size=7, color=IDX_COLORS[idx])))
    fig.add_hline(y=0, line_color=P2_BORDER, line_dash='dot', line_width=1)
    fig.update_layout(xaxis_title='Year', yaxis_title='Median Return (%)', height=430)
    return fig

def fig_rolling_vol(indices):
    fig = base_fig('30-Day Rolling Volatility (Std Dev of Returns)')
    for idx in indices:
        rv = RET(idx).rolling(30).std()
        fig.add_trace(go.Scatter(x=df_raw['Date'], y=rv, mode='lines', name=idx,
                                line=dict(color=IDX_COLORS[idx], width=1.5), opacity=0.9))
    fig.update_layout(xaxis_title='Date', yaxis_title='Volatility (%)', height=430)
    return fig

def fig_cumret(indices):
    fig = base_fig('Cumulative Returns (%) from Start')
    for idx in indices:
        cum = RET(idx).fillna(0).cumsum()
        fig.add_trace(go.Scatter(x=df_raw['Date'], y=cum, mode='lines', name=idx,
                                line=dict(color=IDX_COLORS[idx], width=2)))
    fig.add_hline(y=0, line_color=P2_BORDER, line_dash='dot', line_width=1)
    fig.update_layout(xaxis_title='Date', yaxis_title='Cumulative Return (%)', height=430)
    return fig

def fig_drawdown(indices):
    fig = base_fig('Drawdown Analysis - Peak-to-Trough Decline')
    for idx in indices:
        close = CLO(idx).ffill()
        peak  = close.cummax()
        dd    = (close - peak) / peak * 100
        fig.add_trace(go.Scatter(x=df_raw['Date'], y=dd, mode='lines', name=idx,
                                line=dict(color=IDX_COLORS[idx], width=1.5),
                                fill='tozeroy',
                                fillcolor=hex_rgba(IDX_COLORS[idx], 0.15)))
    fig.add_hline(y=0, line_color=P2_BORDER, line_width=1)
    fig.update_layout(xaxis_title='Date', yaxis_title='Drawdown (%)', height=430)
    return fig

def fig_correlation(n_years=6):
    max_yr = int(df_raw['Year'].max())
    min_yr = max_yr - n_years + 1
    mask_s = df_raw['Year'] >= min_yr if n_years > 1 else df_raw['Year'] == max_yr
    title  = (f'Correlation - Latest Year ({max_yr})' if n_years == 1
              else f'Correlation - Last {n_years} Yrs ({min_yr}-{max_yr})')
    ret_df = pd.DataFrame({idx: RET(idx)[mask_s] for idx in ALL_IDX}).dropna()
    corr   = ret_df.corr().values
    mask   = np.triu(np.ones_like(corr, dtype=bool), k=1)
    z      = corr.copy().astype(float)
    z[mask] = np.nan
    txt    = [[f'{v:.2f}' if not np.isnan(v) else '' for v in row] for row in z]
    fig = base_fig(title)
    fig.add_trace(go.Heatmap(
        z=z, x=ALL_IDX, y=ALL_IDX,
        colorscale=[[0,P2_DNG],[0.5,P2_SURF2],[1,P2_SUC]], zmid=0, zmin=-1, zmax=1,
        text=txt, texttemplate='%{text}', textfont={'size':11,'color':P2_TXT},
        colorbar=dict(tickfont=dict(color=P2_MUTED))))
    fig.update_layout(height=480, yaxis_autorange='reversed')
    return fig

def fig_nifty_dir():
    year_stats = []
    for yr in YEARS:
        sub   = df_raw[df_raw['Year']==yr]
        total = len(sub)
        up    = int((sub['nifty_dir']==1).sum()) if 'nifty_dir' in sub.columns else 0
        dn    = total - up
        pct   = round(up/total*100, 2) if total else 0
        year_stats.append(dict(year=int(yr), total=total, up=up, dn=dn, pct_up=pct))
    yrs  = [str(s['year'])  for s in year_stats]
    ups  = [s['up']         for s in year_stats]
    dns  = [s['dn']         for s in year_stats]
    pcts = [s['pct_up']     for s in year_stats]
    fig = base_fig(rows=1, cols=2,
                   subplot_titles=['Up vs Down Days by Year',
                                   'Opening Direction - % Up Days'])
    fig.add_trace(go.Bar(name='Up (1)',        x=yrs, y=ups,
                         marker_color=P2_SUC, opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(name='Down/Flat (0)', x=yrs, y=dns,
                         marker_color=P2_DNG, opacity=0.85), row=1, col=1)
    bar_colors = [P2_SUC if p>50 else P2_DNG for p in pcts]
    fig.add_trace(go.Bar(name='% Up Days', x=yrs, y=pcts,
                         marker_color=bar_colors, opacity=0.85,
                         text=[f'{p:.1f}%' for p in pcts],
                         textposition='outside', showlegend=False), row=1, col=2)
    fig.add_hline(y=50, line_color=P2_GOLD, line_dash='dot', line_width=1.5, row=1, col=2)
    fig.update_layout(barmode='stack', height=440)
    fig.update_yaxes(title_text='No. of Days',  row=1, col=1)
    fig.update_yaxes(title_text='% Up Days', range=[0,115], row=1, col=2)
    return fig, year_stats

def tbl_nifty_dir(year_stats):
    cols = [
        {'name':'Year',               'id':'year'},
        {'name':'Total Days',         'id':'total'},
        {'name':'Up Days',            'id':'up'},
        {'name':'Down Days',          'id':'dn'},
        {'name':'Opening Dir (% Up)', 'id':'pct_up'},
        {'name':'Trend',              'id':'trend'},
    ]
    rows = []
    for s in year_stats:
        trend = ('Bullish' if s['pct_up']>55 else
                 'Bearish' if s['pct_up']<45 else 'Neutral')
        rows.append({**s, 'trend': trend})
    return rows, cols

def fig_scatter_xy(xi, yi):
    valid = pd.concat([RET(xi).rename('x'), RET(yi).rename('y')], axis=1).dropna()
    fig = base_fig(f'Scatter: {xi} vs {yi} - Daily Returns')
    fig.add_trace(go.Scatter(x=valid['x'], y=valid['y'], mode='markers',
                             marker=dict(color=IDX_COLORS[xi], size=5, opacity=0.45),
                             name=f'{xi} vs {yi}'))
    fig.update_layout(xaxis_title=f'{xi} Return (%)',
                      yaxis_title=f'{yi} Return (%)', height=470)
    return fig

def fig_scatter_ols(xi, yi):
    valid = pd.concat([RET(xi).rename('x'), RET(yi).rename('y')], axis=1).dropna()
    sl, ic, r, _, _ = scipy_stats.linregress(valid['x'].values, valid['y'].values)
    xr = np.linspace(valid['x'].min(), valid['x'].max(), 100)
    fig = base_fig(f'{xi} vs {yi} - Scatter + OLS  (R2={r**2:.3f})')
    fig.add_trace(go.Scatter(x=valid['x'], y=valid['y'], mode='markers',
                             marker=dict(color=IDX_COLORS[xi], size=5, opacity=0.35),
                             name='Data'))
    fig.add_trace(go.Scatter(x=xr, y=sl*xr+ic, mode='lines',
                             line=dict(color=IDX_COLORS[yi], width=2.5),
                             name=f'OLS: y={sl:.3f}x+{ic:.3f}  R2={r**2:.3f}'))
    fig.update_layout(xaxis_title=f'{xi} Return (%)',
                      yaxis_title=f'{yi} Return (%)', height=470)
    return fig

def fig_scatter_vix(xi):
    valid = pd.concat([RET(xi).rename('ret'), RET('VIX').rename('vix')], axis=1).dropna()
    fig = base_fig(f'{xi} Return vs VIX Return')
    fig.add_trace(go.Scatter(x=valid['vix'], y=valid['ret'], mode='markers',
                             marker=dict(color=valid['ret'].tolist(),
                                         colorscale='RdYlGn', size=5, opacity=0.45,
                                         showscale=True,
                                         colorbar=dict(title='Return%',
                                                       tickfont=dict(color=P2_MUTED))),
                             name=f'{xi} vs VIX'))
    fig.update_layout(xaxis_title='VIX Return (%)',
                      yaxis_title=f'{xi} Return (%)', height=470)
    return fig

def fig_scatter_matrix():
    ret_df = pd.DataFrame({idx: RET(idx) for idx in ALL_IDX}).dropna()
    sample = ret_df.sample(min(600, len(ret_df)), random_state=42)
    dims   = [dict(label=idx, values=sample[idx].tolist()) for idx in ALL_IDX]
    fig = go.Figure(go.Splom(
        dimensions=dims, showupperhalf=False, diagonal_visible=True,
        marker=dict(color=P2_ACCENT, size=3, opacity=0.35,
                    line=dict(width=0.3, color=P2_BORDER))))
    fig.update_layout(
        title=dict(text='Scatter Matrix - All Indices Returns',
                   font=dict(size=13, color=P2_TXT), x=0.01),
        paper_bgcolor=P2_BG, plot_bgcolor=P2_SURF,
        font=dict(color=P2_MUTED, size=10),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=800, margin=dict(t=50,b=30,l=60,r=30))
    return fig

def fig_pair_corr():
    ret_df = pd.DataFrame({idx: RET(idx) for idx in ALL_IDX}).dropna()
    corr   = ret_df.corr().values
    txt    = [[f'{v:.2f}' for v in row] for row in corr]
    fig = base_fig('Pairwise Correlation Heatmap - All Indices')
    fig.add_trace(go.Heatmap(
        z=corr, x=ALL_IDX, y=ALL_IDX,
        colorscale=[[0,P2_DNG],[0.5,P2_SURF2],[1,P2_SUC]], zmid=0, zmin=-1, zmax=1,
        text=txt, texttemplate='%{text}', textfont={'size':11,'color':P2_TXT},
        colorbar=dict(tickfont=dict(color=P2_MUTED))))
    fig.update_layout(height=480, yaxis_autorange='reversed')
    return fig

def fig_fe_returns_dist():
    vals = RET('Nifty 50').dropna().tolist()
    fig = base_fig(rows=1, cols=2,
                   subplot_titles=['Daily Returns Distribution', 'Q-Q Normal Plot'])
    fig.add_trace(go.Histogram(x=vals, nbinsx=80, name='Nifty 50',
                               marker=dict(color=P2_ACCENT, opacity=0.8,
                                           line=dict(width=0.3, color=P2_SURF2))),
                  row=1, col=1)
    (osm, osr), (sl, ic, _) = scipy_stats.probplot(vals, dist='norm')
    fig.add_trace(go.Scatter(x=list(osm), y=list(osr), mode='markers',
                             marker=dict(color=P2_ACCENT, size=4, opacity=0.5),
                             name='Sample'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[float(min(osm)), float(max(osm))],
                             y=[sl*float(min(osm))+ic, sl*float(max(osm))+ic],
                             mode='lines', line=dict(color='#ff6b9d', width=2),
                             name='Normal'), row=1, col=2)
    fig.update_xaxes(title_text='Return (%)',           row=1, col=1)
    fig.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
    fig.update_yaxes(title_text='Frequency',            row=1, col=1)
    fig.update_yaxes(title_text='Sample Quantiles',     row=1, col=2)
    fig.update_layout(showlegend=False, height=390)
    return fig

def fig_fe_lag():
    ret  = RET('Nifty 50').dropna()
    lags = list(range(1, 21))
    acf  = [ret.autocorr(lag=l) for l in lags]
    conf = 1.96 / np.sqrt(len(ret))
    fig = base_fig(rows=1, cols=2,
                   subplot_titles=['Autocorrelation - Lag Features',
                                   'Daily Returns + MA-5 / MA-10'])
    fig.add_trace(go.Bar(x=[str(l) for l in lags], y=acf,
                         marker_color=[P2_SUC if v>=0 else P2_DNG for v in acf],
                         name='ACF'), row=1, col=1)
    fig.add_hline(y= conf, line_color=P2_GOLD, line_dash='dot', line_width=1.2, row=1, col=1)
    fig.add_hline(y=-conf, line_color=P2_GOLD, line_dash='dot', line_width=1.2, row=1, col=1)
    fig.add_hline(y=0, line_color=P2_BORDER, line_width=0.8, row=1, col=1)
    fig.add_trace(go.Scatter(x=df_raw['Date'], y=RET('Nifty 50'),
                             mode='lines', line=dict(color=P2_MUTED,width=0.6),
                             opacity=0.4, name='Daily'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df_raw['Date'], y=RET('Nifty 50').rolling(5).mean(),
                             mode='lines', line=dict(color=P2_ACCENT,width=1.8),
                             name='MA-5'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df_raw['Date'], y=RET('Nifty 50').rolling(10).mean(),
                             mode='lines', line=dict(color='#ff6b9d',width=1.8),
                             name='MA-10'), row=1, col=2)
    fig.update_xaxes(title_text='Lag',  row=1, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=2)
    fig.update_yaxes(title_text='ACF',        row=1, col=1)
    fig.update_yaxes(title_text='Return (%)', row=1, col=2)
    fig.update_layout(height=410)
    return fig

def fig_fe_vix():
    valid = pd.concat([RET('VIX').rename('vix_ret'),
                       RET('Nifty 50').rename('nifty_ret')], axis=1).dropna()
    sl, ic, r, _, _ = scipy_stats.linregress(valid['vix_ret'].values, valid['nifty_ret'].values)
    xr = np.linspace(valid['vix_ret'].min(), valid['vix_ret'].max(), 100)
    fig = base_fig(rows=1, cols=2,
                   subplot_titles=['VIX Close - Full Period',
                                   'VIX Return vs Nifty Return'])
    fig.add_trace(go.Scatter(x=df_raw['Date'], y=CLO('VIX'), mode='lines',
                             line=dict(color=P2_DNG,width=1.3),
                             fill='tozeroy', fillcolor=hex_rgba(P2_DNG, 0.12),
                             name='VIX Close'), row=1, col=1)
    fig.add_hline(y=20, line_color=P2_GOLD, line_dash='dot', line_width=1.5, row=1, col=1)
    fig.add_trace(go.Scatter(x=valid['vix_ret'], y=valid['nifty_ret'], mode='markers',
                             marker=dict(color=P2_DNG, size=4, opacity=0.35),
                             name='Data'), row=1, col=2)
    fig.add_trace(go.Scatter(x=xr, y=sl*xr+ic, mode='lines',
                             line=dict(color=P2_GOLD,width=2),
                             name=f'OLS R2={r**2:.3f}'), row=1, col=2)
    fig.update_xaxes(title_text='Date',           row=1, col=1)
    fig.update_xaxes(title_text='VIX Return (%)', row=1, col=2)
    fig.update_yaxes(title_text='VIX Level',      row=1, col=1)
    fig.update_yaxes(title_text='Nifty Return (%)', row=1, col=2)
    fig.update_layout(height=410)
    return fig

def fig_fe_time():
    months    = ['Jan','Feb','Mar','Apr','May','Jun',
                 'Jul','Aug','Sep','Oct','Nov','Dec']
    month_ret = [RET('Nifty 50')[df_raw['Month']==m].dropna().mean()
                 for m in range(1,13)]
    q_data    = [RET('Nifty 50')[df_raw['Quarter']==q].dropna().tolist()
                 for q in [1,2,3,4]]
    yr_cnt    = [len(df_raw[df_raw['Year']==y]) for y in YEARS]
    fig = base_fig(rows=1, cols=3,
                   subplot_titles=['Avg Return by Month',
                                   'Return Dist by Quarter',
                                   'Observations per Year'])
    fig.add_trace(go.Bar(x=months, y=month_ret,
                         marker_color=[P2_SUC if v>=0 else P2_DNG for v in month_ret],
                         name='Monthly Avg'), row=1, col=1)
    q_colors = [P2_ACCENT,'#ff6b9d','#00e5c0',P2_GOLD]
    for i, (q_vals, qc) in enumerate(zip(q_data, q_colors), start=1):
        fig.add_trace(go.Box(y=q_vals, name=f'Q{i}',
                             marker_color=qc, line_color=qc,
                             boxmean=True, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=[str(y) for y in YEARS], y=yr_cnt,
                         marker_color=P2_ACCENT, opacity=0.85,
                         text=yr_cnt, textposition='outside',
                         name='Obs/Year'), row=1, col=3)
    fig.add_hline(y=0, line_color=P2_BORDER, line_width=0.8, row=1, col=1)
    fig.update_xaxes(title_text='Month',   row=1, col=1)
    fig.update_xaxes(title_text='Quarter', row=1, col=2)
    fig.update_xaxes(title_text='Year',    row=1, col=3)
    fig.update_yaxes(title_text='Avg Return (%)', row=1, col=1)
    fig.update_yaxes(title_text='Return (%)',     row=1, col=2)
    fig.update_yaxes(title_text='Count',           row=1, col=3)
    fig.update_layout(showlegend=False, height=420)
    return fig

def fig_fe_target():
    up = int((df_raw['nifty_dir'] == 1).sum()) if 'nifty_dir' in df_raw.columns else 0
    dn = int((df_raw['nifty_dir'] == 0).sum()) if 'nifty_dir' in df_raw.columns else 0
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=['Up (1)', 'Down/Flat (0)'],
        values=[up, dn],
        hole=0.45,
        marker=dict(colors=[P2_SUC, P2_DNG], line=dict(color=P2_BG, width=2)),
        textinfo='percent+label',
        textfont=dict(color=P2_TXT, size=12),
        sort=False, showlegend=True,
    ))
    fig.update_layout(
        title=dict(text='Target Class Distribution', font=dict(size=13, color=P2_TXT), x=0.01),
        paper_bgcolor=P2_BG, plot_bgcolor=P2_BG,
        font=dict(color=P2_MUTED, size=11, family='Segoe UI, system-ui, sans-serif'),
        margin=dict(t=46, b=20, l=20, r=20),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=P2_MUTED, size=11)),
        hoverlabel=dict(bgcolor=P2_SURF2, bordercolor=P2_BORDER, font=dict(color=P2_TXT)),
        height=380,
    )
    return fig

def tbl_vif():
    from numpy.linalg import inv as np_inv
    feat_cols  = [INDICES[idx]['ret'] for idx in ALL_IDX] + ['Close_^VIX','Year','Quarter','Month']
    feat_names = [f'returns_{idx}' for idx in ALL_IDX] + ['Close_^VIX','Year','Quarter','Month']
    available  = [c for c in feat_cols if c in df_raw.columns]
    avail_names= [feat_names[feat_cols.index(c)] for c in available]
    sub  = df_raw[available].dropna()
    corr = sub.corr().values
    try:
        vifs = np.diag(np_inv(corr))
    except np.linalg.LinAlgError:
        vifs = [None]*len(avail_names)
    rows = []
    for name, v in zip(avail_names, vifs):
        status = ('OK' if v is not None and v<5
                  else 'Moderate' if v is not None and v<10
                  else 'High')
        rows.append({'Feature':name,
                     'VIF Score': f'{v:.3f}' if v is not None else '-',
                     'Status':status})
    cols = [{'name':c,'id':c} for c in ['Feature','VIF Score','Status']]
    return rows, cols

# ── Phase 2 UI helpers ─────────────────────────────────────────
P2_CARD_S  = {'background':P2_SURF, 'border':f'1px solid {P2_BORDER}',
               'borderRadius':'14px', 'padding':'18px', 'marginBottom':'14px'}
P2_PANEL_S = {'background':P2_SURF2, 'border':f'1px solid {P2_BORDER}',
               'borderRadius':'16px', 'padding':'18px 22px', 'marginBottom':'18px'}
P2_LABEL_S = {'fontSize':'0.75rem', 'fontWeight':'700',
               'textTransform':'uppercase', 'letterSpacing':'0.07em',
               'color':P2_MUTED, 'marginBottom':'8px'}
P2_BTN_S   = {'background':'linear-gradient(135deg,#6c8cff,#4a6bff)',
               'border':'none', 'color':'white', 'fontWeight':'700',
               'fontSize':'0.88rem', 'padding':'10px 24px',
               'borderRadius':'10px', 'cursor':'pointer',
               'boxShadow':'0 4px 20px rgba(108,140,255,0.35)', 'marginTop':'8px'}
P2_TBL_S   = {'overflowX':'auto', 'borderRadius':'10px'}
P2_HDR_C    = {'backgroundColor':P2_SURF2, 'color':P2_ACCENT,
               'fontSize':'0.73rem', 'fontWeight':'700',
               'padding':'9px 13px', 'textTransform':'uppercase',
               'letterSpacing':'0.05em', 'border':f'1px solid {P2_BORDER}'}
P2_DAT_C    = {'backgroundColor':P2_SURF, 'color':P2_TXT,
               'fontSize':'0.82rem', 'padding':'8px 13px',
               'border':f'1px solid {P2_BORDER}'}

# ── Tab option lists (FIXED — these were missing!) ────────────
T1_OPTS = [
    'Boxplot by Year',
    'Yearly Stats (Chart + Table)',
    'Median Bar by Year',
    'Heatmap Year x Quarter',
    'Multivariate Median (All 7)',
    'Rolling Volatility-30d',
    'Cumulative Returns',
    'Drawdown',
    'Correlation - 6 Years',
    'Correlation - Latest Year',
    'Nifty Opening Direction',
    'Opening Direction Table',
]

T2_OPTS = [
    'Scatter X vs Y Returns',
    'Scatter + OLS Trendline',
    'Scatter Return vs VIX',
    'Scatter Matrix (All Indices)',
    'Pairwise Correlation Heatmap',
]

# Pill label style
_PL = {'display':'inline-flex', 'alignItems':'center', 'margin':'3px 4px',
        'padding':'6px 14px', 'borderRadius':'20px',
        'border':f'1.5px solid {P2_BORDER}', 'background':P2_SURF,
        'color':P2_MUTED, 'cursor':'pointer', 'fontSize':'0.8rem',
        'fontWeight':'600', 'transition':'all 0.18s'}
_PI = {'display':'none'}
_PW = {'display':'flex', 'flexWrap':'wrap', 'gap':'0px'}

def pill_check(pid, options, value):
    return dcc.Checklist(id=pid,
                         options=[{'label':o,'value':o} for o in options],
                         value=value,
                         labelStyle=_PL, inputStyle=_PI, style=_PW)

def pill_radio(pid, options, value):
    return dcc.RadioItems(id=pid,
                          options=[{'label':o,'value':o} for o in options],
                          value=value,
                          labelStyle=_PL, inputStyle=_PI, style=_PW)

def p2_make_tbl(data, cols, style_cond=None):
    return dash_table.DataTable(
        data=data, columns=cols,
        style_table=P2_TBL_S, style_header=P2_HDR_C, style_data=P2_DAT_C,
        style_cell={'fontFamily':'Segoe UI,system-ui,sans-serif','padding':'8px 13px'},
        style_data_conditional=style_cond or [],
        page_size=15, sort_action='native',
        filter_action='native', export_format='csv')

def ftag(label, color=None):
    color = color or P2_ACCENT
    h = color.lstrip('#')
    r2, g2, b2 = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return html.Span(label, style={
        'display':'inline-block',
        'background':f'rgba({r2},{g2},{b2},0.12)',
        'border':f'1px solid rgba({r2},{g2},{b2},0.28)',
        'color':color, 'borderRadius':'20px',
        'padding':'3px 11px', 'fontSize':'0.76rem', 'fontWeight':'600',
        'marginRight':'6px', 'marginBottom':'6px'})

def metric_card(lbl, val, sub, vc=None):
    vc = vc or P2_TXT
    return html.Div(style={
        'flex':'1', 'minWidth':'130px', 'background':P2_SURF2,
        'border':f'1px solid {P2_BORDER}', 'borderRadius':'11px', 'padding':'12px 16px'},
        children=[
            html.Div(lbl, style={'fontSize':'0.7rem','fontWeight':'700',
                                 'textTransform':'uppercase',
                                 'letterSpacing':'0.06em','color':P2_MUTED}),
            html.Div(val, style={'fontSize':'1.35rem','fontWeight':'800',
                                 'color':vc,'marginTop':'3px'}),
            html.Div(sub, style={'fontSize':'0.73rem','color':P2_MUTED}),
        ])

def sec_hdr(dot_color, title, btn_id):
    return html.Div(
        style={'display':'flex','alignItems':'center','gap':'8px','marginBottom':'12px'},
        children=[
            html.Div(style={'width':'8px','height':'8px','borderRadius':'50%',
                            'background':dot_color,'flexShrink':'0'}),
            html.Strong(title, style={'color':P2_TXT,'fontSize':'0.9rem'}),
            html.Button('Load', id=btn_id, n_clicks=0,
                        style={**P2_BTN_S,'padding':'5px 14px',
                               'fontSize':'0.76rem','marginTop':'0',
                               'marginLeft':'auto'}),
        ])

def code_box(lines_list, color='#00e5c0'):
    children = []
    for i, line in enumerate(lines_list):
        children.append(line)
        if i < len(lines_list)-1:
            children.append(html.Br())
    return html.Div(children, style={
        'background':'#0a0d14', 'border':f'1px solid {P2_BORDER}',
        'borderRadius':'9px', 'padding':'13px 18px',
        'fontFamily':'Courier New, monospace', 'fontSize':'0.88rem',
        'color':color, 'lineHeight':'1.9', 'marginBottom':'12px'})

# ── Phase 2 tab layout builders ────────────────────────────────

def p2_layout_tab1():
    return html.Div([
        html.Div(style=P2_PANEL_S, children=[
            html.Div([
                html.Div('Select Indices (multi-select)', style=P2_LABEL_S),
                html.Div(className='pill-multi',
                         children=[pill_check('p2-t1-indices', ALL_IDX, ['Nifty 50'])]),
            ], style={'marginBottom':'14px'}),
            html.Div([
                html.Div('Select Plot Types (multi-select)', style=P2_LABEL_S),
                html.Div(className='pill-multi',
                         children=[pill_check('p2-t1-plot-types', T1_OPTS,
                                              ['Boxplot by Year',
                                               'Yearly Stats (Chart + Table)'])]),
            ], style={'marginBottom':'14px'}),
            html.Button('Generate Plots', id='p2-t1-btn', n_clicks=0, style=P2_BTN_S),
        ]),
        html.Div(id='p2-t1-status',
                 style={'display':'none','alignItems':'center','gap':'6px',
                        'padding':'7px 14px','marginBottom':'10px',
                        'background':'rgba(108,140,255,0.08)',
                        'border':'1px solid rgba(108,140,255,0.2)',
                        'borderRadius':'8px','fontSize':'0.78rem',
                        'color':P2_ACCENT,'fontWeight':'600'}),
        dcc.Loading(type='circle', color=P2_ACCENT, children=html.Div(id='p2-t1-plots-container')),
    ])

def p2_layout_tab2():
    return html.Div([
        html.Div(style=P2_PANEL_S, children=[
            html.Div(style={'display':'flex','gap':'32px','flexWrap':'wrap',
                            'marginBottom':'14px'}, children=[
                html.Div([
                    html.Div('Index X (single-select)', style=P2_LABEL_S),
                    html.Div(className='pill-single',
                             children=[pill_radio('p2-t2-x', ALL_IDX, 'Nifty 50')]),
                ]),
                html.Div([
                    html.Div('Index Y (single-select)', style=P2_LABEL_S),
                    html.Div(className='pill-single',
                             children=[pill_radio('p2-t2-y', ALL_IDX, 'DJI')]),
                ]),
            ]),
            html.Div([
                html.Div('Plot Types (multi-select)', style=P2_LABEL_S),
                html.Div(className='pill-multi',
                         children=[pill_check('p2-t2-plot-types', T2_OPTS,
                                              ['Scatter X vs Y Returns',
                                               'Scatter + OLS Trendline'])]),
            ], style={'marginBottom':'14px'}),
            html.Button('Generate Plots', id='p2-t2-btn', n_clicks=0, style=P2_BTN_S),
        ]),
        html.Div(id='p2-t2-status',
                 style={'display':'none','alignItems':'center','gap':'6px',
                        'padding':'7px 14px','marginBottom':'10px',
                        'background':'rgba(108,140,255,0.08)',
                        'border':'1px solid rgba(108,140,255,0.2)',
                        'borderRadius':'8px','fontSize':'0.78rem',
                        'color':P2_ACCENT,'fontWeight':'600'}),
        dcc.Loading(type='circle', color=P2_ACCENT, children=html.Div(id='p2-t2-plots-container')),
    ])

def p2_layout_tab3():
    return html.Div([
        html.Div(style={'display':'flex','gap':'10px','flexWrap':'wrap','marginBottom':'16px'},
                 children=[
            metric_card('Total Observations', '2,347', 'Trading Days 2017-2025'),
            metric_card('Indices Covered',    '7',
                        'Nifty 50 | DAX | DJI | N225 | NASDAQ | HSI | VIX'),
            metric_card('Target: Up Days',    '1,581', '67.3% of all days',    P2_SUC),
            metric_card('Target: Down Days',  '766',   '32.7% of all days',    P2_DNG),
            metric_card('Features Engineered','25+',   'Lags | VIX | Time | Ratios', P2_ACCENT),
        ]),
        dbc.Row([
            dbc.Col(html.Div(style=P2_CARD_S, children=[
                sec_hdr('#00e5c0', 'Daily Returns - Formula, Distribution & Q-Q', 'p2-fe-ret-btn'),
                code_box([
                    'Daily Return (%) = [ (Pt - Pt-1) / Pt-1 ] x 100',
                    '',
                    '  Pt   = Closing price at time t',
                    '  Pt-1 = Closing price at previous step',
                ]),
                html.Div([ftag(t, '#00e5c0') for t in
                          ['7 Indices', 'Pre-computed in dataset', 'returns_^INDEX column']]),
                dcc.Loading(type='circle', color=P2_ACCENT,
                    children=dcc.Graph(id='p2-fe-ret-graph',
                                       figure=no_data_fig('Click Load to render'),
                                       config={'displayModeBar':False})),
            ]), width=12),
            dbc.Col(html.Div(style=P2_CARD_S, children=[
                sec_hdr(P2_ACCENT, 'Lag-Based Market Indicators', 'p2-fe-lag-btn'),
                html.Div([ftag(t, P2_ACCENT) for t in
                          ['ret_lag1','ret_lag2','ret_lag3',
                           'rolling_mean_5','rolling_mean_10','rolling_std_5']]),
                html.P('Lag features capture market memory and autocorrelation. '
                       'Past global returns serve as predictive signals for Nifty '
                       'opening direction via time-zone lead-lag relationships.',
                       style={'fontSize':'0.82rem','color':P2_MUTED,'lineHeight':'1.65','marginTop':'8px'}),
                dcc.Loading(type='circle', color=P2_ACCENT,
                    children=dcc.Graph(id='p2-fe-lag-graph',
                                       figure=no_data_fig('Click Load to render'),
                                       config={'displayModeBar':False})),
            ]), width=6),
            dbc.Col(html.Div(style=P2_CARD_S, children=[
                sec_hdr(P2_DNG, 'USA VIX Volatility Features', 'p2-fe-vix-btn'),
                html.Div([ftag(t, '#ff6b9d') for t in
                          ['Close_^VIX','returns_^VIX','VIX_lag1','VIX_ma5']] +
                         [ftag('VIX > 20 fear flag', P2_DNG)]),
                html.P('VIX measures market fear. High VIX indicates bearish opening. '
                       'Low VIX indicates stable or bullish conditions. '
                       'Critical predictor for Nifty opening direction.',
                       style={'fontSize':'0.82rem','color':P2_MUTED,'lineHeight':'1.65','marginTop':'8px'}),
                dcc.Loading(type='circle', color=P2_ACCENT,
                    children=dcc.Graph(id='p2-fe-vix-graph',
                                       figure=no_data_fig('Click Load to render'),
                                       config={'displayModeBar':False})),
            ]), width=6),
            dbc.Col(html.Div(style=P2_CARD_S, children=[
                sec_hdr(P2_GOLD, 'Time Indicators', 'p2-fe-time-btn'),
                html.Div([ftag(t, P2_GOLD) for t in
                          ['Year 2017-2025','Quarter 1-4','Month 1-12',
                           'Day of Week','Week of Year']]),
                html.P('Seasonal and cyclical patterns drive market behaviour. '
                       'Budget quarters, earnings months and day-of-week '
                       'effects are key classification signals.',
                       style={'fontSize':'0.82rem','color':P2_MUTED,'lineHeight':'1.65','marginTop':'8px'}),
                dcc.Loading(type='circle', color=P2_ACCENT,
                    children=dcc.Graph(id='p2-fe-time-graph',
                                       figure=no_data_fig('Click Load to render'),
                                       config={'displayModeBar':False})),
            ]), width=6),
            dbc.Col(html.Div(style=P2_CARD_S, children=[
                sec_hdr(P2_SUC, 'Target Variable: Nifty Opening Direction', 'p2-fe-tgt-btn'),
                code_box([
                    'nifty_dir = 1  if  Open(t) > Close(t-1)',
                    'nifty_dir = 0  otherwise',
                ], color=P2_GOLD),
                html.Div([ftag(t, '#00e5c0') for t in
                          ['Binary Classification','Class 1 (Up): 1,581',
                           'Class 0 (Down): 766']] + [ftag('Ratio 2.06:1', P2_DNG)]),
                dcc.Loading(type='circle', color=P2_ACCENT,
                    children=dcc.Graph(id='p2-fe-tgt-graph',
                                       figure=no_data_fig('Click Load to render'),
                                       config={'displayModeBar':False})),
            ]), width=6),
            dbc.Col(html.Div(style=P2_CARD_S, children=[
                sec_hdr('#ff6b9d', 'VIF Scores - Variance Inflation Factor', 'p2-fe-vif-btn'),
                html.P('VIF < 5 = OK  |  VIF 5-10 = Moderate  |  VIF > 10 = High',
                       style={'fontSize':'0.78rem','color':P2_MUTED,'marginBottom':'10px'}),
                dcc.Loading(type='circle', color=P2_ACCENT,
                    children=html.Div(id='p2-fe-vif-table')),
            ]), width=12),
        ]),
    ])

def phase2_layout():
    return html.Div(
        style={'background':P2_BG, 'minHeight':'100vh',
               'fontFamily':'Segoe UI, system-ui, sans-serif', 'color':P2_TXT},
        children=[
            html.Div(style={'background':'linear-gradient(135deg,#0d0f1a,#141726 50%,#1a1040)',
                            'borderBottom':f'1px solid {P2_BORDER}', 'padding':'16px 28px',
                            'display':'flex', 'alignItems':'center', 'gap':'14px',
                            'position':'sticky', 'top':'0', 'zIndex':'200'},
            children=[
                html.Div('EDA', style={'width':'40px','height':'40px','borderRadius':'10px',
                                       'background':'linear-gradient(135deg,#6c8cff,#ff6b9d)',
                                       'display':'flex','alignItems':'center',
                                       'justifyContent':'center','fontSize':'14px',
                                       'fontWeight':'900','color':'white','flexShrink':'0'}),
                html.Div([
                    html.Div([html.Span('Phase 2 - ', style={'color':P2_TXT}),
                              html.Span('EDA Dashboard', style={'color':P2_ACCENT})],
                             style={'fontSize':'1.15rem','fontWeight':'700'}),
                    html.Div('Global Indices 2017-2025 | 2,347 Observations | Dash + Plotly',
                             style={'fontSize':'0.73rem','color':P2_MUTED,'marginTop':'2px'}),
                ]),
                html.Div('Exploratory Analysis',
                         style={'marginLeft':'auto','background':'rgba(108,140,255,0.12)',
                                'border':f'1px solid {P2_ACCENT}','color':P2_ACCENT,
                                'borderRadius':'20px','padding':'4px 12px',
                                'fontSize':'0.73rem','fontWeight':'700'}),
            ]),
            dcc.Tabs(id='p2-main-tabs', value='tab1', className='eda-tabs',
                     style={'background':P2_SURF,'padding':'0 28px',
                            'position':'sticky','top':'72px','zIndex':'199'},
                     children=[
                         dcc.Tab(label='Single-Index Views',    value='tab1'),
                         dcc.Tab(label='Index-wise / Pairwise', value='tab2'),
                         dcc.Tab(label='Feature Engineering',   value='tab3'),
                     ]),
            html.Div(id='p2-tab1-panel', style={'padding':'22px 28px', 'display':'block'},
                     children=[p2_layout_tab1()]),
            html.Div(id='p2-tab2-panel', style={'padding':'22px 28px', 'display':'none'},
                     children=[p2_layout_tab2()]),
            html.Div(id='p2-tab3-panel', style={'padding':'22px 28px', 'display':'none'},
                     children=[p2_layout_tab3()]),
        ],
    )

# ══════════════════════════════════════════════════════════════
#  PHASE 3 & 4 – LAYOUT
# ══════════════════════════════════════════════════════════════

def p3_model_button(name, active=False):
    return html.Div(
        id={"type":"p3-model-btn","index":name}, children=name, n_clicks=0, className="model-btn",
        style={"padding":"9px 14px","marginBottom":"6px","borderRadius":"6px","cursor":"pointer",
               "fontSize":"12px","fontWeight":"700" if active else "500",
               "border":f"2px solid {ACCENT}" if active else f"1px solid {BORDER}",
               "background":ACCENT if active else "#21262d",
               "color":"#0d1117" if active else TEXT,"transition":"all 0.25s ease",
               "textAlign":"center","userSelect":"none",
               "boxShadow":"0 0 10px rgba(88,166,255,0.5)" if active else "none"})

def p3_date_picker(pid, min_d, max_d, init_m, default_d, mb="12px"):
    return dcc.DatePickerSingle(id=pid,min_date_allowed=min_d,max_date_allowed=max_d,
        initial_visible_month=init_m,date=str(default_d.date()) if hasattr(default_d,'date') else str(default_d),
        style={"width":"100%","marginBottom":mb},className="dark-date",display_format="DD-MMM-YYYY")

def phase34_layout():
    if X_TRAIN.empty:
        return html.Div([html.H4("⚠️ ML Data Files Not Found", style={"color":WARN,"padding":"40px"}),
                         html.P("Please ensure x_train2_5.csv, y_train2_5.csv, x_test2_5.csv, y_test2_5.csv are in the same folder.",
                                style={"color":MUTED,"padding":"0 40px"})])
    tr_min=X_TRAIN["Date"].min(); tr_max=X_TRAIN["Date"].max()
    te_min=X_TEST["Date"].min();  te_max=X_TEST["Date"].max()
    cutoff_marks={i:{"label":str(v),"style":{"color":"#a0cfff","fontSize":"11px"}} for i,v in enumerate(CUTOFF_VALUES)}
    return html.Div(style={"background":DARK_BG,"minHeight":"100vh","fontFamily":"'Segoe UI', sans-serif","color":TEXT},children=[
        html.Div(style={"background":"linear-gradient(135deg, #1f2a3c 0%, #0d1117 100%)","borderBottom":f"2px solid {ACCENT}","padding":"22px 40px","marginBottom":"28px"},children=[
            html.H2("📈  ML Models Prediction Dashboard",style={"color":ACCENT,"margin":0,"fontWeight":"700","fontSize":"26px"}),
            html.P("Phase 3 & 4 — Nifty Direction Forecasting | Binary Classification",style={"color":MUTED,"margin":"4px 0 0","fontSize":"13px"}),
        ]),
        dbc.Container(fluid=True,style={"padding":"0 32px"},children=[
            dbc.Row([
                dbc.Col(width=3,children=[
                    html.Div(style=CARD_STYLE,children=[
                        html.P("🗓  TRAINING DATE RANGE",style={**LABEL_STYLE,"marginBottom":"14px"}),
                        html.P("Start Date",style={"color":MUTED,"fontSize":"12px","marginBottom":"4px"}),
                        p3_date_picker("p3-train-start",tr_min,tr_max,tr_min,tr_min),
                        html.P("End Date",style={"color":MUTED,"fontSize":"12px","marginBottom":"4px"}),
                        p3_date_picker("p3-train-end",tr_min,tr_max,tr_max,tr_max,mb="0"),
                    ]),
                    html.Div(style=CARD_STYLE,children=[
                        html.P("🗓  TESTING DATE RANGE",style={**LABEL_STYLE,"marginBottom":"14px"}),
                        html.P("Start Date",style={"color":MUTED,"fontSize":"12px","marginBottom":"4px"}),
                        p3_date_picker("p3-test-start",te_min,te_max,te_min,te_min),
                        html.P("End Date",style={"color":MUTED,"fontSize":"12px","marginBottom":"4px"}),
                        p3_date_picker("p3-test-end",te_min,te_max,te_max,te_max,mb="0"),
                    ]),
                    html.Div(style=CARD_STYLE,children=[
                        html.P("🤖  SELECT ML MODEL",style={**LABEL_STYLE,"marginBottom":"12px"}),
                        *[p3_model_button(n,active=(n==MODEL_NAMES[0])) for n in MODEL_NAMES],
                        dcc.Store(id="p3-selected-model",data=MODEL_NAMES[0]),
                    ]),
                    html.Div(style=CARD_STYLE,children=[
                        html.P("🎚  PROBABILITY CUTOFF (THRESHOLD)",style={**LABEL_STYLE,"marginBottom":"16px"}),
                        dcc.Slider(id="p3-cutoff-slider",min=0,max=len(CUTOFF_VALUES)-1,step=1,value=4,
                                   marks=cutoff_marks,tooltip={"placement":"top","always_visible":True}),
                        html.Div(id="p3-cutoff-display",style={"textAlign":"center","marginTop":"20px","color":GREEN,"fontSize":"20px","fontWeight":"700"}),
                    ]),
                    html.Button("▶  TRAIN & TEST MODEL",id="p3-run-btn",n_clicks=0,
                        style={"width":"100%","padding":"14px","background":f"linear-gradient(135deg, {ACCENT} 0%, #1f6feb 100%)",
                               "color":"#0d1117","border":"none","borderRadius":"8px","fontSize":"14px","fontWeight":"700",
                               "cursor":"pointer","letterSpacing":"0.5px","boxShadow":"0 4px 15px rgba(88,166,255,0.4)"}),
                ]),
                dbc.Col(width=9,children=[
                    html.Div(id="p3-status-bar",style={**CARD_STYLE,"fontSize":"13px","color":MUTED},
                             children="⬅  Configure settings on the left panel and click Train & Test Model."),
                    html.Div(id="p3-metric-cards",style={"marginBottom":"4px"}),
                    dbc.Row([
                        dbc.Col(width=6,children=[html.Div(id="p3-roc-curve-div",style=CARD_STYLE)]),
                        dbc.Col(width=6,children=[html.Div(id="p3-conf-matrix-div",style=CARD_STYLE)]),
                    ]),
                    html.Div(id="p3-clf-report-div",style=CARD_STYLE),
                    html.Div(id="p3-nifty-dir-div",style=CARD_STYLE),
                ]),
            ])
        ])
    ])

# ══════════════════════════════════════════════════════════════
#  PHASE 5 – SENTIMENT HELPERS & LAYOUT
# ══════════════════════════════════════════════════════════════
P5_TAB_S = {'padding':'8px 18px','fontFamily':'Inter,sans-serif','fontSize':'13px','color':MUTED,
             'backgroundColor':CARD_BG,'borderTop':'3px solid transparent','borderBottom':f'1px solid {BORDER}'}
P5_TAB_SEL = {**P5_TAB_S,'color':ACCENT,'borderTop':f'3px solid {ACCENT}','backgroundColor':PANEL_BG}

def clean_text(text):
    text=str(text).lower(); text=re.sub(r'<.*?>','',text); text=re.sub(r'[^a-z0-9\s]','',text)
    return re.sub(r'\s+',' ',text).strip()

def lexicon_sentiment(text):
    tokens=text.split()
    pos=sum(1 for t in tokens if t in positive_words); neg=sum(1 for t in tokens if t in negative_words)
    score=(pos-neg)/np.sqrt(max(len(tokens),1))
    return {"pos_count":pos,"neg_count":neg,"score":score}

def vader_sentiment_fn(text):
    if not NLP_AVAILABLE: return 'N/A'
    c=sid_vader.polarity_scores(text)['compound']
    return 'Positive' if c>=0.05 else('Negative' if c<=-0.05 else 'Neutral')

def finbert_sentiment_fn(text):
    if not NLP_AVAILABLE: return 'N/A'
    inp=_tok(text,return_tensors='pt',truncation=True,padding=True)
    out=_finbert(**inp); prob=torch.softmax(out.logits,dim=1)
    return ['Negative','Neutral','Positive'][torch.argmax(prob).item()]

def process_sentiment_df(df_in):
    df=df_in.copy()
    if '0' in df.columns: df.rename(columns={'0':'text'},inplace=True)
    if 'Unnamed: 0' in df.columns: df.rename(columns={'Unnamed: 0':'idx'},inplace=True)
    df['raw_text']=df.iloc[:,1] if df.shape[1]>1 else df.iloc[:,0]
    df['clean_text']=df['raw_text'].apply(clean_text)
    scores=df['clean_text'].apply(lexicon_sentiment).apply(pd.Series)
    df=pd.concat([df,scores],axis=1)
    df['sentiment_label']=df['score'].apply(lambda x:'positive' if x>0 else('negative' if x<0 else 'neutral'))
    df['lexicon_sentiment']=df['sentiment_label']
    df['vader_sentiment']=df['clean_text'].apply(vader_sentiment_fn)
    if NLP_AVAILABLE:
        print('  Running FinBERT...')
        df['finbert_sentiment']=df['clean_text'].apply(finbert_sentiment_fn)
    else:
        df['finbert_sentiment']='N/A'
    df.fillna('NA',inplace=True)
    return df

def make_wordcloud_b64_fn(df):
    if not NLP_AVAILABLE: return ""
    try:
        wc=WordCloud(width=1400,height=700,background_color='white',collocations=False)
        wc.generate(' '.join(df['clean_text']))
        buf=io.BytesIO(); wc.to_image().save(buf,format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except: return ""

def p5_darkify(fig):
    fig.update_layout(paper_bgcolor=CARD_BG,plot_bgcolor=DARK_BG,
        font=dict(color=TEXT,family='Inter, sans-serif'),title_font=dict(size=15,color=TEXT),
        legend=dict(bgcolor='rgba(0,0,0,0)'),margin=dict(t=55,b=35,l=35,r=35))
    fig.update_xaxes(gridcolor=BORDER,zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER,zerolinecolor=BORDER)
    return fig

def p5_make_table(data_df,cols):
    return dash_table.DataTable(data=data_df[cols].head(50).to_dict('records'),
        columns=[{'name':c,'id':c} for c in cols],page_size=10,
        style_table={'overflowX':'auto','borderRadius':'8px','border':f'1px solid {BORDER}'},
        style_header={'backgroundColor':'#21262d','color':ACCENT,'fontWeight':'600','fontFamily':'Inter,sans-serif','fontSize':'13px'},
        style_data={'backgroundColor':CARD_BG,'color':TEXT,'fontFamily':'Inter,sans-serif','fontSize':'13px','border':f'1px solid {BORDER}'},
        style_cell={'overflow':'hidden','textOverflow':'ellipsis','maxWidth':'320px'},
        style_data_conditional=[{'if':{'row_index':'odd'},'backgroundColor':PANEL_BG}])

def p5_build_figs(df):
    lc=df['lexicon_sentiment'].value_counts().reset_index(); lc.columns=['Sentiment','Count']
    fig_lex_bar=p5_darkify(px.bar(lc,x='Sentiment',y='Count',title='Sentiment Label Counts (Lexicon)',color='Sentiment'))
    fig_lex_hist=p5_darkify(px.histogram(df,x='score',nbins=30,title='Histogram of Lexicon Sentiment Scores'))
    vc=df['vader_sentiment'].value_counts().reset_index(); vc.columns=['Sentiment','Count']
    fig_vader=p5_darkify(px.pie(vc,names='Sentiment',values='Count',title='VADER Sentiment Distribution'))
    fc=df['finbert_sentiment'].value_counts().reset_index(); fc.columns=['Sentiment','Count']
    fig_finbert=p5_darkify(px.pie(fc,names='Sentiment',values='Count',title='FinBERT Sentiment Distribution'))
    cdf=pd.DataFrame({'Sentiment':['Negative','Neutral','Positive'],
        'Lexicon':[(df['lexicon_sentiment']=='negative').sum(),(df['lexicon_sentiment']=='neutral').sum(),(df['lexicon_sentiment']=='positive').sum()],
        'VADER':[(df['vader_sentiment']=='Negative').sum(),(df['vader_sentiment']=='Neutral').sum(),(df['vader_sentiment']=='Positive').sum()],
        'FinBERT':[(df['finbert_sentiment']=='Negative').sum(),(df['finbert_sentiment']=='Neutral').sum(),(df['finbert_sentiment']=='Positive').sum()]})
    fig_cmp=p5_darkify(px.bar(cdf,x='Sentiment',y=['Lexicon','VADER','FinBERT'],barmode='group',title='Comparison: Custom vs VADER vs FinBERT'))
    ddf=pd.DataFrame({'Model':['Custom','VADER','FinBERT'],
        'negative':[(df['sentiment_label']=='negative').sum(),(df['vader_sentiment']=='Negative').sum(),(df['finbert_sentiment']=='Negative').sum()],
        'neutral':[(df['sentiment_label']=='neutral').sum(),(df['vader_sentiment']=='Neutral').sum(),(df['finbert_sentiment']=='Neutral').sum()],
        'positive':[(df['sentiment_label']=='positive').sum(),(df['vader_sentiment']=='Positive').sum(),(df['finbert_sentiment']=='Positive').sum()]})
    fig_div=go.Figure()
    for s in ['negative','neutral','positive']: fig_div.add_bar(name=s,x=ddf['Model'],y=ddf[s])
    fig_div.update_layout(barmode='group',title='Divergence: FinBERT vs VADER',yaxis_title='Number of Headlines')
    p5_darkify(fig_div)
    return fig_lex_bar,fig_lex_hist,fig_vader,fig_finbert,fig_cmp,fig_div

def p5_make_dashboard_content(df,wc_b64,wc_uploaded_b64=None):
    if df.empty:
        return html.Div([html.H5("Upload a web_scrape.csv file to begin",style={"color":MUTED,"padding":"40px","textAlign":"center"})])
    f1,f2,f3,f4,f5,f6=p5_build_figs(df)
    wordcloud_src=f'data:image/png;base64,{wc_uploaded_b64}' if wc_uploaded_b64 else(f'data:image/png;base64,{wc_b64}' if wc_b64 else None)
    return dbc.Tabs(style={'borderRadius':'10px','border':f'1px solid {BORDER}','backgroundColor':CARD_BG,'marginBottom':'6px'},children=[
        dbc.Tab(label='📝 Raw & Clean Text',tab_style=P5_TAB_S,active_tab_style=P5_TAB_SEL,
            children=[html.Div(style={'padding':'20px'},children=[p5_make_table(df,['raw_text','clean_text'])])]),
        dbc.Tab(label='☁️ WordCloud',tab_style=P5_TAB_S,active_tab_style=P5_TAB_SEL,
            children=[html.Div(style={'padding':'30px','textAlign':'center'},children=[
                html.Img(src=wordcloud_src,style={'width':'78%','borderRadius':'12px','border':f'1px solid {BORDER}'}) if wordcloud_src
                else html.P("WordCloud requires the wordcloud library and a processed CSV.",style={"color":MUTED})])]),
        dbc.Tab(label='🔍 Sentiment Preview',tab_style=P5_TAB_S,active_tab_style=P5_TAB_SEL,
            children=[html.Div(style={'padding':'20px'},children=[
                html.H5('Lexicon',style={'color':ACCENT}),
                p5_make_table(df,['raw_text','clean_text','pos_count','neg_count','score','sentiment_label']),
                html.Hr(style={'borderColor':BORDER}),
                html.H5('Model Comparison',style={'color':PURPLE}),
                p5_make_table(df,['clean_text','lexicon_sentiment','vader_sentiment','finbert_sentiment'])])]),
        dbc.Tab(label='📈 Lexicon Analysis',tab_style=P5_TAB_S,active_tab_style=P5_TAB_SEL,
            children=[html.Div(style={'padding':'20px'},children=[dcc.Graph(figure=f1),dcc.Graph(figure=f2)])]),
        dbc.Tab(label='📊 Lexicon Score Details',tab_style=P5_TAB_S,active_tab_style=P5_TAB_SEL,
            children=[html.Div(style={'padding':'20px'},children=[p5_make_table(df,['raw_text','clean_text','pos_count','neg_count','score'])])]),
        dbc.Tab(label='🤖 Model-wise Sentiment',tab_style=P5_TAB_S,active_tab_style=P5_TAB_SEL,
            children=[html.Div(style={'padding':'20px'},children=[dbc.Row([dbc.Col(dcc.Graph(figure=f3),md=6),dbc.Col(dcc.Graph(figure=f4),md=6)])])]),
        dbc.Tab(label='⚖️ Sentiment Comparison',tab_style=P5_TAB_S,active_tab_style=P5_TAB_SEL,
            children=[html.Div(style={'padding':'20px'},children=[dcc.Graph(figure=f5),dcc.Graph(figure=f6)])]),
    ])

def phase5_layout():
    nlp_warn = [] if NLP_AVAILABLE else [
        html.Div(style={'background':'rgba(248,81,73,0.1)','border':f'1px solid {WARN}','borderRadius':'8px',
                        'padding':'10px 20px','margin':'12px 32px 0','fontSize':'13px','color':WARN},
                 children="⚠️  NLP libraries (nltk, transformers, torch, wordcloud) not installed. VADER & FinBERT will show N/A.")]
    return html.Div(style={'backgroundColor':DARK_BG,'minHeight':'100vh','fontFamily':'Inter,sans-serif'},children=[
        html.Div(style={'background':'linear-gradient(90deg,#0d1117 0%,#1a2d4a 50%,#0d1117 100%)',
                        'padding':'14px 32px','display':'flex','alignItems':'center','justifyContent':'space-between',
                        'borderBottom':f'1px solid {BORDER}','boxShadow':'0 2px 16px rgba(0,0,0,0.5)'},
            children=[
                html.Div(children=[html.Span('📊 ',style={'fontSize':'22px'}),
                    html.Span('Sentiment Analysis Dashboard',style={'fontSize':'20px','fontWeight':'700','color':TEXT}),
                    html.Span('  Phase 5',style={'fontSize':'12px','color':ACCENT,'fontWeight':'600',
                        'marginLeft':'10px','background':'rgba(88,166,255,0.13)','padding':'3px 10px','borderRadius':'20px'})]),
                html.Div(style={'display':'flex','gap':'10px','alignItems':'center'},children=[
                    dcc.Upload(id='p5-upload-csv',accept='.csv',children=html.Button('⬆ Upload CSV',style=BTN_BLUE)),
                    dcc.Upload(id='p5-upload-wordcloud',accept='.png,.jpg,.jpeg',children=html.Button('🖼 Upload WordCloud',style={**BTN_BLUE,'background':'linear-gradient(135deg,#4a1d8f 0%,#bc8cff 100%)'})),
                    html.Button('⬇ Download CSV',id='p5-btn-download',style=BTN_GREEN),
                    dcc.Download(id='p5-download-csv'),
                ])
            ]),
        *nlp_warn,
        html.Div(id='p5-status-bar',style={'padding':'8px 32px','backgroundColor':CARD_BG,'borderBottom':f'1px solid {BORDER}',
                                            'fontSize':'12px','color':MUTED,'display':'flex','alignItems':'center','gap':'8px'},
            children=[html.Span('ℹ️'),html.Span('Upload a web_scrape.csv file to start analysis.',style={'color':MUTED})]),
        html.Div(id='p5-main-content',style={'padding':'20px 32px'},children=[
            html.Div(style={'padding':'60px','textAlign':'center'},children=[
                html.Div('📤',style={'fontSize':'48px'}),
                html.H4('Upload your web_scrape.csv file',style={'color':TEXT,'marginTop':'16px'}),
                html.P('Use the "Upload CSV" button above to load your scraped financial news data.',style={'color':MUTED}),
            ])
        ]),
        html.Div(id='p5-toast-container',style={'position':'fixed','bottom':'28px','right':'28px','zIndex':'9999','display':'none'}),
        dcc.Store(id='p5-store-df-json'),
        dcc.Store(id='p5-store-wc-b64'),
        dcc.Store(id='p5-store-wc-generated-b64'),
    ])

# ══════════════════════════════════════════════════════════════
#  MASTER NAV + APP LAYOUT
# ══════════════════════════════════════════════════════════════
NAV_TABS = [
    {"value":"phase1",  "label":"📊 Phase 1",    "sub":"Master Data"},
    {"value":"phase2",  "label":"🔍 Phase 2",    "sub":"EDA"},
    {"value":"phase34", "label":"📈 Phase 3 & 4","sub":"ML Models"},
    {"value":"phase5",  "label":"💬 Phase 5",    "sub":"Sentiment"},
]

app.layout = html.Div(
    style={"background":DARK_BG,"minHeight":"100vh","fontFamily":"'Segoe UI', sans-serif","color":TEXT},
    children=[
        # ══ GLOBAL TOP NAV BAR ═══════════════════════════════
        html.Div(
            style={"background":NAV_BG,"borderBottom":f"1px solid {BORDER}",
                   "display":"flex","alignItems":"stretch","position":"sticky","top":"0","zIndex":"9000",
                   "boxShadow":"0 2px 20px rgba(0,0,0,0.6)"},
            children=[
                html.Div(style={"padding":"14px 28px","display":"flex","alignItems":"center",
                                 "gap":"10px","borderRight":f"1px solid {BORDER}","flexShrink":"0"},
                    children=[
                        html.Div("📉",style={"fontSize":"22px"}),
                        html.Div([
                            html.Div("Analytics Suite",style={"fontSize":"14px","fontWeight":"700","color":TEXT,"lineHeight":"1.1"}),
                            html.Div("Multi-Phase Dashboard",style={"fontSize":"10px","color":MUTED}),
                        ]),
                    ]),
                dcc.Tabs(
                    id="phase-tabs",
                    value="phase1",
                    className="phase-nav",
                    style={"flex":"1","background":"transparent","border":"none","height":"62px"},
                    children=[
                        dcc.Tab(
                            label=f"{t['label']}  {t['sub']}", value=t["value"],
                            style={"padding":"0 28px","fontWeight":"600","fontSize":"13px",
                                   "color":"#8b949e","background":"transparent",
                                   "border":"none","borderBottom":"3px solid transparent",
                                   "height":"62px","display":"flex","alignItems":"center"},
                            selected_style={"padding":"0 28px","fontWeight":"700","fontSize":"13px",
                                            "color":"#58a6ff","background":"rgba(88,166,255,0.09)",
                                            "border":"none","borderBottom":"3px solid #58a6ff",
                                            "height":"62px","display":"flex","alignItems":"center"},
                        )
                        for t in NAV_TABS
                    ]
                ),
            ]
        ),
        # ══ ALL PHASE PANELS — always in DOM, toggled show/hide ═
        html.Div(id="phase1-panel",  style={"display":"block"}, children=[phase1_layout()]),
        html.Div(id="phase2-panel",  style={"display":"none"},  children=[phase2_layout()]),
        html.Div(id="phase34-panel", style={"display":"none"},  children=[phase34_layout()]),
        html.Div(id="phase5-panel",  style={"display":"none"},  children=[phase5_layout()]),
    ]
)

# ══════════════════════════════════════════════════════════════
#  MASTER PHASE-SWITCH CALLBACK
# ══════════════════════════════════════════════════════════════
@app.callback(
    Output("phase1-panel",  "style"),
    Output("phase2-panel",  "style"),
    Output("phase34-panel", "style"),
    Output("phase5-panel",  "style"),
    Input("phase-tabs", "value"),
)
def show_phase(phase):
    on  = {"display":"block"}
    off = {"display":"none"}
    return (
        on  if phase == "phase1"  else off,
        on  if phase == "phase2"  else off,
        on  if phase == "phase34" else off,
        on  if phase == "phase5"  else off,
    )

# ══════════════════════════════════════════════════════════════
#  PHASE 1 CALLBACKS
# ══════════════════════════════════════════════════════════════
@app.callback(
    Output("p1-built-data-store","data"),
    Output("p1-data-table-container","children"),
    Output("p1-row-count-label","children"),
    Output("p1-stats-row","children"),
    Output("p1-status-msg","children"),
    Input("p1-btn-build","n_clicks"),
    State("p1-start-date","date"), State("p1-end-date","date"),
    prevent_initial_call=True,
)
def p1_build(n, start, end):
    if SOURCE_DF.empty:
        return dash.no_update,dash.no_update,dash.no_update,dash.no_update, html.Span("⚠️ df.csv not found.",style={"color":"#f59e0b"})
    if not start or not end:
        return dash.no_update,dash.no_update,dash.no_update,dash.no_update, html.Span("⚠️ Select both dates.",style={"color":"#f59e0b"})
    s=pd.to_datetime(start); e=pd.to_datetime(end)
    if s>=e: return dash.no_update,dash.no_update,dash.no_update,dash.no_update, html.Span("❌ Start must be before End.",style={"color":"#ef4444"})
    mask=(SOURCE_DF["Date"]>=s)&(SOURCE_DF["Date"]<=e); df=SOURCE_DF[mask].copy()
    df["Date"]=df["Date"].dt.strftime("%Y-%m-%d")
    if df.empty: return dash.no_update,dash.no_update,dash.no_update,dash.no_update, html.Span("⚠️ No data in range.",style={"color":"#f59e0b"})
    def stat_card(lbl,val,color):
        return html.Div(style={"background":P1_BG_CARD,"border":f"1px solid {color}","borderLeft":f"4px solid {color}",
                                "borderRadius":"10px","padding":"14px 20px","flex":"1","minWidth":"150px"},
                        children=[html.Div(lbl,style={"color":P1_MUTED,"fontSize":"0.78rem","fontWeight":"600"}),
                                  html.Div(val,style={"color":P1_TEXT,"fontSize":"1.4rem","fontWeight":"700"})])
    close_cols=[c for c in df.columns if c.startswith("Close_")]
    stats=html.Div(style={"display":"flex","gap":"14px","flexWrap":"wrap","marginBottom":"20px"},children=[
        stat_card("Total Rows",f"{len(df):,}",P1_ACCENT1), stat_card("Total Columns",f"{len(df.columns)}",P1_ACCENT2),
        stat_card("Indices",f"{len(close_cols)}",P1_ACCENT3),
        stat_card("Date Range",f"{df['Date'].iloc[0]} → {df['Date'].iloc[-1]}","#a855f7")])
    cols=[{"name":c,"id":c} for c in df.columns]
    table=dash_table.DataTable(id="p1-master-table",columns=cols,data=df.head(20).to_dict("records"),page_size=20,
        style_table={"overflowX":"auto","borderRadius":"8px","border":f"1px solid {P1_BORDER}"},
        style_header={"backgroundColor":P1_BG_HEADER,"color":P1_TEXT,"fontWeight":"700","fontSize":"0.75rem","border":f"1px solid {P1_BORDER}","textAlign":"center"},
        style_cell={"backgroundColor":P1_BG_CARD,"color":P1_TEXT,"fontSize":"0.78rem","border":f"1px solid {P1_BORDER}","textAlign":"center","padding":"8px 10px","minWidth":"90px","maxWidth":"160px"},
        sort_action="native",filter_action="native")
    return df.to_json(date_format="iso",orient="split"), table, f"Showing first 20 of {len(df):,} rows", stats, \
           html.Span(f"✅ {len(df):,} rows built",style={"color":P1_ACCENT2,"fontWeight":"600"})

@app.callback(Output("p1-status-msg","children",allow_duplicate=True), Output("p1-saved-flag","data"),
    Input("p1-btn-save","n_clicks"), State("p1-built-data-store","data"), prevent_initial_call=True)
def p1_save(n,json_data):
    if not json_data: return html.Span("⚠️ Build data first.",style={"color":"#f59e0b"}), False
    df=pd.read_json(io.StringIO(json_data),orient="split"); df.to_csv("master_data.csv",index=False)
    return html.Span(f"💾 Saved → master_data.csv ({len(df):,} rows)",style={"color":P1_ACCENT2,"fontWeight":"600"}), True

@app.callback(Output("p1-download-csv","data"), Output("p1-status-msg","children",allow_duplicate=True),
    Input("p1-btn-download","n_clicks"), State("p1-built-data-store","data"), prevent_initial_call=True)
def p1_download(n,json_data):
    if not json_data: return dash.no_update, html.Span("⚠️ Build data first.",style={"color":"#f59e0b"})
    df=pd.read_json(io.StringIO(json_data),orient="split")
    return dcc.send_data_frame(df.to_csv,"master_data.csv",index=False), html.Span("⬇️ Downloading…",style={"color":P1_ACCENT3,"fontWeight":"600"})

# ══════════════════════════════════════════════════════════════
#  PHASE 2 CALLBACKS
# ══════════════════════════════════════════════════════════════
@app.callback(
    Output('p2-tab1-panel','style'),
    Output('p2-tab2-panel','style'),
    Output('p2-tab3-panel','style'),
    Input('p2-main-tabs','value'),
)
def p2_switch_tab(tab):
    on  = {'padding':'22px 28px', 'display':'block'}
    off = {'padding':'22px 28px', 'display':'none'}
    return (
        on  if tab == 'tab1' else off,
        on  if tab == 'tab2' else off,
        on  if tab == 'tab3' else off,
    )

@app.callback(
    Output('p2-t1-plots-container','children'),
    Output('p2-t1-status','children'),
    Output('p2-t1-status','style'),
    Input('p2-t1-btn','n_clicks'),
    State('p2-t1-indices','value'),
    State('p2-t1-plot-types','value'),
    prevent_initial_call=True,
)
def p2_cb_t1(n, indices, plot_types):
    if not indices or not plot_types:
        return (html.P('Select at least one index and plot type.',
                        style={'color':P2_DNG}), '', {'display':'none'})
    cfg = {'displayModeBar':False,'responsive':True}

    def gcard(fig, title, dot=P2_ACCENT, wide=False):
        return html.Div(
            style={**P2_CARD_S,'gridColumn':'1 / -1' if wide else 'auto'},
            children=[
                html.Div(style={'display':'flex','alignItems':'center',
                                'gap':'8px','marginBottom':'6px'}, children=[
                    html.Div(style={'width':'8px','height':'8px',
                                    'borderRadius':'50%','background':dot,'flexShrink':'0'}),
                    html.Strong(title, style={'color':P2_TXT,'fontSize':'0.87rem'}),
                ]),
                dcc.Graph(figure=fig, config=cfg),
            ])

    def tcard(content, title):
        return html.Div(
            style={**P2_CARD_S,'gridColumn':'1 / -1'},
            children=[
                html.Div(style={'display':'flex','alignItems':'center',
                                'gap':'8px','marginBottom':'10px'}, children=[
                    html.Div(style={'width':'8px','height':'8px',
                                    'borderRadius':'50%','background':P2_GOLD,'flexShrink':'0'}),
                    html.Strong(title, style={'color':P2_TXT,'fontSize':'0.87rem'}),
                ]),
                content,
            ])

    items = []
    if 'Boxplot by Year'              in plot_types:
        items.append(gcard(fig_boxplot(indices), 'Boxplot by Year - Returns Distribution', P2_ACCENT, True))
    if 'Yearly Stats (Chart + Table)' in plot_types:
        items.append(gcard(fig_yearly_stats(indices), 'Yearly Stats - Mean and Std Dev', '#ff6b9d', True))
        td, tc = tbl_yearly_stats(indices)
        items.append(tcard(p2_make_tbl(td, tc), 'Yearly Statistics Table - N Obs / Mean / Std Dev'))
    if 'Median Bar by Year'           in plot_types:
        items.append(gcard(fig_median_bar(indices), 'Median Bar by Year', '#00e5c0', True))
    if 'Heatmap Year x Quarter'       in plot_types:
        items.append(gcard(fig_heatmap_yq(indices[0]), f'Heatmap Year x Quarter - {indices[0]}', P2_GOLD))
    if 'Multivariate Median (All 7)'  in plot_types:
        items.append(gcard(fig_multivariate_median(), 'Multivariate Median - All 7 Indices', '#a78bfa', True))
    if 'Rolling Volatility-30d'       in plot_types:
        items.append(gcard(fig_rolling_vol(indices), 'Rolling Volatility-30d', '#f97316', True))
    if 'Cumulative Returns'           in plot_types:
        items.append(gcard(fig_cumret(indices), 'Cumulative Returns', P2_SUC, True))
    if 'Drawdown'                     in plot_types:
        items.append(gcard(fig_drawdown(indices), 'Drawdown Analysis', P2_DNG, True))
    if 'Correlation - 6 Years'        in plot_types:
        items.append(gcard(fig_correlation(6), 'Correlation - Last 6 Years', P2_ACCENT))
    if 'Correlation - Latest Year'    in plot_types:
        items.append(gcard(fig_correlation(1), 'Correlation - Latest Year', '#ff6b9d'))
    if 'Nifty Opening Direction'      in plot_types:
        nd_fig, _ = fig_nifty_dir()
        items.append(gcard(nd_fig, 'Nifty Opening Direction', '#00e5c0', True))
    if 'Opening Direction Table'      in plot_types:
        _, nd_stats = fig_nifty_dir()
        td, tc = tbl_nifty_dir(nd_stats)
        sc = [
            {'if':{'filter_query':'{pct_up} > 50','column_id':'pct_up'},'color':P2_SUC},
            {'if':{'filter_query':'{pct_up} <= 50','column_id':'pct_up'},'color':P2_DNG},
        ]
        items.append(tcard(p2_make_tbl(td, tc, sc), 'Opening Direction Table - Year-wise Summary'))

    grid = html.Div(items, style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':'14px'})
    status_msg = [html.Div(style={'width':'6px','height':'6px','borderRadius':'50%','background':P2_SUC}),
                  f'  {len(items)} chart(s) for: {", ".join(indices)}']
    status_style = {'display':'flex','alignItems':'center','gap':'6px','padding':'7px 14px',
                    'marginBottom':'10px','background':'rgba(108,140,255,0.08)',
                    'border':'1px solid rgba(108,140,255,0.2)','borderRadius':'8px',
                    'fontSize':'0.78rem','color':P2_ACCENT,'fontWeight':'600'}
    return grid, status_msg, status_style

@app.callback(
    Output('p2-t2-plots-container','children'),
    Output('p2-t2-status','children'),
    Output('p2-t2-status','style'),
    Input('p2-t2-btn','n_clicks'),
    State('p2-t2-x','value'),
    State('p2-t2-y','value'),
    State('p2-t2-plot-types','value'),
    prevent_initial_call=True,
)
def p2_cb_t2(n, xi, yi, plot_types):
    if not xi or not yi or not plot_types:
        return (html.P('Select Index X, Y and a plot type.', style={'color':P2_DNG}), '', {'display':'none'})
    cfg = {'displayModeBar':False,'responsive':True}

    def gcard(fig, title, dot=P2_ACCENT, wide=False):
        return html.Div(
            style={**P2_CARD_S,'gridColumn':'1 / -1' if wide else 'auto'},
            children=[
                html.Div(style={'display':'flex','alignItems':'center',
                                'gap':'8px','marginBottom':'6px'}, children=[
                    html.Div(style={'width':'8px','height':'8px',
                                    'borderRadius':'50%','background':dot,'flexShrink':'0'}),
                    html.Strong(title, style={'color':P2_TXT,'fontSize':'0.87rem'}),
                ]),
                dcc.Graph(figure=fig, config=cfg),
            ])

    items = []
    if 'Scatter X vs Y Returns'       in plot_types:
        items.append(gcard(fig_scatter_xy(xi,yi), f'Scatter: {xi} vs {yi} Returns', IDX_COLORS[xi]))
    if 'Scatter + OLS Trendline'      in plot_types:
        items.append(gcard(fig_scatter_ols(xi,yi), f'Scatter + OLS: {xi} vs {yi}', IDX_COLORS[yi]))
    if 'Scatter Return vs VIX'        in plot_types:
        items.append(gcard(fig_scatter_vix(xi), f'{xi} Return vs VIX', P2_DNG))
    if 'Scatter Matrix (All Indices)' in plot_types:
        items.append(gcard(fig_scatter_matrix(), 'Scatter Matrix - All Indices', '#a78bfa', True))
    if 'Pairwise Correlation Heatmap' in plot_types:
        items.append(gcard(fig_pair_corr(), 'Pairwise Correlation Heatmap', P2_GOLD))

    grid = html.Div(items, style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':'14px'})
    status_msg = [html.Div(style={'width':'6px','height':'6px','borderRadius':'50%','background':P2_SUC}),
                  f'  Charts for {xi} x {yi}']
    status_style = {'display':'flex','alignItems':'center','gap':'6px','padding':'7px 14px',
                    'marginBottom':'10px','background':'rgba(108,140,255,0.08)',
                    'border':'1px solid rgba(108,140,255,0.2)','borderRadius':'8px',
                    'fontSize':'0.78rem','color':P2_ACCENT,'fontWeight':'600'}
    return grid, status_msg, status_style

@app.callback(Output('p2-fe-ret-graph','figure'),
              Input('p2-fe-ret-btn','n_clicks'), prevent_initial_call=True)
def p2_cb_fe_ret(_):  return fig_fe_returns_dist()

@app.callback(Output('p2-fe-lag-graph','figure'),
              Input('p2-fe-lag-btn','n_clicks'), prevent_initial_call=True)
def p2_cb_fe_lag(_):  return fig_fe_lag()

@app.callback(Output('p2-fe-vix-graph','figure'),
              Input('p2-fe-vix-btn','n_clicks'), prevent_initial_call=True)
def p2_cb_fe_vix(_):  return fig_fe_vix()

@app.callback(Output('p2-fe-time-graph','figure'),
              Input('p2-fe-time-btn','n_clicks'), prevent_initial_call=True)
def p2_cb_fe_time(_): return fig_fe_time()

@app.callback(Output('p2-fe-tgt-graph','figure'),
              Input('p2-fe-tgt-btn','n_clicks'), prevent_initial_call=True)
def p2_cb_fe_tgt(_):  return fig_fe_target()

@app.callback(Output('p2-fe-vif-table','children'),
              Input('p2-fe-vif-btn','n_clicks'), prevent_initial_call=True)
def p2_cb_fe_vif(_):
    rows, cols = tbl_vif()
    sc = [
        {'if':{'filter_query':'{Status} = \'OK\'',       'column_id':'Status'},'color':P2_SUC},
        {'if':{'filter_query':'{Status} = \'Moderate\'', 'column_id':'Status'},'color':P2_GOLD},
        {'if':{'filter_query':'{Status} = \'High\'',     'column_id':'Status'},'color':P2_DNG},
    ]
    return p2_make_tbl(rows, cols, sc)

# ══════════════════════════════════════════════════════════════
#  PHASE 3 & 4 CALLBACKS
# ══════════════════════════════════════════════════════════════
@app.callback(
    [Output({"type":"p3-model-btn","index":n},"style") for n in MODEL_NAMES]+[Output("p3-selected-model","data")],
    [Input({"type":"p3-model-btn","index":n},"n_clicks") for n in MODEL_NAMES],
    State("p3-selected-model","data"),
)
def p3_toggle_model(*args):
    n_m=len(MODEL_NAMES); clicks=list(args[:n_m]); current=args[n_m]
    ctx=callback_context
    if not ctx.triggered or all(c==0 for c in clicks): selected=current
    else:
        try: selected=json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
        except: selected=current
    styles=[]
    for name in MODEL_NAMES:
        active=name==selected
        styles.append({"padding":"9px 14px","marginBottom":"6px","borderRadius":"6px","cursor":"pointer","fontSize":"12px",
                        "fontWeight":"700" if active else "500","border":f"2px solid {ACCENT}" if active else f"1px solid {BORDER}",
                        "background":ACCENT if active else "#21262d","color":"#0d1117" if active else TEXT,
                        "transition":"all 0.25s ease","textAlign":"center","userSelect":"none",
                        "boxShadow":"0 0 10px rgba(88,166,255,0.5)" if active else "none"})
    return styles+[selected]

@app.callback(Output("p3-cutoff-display","children"), Input("p3-cutoff-slider","value"))
def p3_show_cutoff(idx): return f"Selected Threshold : {CUTOFF_VALUES[idx]:.2f}"

@app.callback(
    Output("p3-status-bar","children"), Output("p3-metric-cards","children"),
    Output("p3-roc-curve-div","children"), Output("p3-conf-matrix-div","children"),
    Output("p3-clf-report-div","children"), Output("p3-nifty-dir-div","children"),
    Input("p3-run-btn","n_clicks"),
    State("p3-train-start","date"), State("p3-train-end","date"),
    State("p3-test-start","date"),  State("p3-test-end","date"),
    State("p3-selected-model","data"), State("p3-cutoff-slider","value"),
    prevent_initial_call=True,
)
def p3_run_model(_,tr_start,tr_end,te_start,te_end,model_name,cutoff_idx):
    if X_TRAIN.empty: return "⚠️ Data files not found.",[],[],[],[],[]
    cutoff=CUTOFF_VALUES[cutoff_idx]
    tr_s=pd.Timestamp(tr_start); tr_e=pd.Timestamp(tr_end)
    te_s=pd.Timestamp(te_start); te_e=pd.Timestamp(te_end)
    x_tr=X_TRAIN[(X_TRAIN["Date"]>=tr_s)&(X_TRAIN["Date"]<=tr_e)].copy()
    y_tr=Y_TRAIN[(Y_TRAIN["Date"]>=tr_s)&(Y_TRAIN["Date"]<=tr_e)].copy()
    x_te=X_TEST [(X_TEST ["Date"]>=te_s)&(X_TEST ["Date"]<=te_e)].copy()
    y_te=Y_TEST [(Y_TEST ["Date"]>=te_s)&(Y_TEST ["Date"]<=te_e)].copy()
    if len(x_tr)==0 or len(x_te)==0: return "⚠️ No data in selected range.",[],[],[],[],[]
    X_tr=x_tr[FEATURE_COLS]; y_tr_flat=y_tr["nifty_dir"].values.ravel()
    X_te=x_te[FEATURE_COLS]; y_te_flat=y_te["nifty_dir"].values.ravel()
    clf=Pipeline([("scaler",StandardScaler()),("model",ML_MODELS[model_name])])
    clf.fit(X_tr,y_tr_flat)
    y_prob=clf.predict_proba(X_te)[:,1]; y_pred=(y_prob>=cutoff).astype(int)
    acc=accuracy_score(y_te_flat,y_pred); auc_val=roc_auc_score(y_te_flat,y_prob)
    cm=confusion_matrix(y_te_flat,y_pred); TN,FP,FN,TP=cm.ravel()
    sens=TP/(TP+FN) if (TP+FN)>0 else 0.; spec=TN/(TN+FP) if (TN+FP)>0 else 0.
    rpt=classification_report(y_te_flat,y_pred,output_dict=True); fpr,tpr,_=roc_curve(y_te_flat,y_prob)
    status=html.Div([html.Span("✅  Trained & tested  |  ",style={"color":GREEN}),
                     html.Span(f"Model: {model_name}  |  Threshold: {cutoff:.2f}  |  Train: {len(x_tr)} rows  |  Test: {len(x_te)} rows",style={"color":MUTED})])
    def mcard(lbl,val,color):
        return dbc.Col(html.Div([html.P(lbl,style={"color":MUTED,"fontSize":"11px","marginBottom":"4px","textTransform":"uppercase"}),
                                  html.H4(val,style={"color":color,"margin":0,"fontWeight":"700"})],style=METRIC_BOX_P3),width=2)
    cards=dbc.Row([mcard("Model",model_name.split("(")[0].strip(),ACCENT),mcard("Threshold",f"{cutoff:.2f}",GREEN),
                   mcard("Accuracy",f"{acc:.4f}",GREEN),mcard("ROC-AUC",f"{auc_val:.4f}",ACCENT),
                   mcard("Sensitivity",f"{sens:.4f}",WARN),mcard("Specificity",f"{spec:.4f}",WARN)],className="g-2",style={"marginBottom":"16px"})
    roc_fig=go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",line=dict(color=ACCENT,width=2.5),name=f"AUC={auc_val:.4f}",fill="tozeroy",fillcolor="rgba(88,166,255,0.08)"))
    roc_fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(color=MUTED,dash="dash",width=1),name="Random"))
    roc_fig.update_layout(title=dict(text=f"ROC Curve — {model_name}",font=dict(color=TEXT,size=13)),
        xaxis=dict(title="FPR",color=MUTED,gridcolor=BORDER),yaxis=dict(title="TPR",color=MUTED,gridcolor=BORDER),
        paper_bgcolor=CARD_BG,plot_bgcolor=CARD_BG,font=dict(color=TEXT),legend=dict(bgcolor=CARD_BG),
        margin=dict(l=40,r=10,t=45,b=40),height=320)
    cm_fig=go.Figure(go.Heatmap(z=cm.tolist(),x=["Predicted 0","Predicted 1"],y=["Actual 0","Actual 1"],
        colorscale=[[0,"#21262d"],[1,ACCENT]],showscale=False,
        text=[[f"TN={TN}",f"FP={FP}"],[f"FN={FN}",f"TP={TP}"]],texttemplate="%{text}",textfont=dict(size=15,color="white")))
    cm_fig.update_layout(title=dict(text="Confusion Matrix",font=dict(color=TEXT,size=13)),
        paper_bgcolor=CARD_BG,plot_bgcolor=CARD_BG,font=dict(color=TEXT),
        xaxis=dict(color=TEXT),yaxis=dict(color=TEXT),margin=dict(l=60,r=10,t=45,b=40),height=320)
    th={"color":MUTED,"padding":"8px 12px","borderBottom":f"1px solid {BORDER}","textAlign":"left"}
    clf_rows=[]
    for lbl in ["0","1","macro avg","weighted avg"]:
        if lbl in rpt:
            r=rpt[lbl]
            clf_rows.append(html.Tr([html.Td(lbl,style={"color":ACCENT,"padding":"8px 12px"}),
                html.Td(f"{r.get('precision',0):.4f}",style={"color":TEXT,"padding":"8px 12px"}),
                html.Td(f"{r.get('recall',0):.4f}",style={"color":TEXT,"padding":"8px 12px"}),
                html.Td(f"{r.get('f1-score',0):.4f}",style={"color":TEXT,"padding":"8px 12px"}),
                html.Td(str(int(r.get("support",0))),style={"color":MUTED,"padding":"8px 12px"})]))
    clf_div=html.Div([html.P("📋  CLASSIFICATION REPORT",style=LABEL_STYLE),
        html.Table([html.Thead(html.Tr([html.Th(h,style=th) for h in ["Class","Precision","Recall","F1-Score","Support"]])),html.Tbody(clf_rows)],
                   style={"width":"100%","borderCollapse":"collapse"}),
        html.Div(f"Accuracy: {rpt.get('accuracy',acc):.4f}  |  Train: {len(x_tr)}  |  Test: {len(x_te)}",
                 style={"color":MUTED,"fontSize":"12px","marginTop":"12px","borderTop":f"1px solid {BORDER}","paddingTop":"10px"})])
    nth={"color":MUTED,"padding":"8px 12px","borderBottom":f"1px solid {BORDER}","textAlign":"left","position":"sticky","top":0,"background":CARD_BG}
    nifty_rows=[html.Tr([
        html.Td(str(pd.Timestamp(d).date()),style={"color":MUTED,"padding":"6px 12px"}),
        html.Td(f"{prob:.4f}",style={"color":TEXT,"padding":"6px 12px"}),
        html.Td("↑ 1 (Up)" if pred==1 else "↓ 0 (Down)",style={"color":GREEN if pred==1 else WARN,"padding":"6px 12px","fontWeight":"600"}),
        html.Td("↑ 1 (Up)" if actual==1 else "↓ 0 (Down)",style={"color":GREEN if actual==1 else WARN,"padding":"6px 12px"}),
        html.Td("✅" if pred==actual else "❌",style={"padding":"6px 12px","textAlign":"center"}),
    ]) for d,prob,pred,actual in zip(x_te["Date"].values,y_prob,y_pred,y_te_flat)]
    nifty_div=html.Div([html.P("📊  NIFTY DIRECTION PREDICTIONS",style=LABEL_STYLE),
        html.Div(html.Table([html.Thead(html.Tr([html.Th(h,style=nth) for h in ["Date","Probability","Predicted Direction","Actual Direction","Correct"]])),
                              html.Tbody(nifty_rows)],style={"width":"100%","borderCollapse":"collapse"}),
                 style={"overflowY":"auto","maxHeight":"320px"})])
    return (status, cards,
        [html.P("📉  ROC CURVE",style=LABEL_STYLE), dcc.Graph(figure=roc_fig,config={"displayModeBar":False})],
        [html.P("🔲  CONFUSION MATRIX",style=LABEL_STYLE), dcc.Graph(figure=cm_fig,config={"displayModeBar":False})],
        clf_div, nifty_div)

# ══════════════════════════════════════════════════════════════
#  PHASE 5 CALLBACKS
# ══════════════════════════════════════════════════════════════
@app.callback(
    Output("p5-main-content","children"), Output("p5-status-bar","children"),
    Output("p5-toast-container","children"), Output("p5-toast-container","style"),
    Output("p5-store-df-json","data"), Output("p5-store-wc-generated-b64","data"),
    Input("p5-upload-csv","contents"), State("p5-upload-csv","filename"),
    State("p5-store-wc-b64","data"), prevent_initial_call=True,
)
def p5_on_csv_upload(contents,filename,uploaded_wc_b64):
    toast_show={"position":"fixed","bottom":"28px","right":"28px","zIndex":"9999","display":"block"}
    if contents is None: raise dash.exceptions.PreventUpdate
    try:
        _,content_str=contents.split(","); raw_bytes=base64.b64decode(content_str)
        df_new=pd.read_csv(io.StringIO(raw_bytes.decode("utf-8")))
        df_new=process_sentiment_df(df_new)
        wc_new=make_wordcloud_b64_fn(df_new)
        dashboard=p5_make_dashboard_content(df_new,wc_new,uploaded_wc_b64)
        status=[html.Span("✅ "),html.Span(f"Uploaded: {filename}",style={"color":GREEN,"fontWeight":"600"}),
                html.Span(f" · {len(df_new)} rows · Lexicon · VADER · FinBERT",style={"color":MUTED})]
        toast=html.Div(style={"background":"linear-gradient(135deg,#196c2e,#3fb950)","color":"#fff",
                                "padding":"14px 22px","borderRadius":"10px","fontFamily":"Inter,sans-serif",
                                "fontSize":"14px","boxShadow":"0 6px 24px rgba(63,185,80,0.4)",
                                "display":"flex","alignItems":"center","gap":"10px","minWidth":"280px"},
            children=[html.Span("✅",style={"fontSize":"20px"}),
                      html.Div([html.Div("CSV uploaded successfully!",style={"fontWeight":"700"}),
                                html.Div(f"{filename} · {len(df_new)} rows",style={"fontSize":"12px","opacity":"0.85"})])])
        return [dashboard],status,toast,toast_show,df_new.to_json(date_format="iso",orient="split"),wc_new
    except Exception as e:
        err_toast=html.Div(style={"background":"linear-gradient(135deg,#6a0c0c,#f85149)","color":"#fff",
                                   "padding":"14px 22px","borderRadius":"10px","minWidth":"280px"},
            children=[html.Div("❌ Upload Error",style={"fontWeight":"700"}),html.Div(str(e),style={"fontSize":"12px","marginTop":"4px"})])
        return dash.no_update,[html.Span("❌ Error: "),html.Span(str(e),style={"color":WARN})],err_toast,toast_show,dash.no_update,dash.no_update

@app.callback(
    Output("p5-store-wc-b64","data"),
    Output("p5-toast-container","children",allow_duplicate=True),
    Output("p5-toast-container","style",allow_duplicate=True),
    Input("p5-upload-wordcloud","contents"),
    State("p5-upload-wordcloud","filename"),
    State("p5-store-df-json","data"),
    State("p5-store-wc-generated-b64","data"),
    prevent_initial_call=True)
def p5_on_wc_upload(contents,filename,df_json,wc_gen_b64):
    toast_show={"position":"fixed","bottom":"28px","right":"28px","zIndex":"9999","display":"block"}
    if contents is None: raise dash.exceptions.PreventUpdate
    _,content_str=contents.split(",")
    toast=html.Div(style={"background":"linear-gradient(135deg,#4a1d8f,#bc8cff)","color":"#fff","padding":"14px 22px","borderRadius":"10px",
                           "fontFamily":"Inter,sans-serif","fontSize":"14px","minWidth":"280px"},
        children=[html.Span("🖼",style={"fontSize":"20px"}),
                  html.Div([html.Div("WordCloud uploaded!",style={"fontWeight":"700"}),
                            html.Div(f"{filename} — see ☁️ WordCloud tab",style={"fontSize":"12px","opacity":"0.85"})])])
    return content_str,toast,toast_show

@app.callback(
    Output("p5-main-content","children",allow_duplicate=True),
    Input("p5-store-wc-b64","data"),
    State("p5-store-df-json","data"),
    State("p5-store-wc-generated-b64","data"),
    prevent_initial_call=True)
def p5_refresh_wc(uploaded_wc_b64,df_json,wc_gen_b64):
    if uploaded_wc_b64 is None: raise dash.exceptions.PreventUpdate
    if df_json: df=pd.read_json(io.StringIO(df_json),orient="split")
    else: return dash.no_update
    return [p5_make_dashboard_content(df,wc_gen_b64 or "",uploaded_wc_b64)]

@app.callback(
    Output("p5-download-csv","data"),
    Input("p5-btn-download","n_clicks"),
    State("p5-store-df-json","data"),
    prevent_initial_call=True)
def p5_download(n,df_json):
    if df_json:
        df=pd.read_json(io.StringIO(df_json),orient="split")
        return dcc.send_data_frame(df.to_csv,"sentiment_results.csv",index=False)
    return dash.no_update

# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
PORT = int(os.environ.get("PORT", 8050))

if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  Multi-Phase Analytics Dashboard")
    print(f"  Open: http://0.0.0.0:{PORT}")
    print("═"*60 + "\n")
    app.run_server(debug=False, host="0.0.0.0", port=PORT)