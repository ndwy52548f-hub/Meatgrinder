"""
app.py — Meatgrinder Performance Analytics
Clean professional UI. Works within Streamlit's actual CSS constraints.
Light theme so native widgets blend naturally.
"""

import hmac
import numpy as np
import pandas as pd
import streamlit as st
from report import build_report
import re as _re

from analytics import (
    geo_return, ann_vol, sharpe_with_rf, sortino_with_rf,
    max_drawdown, calmar, skewness, excess_kurtosis, jarque_bera,
    compute_outliers, normality_tests, top_drawdowns, period_stats,
    seasonality, macro_events_table, piecewise_beta_regression,
    build_exec_summary, build_ddq,
    sharpe_significance, tail_risk, capture_ratios, desmooth_stats,
    acf, acf_conf_band, rolling_metrics, parse_uploaded_file,
    MSCI_DF, AGG_DF, HFRX_INDICES,
)
from pdf_extract import extract_return_series
from charts import (
    chart_cumulative, chart_drawdowns, chart_monthly_bars,
    chart_histogram, chart_qq, chart_acf, chart_calendar_heatmap,
    chart_rolling, chart_regression, chart_seasonality_monthly,
    chart_seasonality_quarterly, chart_best_worst, chart_up_down_capture, chart_waterfall, chart_shocks, chart_desmooth, MN,
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Argus',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
# Design rules:
#   1. Only target selectors verified to work in Streamlit 1.58
#   2. Light background so native widgets blend naturally
#   3. All custom elements (stat banner, tables, headers) carry the design
#   4. No fighting widget internals

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Confirmed working: page background and layout ── */
.stApp {
  background: #F1F6F6 !important;
}
.main .block-container {
  padding: 0 0 80px 0 !important;
  max-width: 100% !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"]        { display: none !important; }
[data-testid="stDecoration"]     { display: none !important; }
[data-testid="stStatusWidget"]   { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Controls area — white strip, full width ── */
[data-testid="stHorizontalBlock"]:first-of-type {
  background: #FFFFFF;
  padding: 6px 0;
  gap: 24px !important;
}

/* Keep column rows on a single line; let columns shrink to fit */
[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
[data-testid="stHorizontalBlock"] > div { min-width: 0 !important; }

/* Input section header */
.mg-input-hdr {
  font-family: 'Inter', sans-serif;
  font-size: 22px;
  font-weight: 700;
  color: #006B7A;
  letter-spacing: -0.5px;
  background: #FFFFFF;
  padding: 10px 0 0;
}

/* Outlier toggle — teal active state to match the buttons; tinted off-state */
[data-baseweb="checkbox"] [aria-checked="true"] {
  background-color: #006B7A !important;
  border-color: #006B7A !important;
}
[data-baseweb="checkbox"] [aria-checked="false"] {
  background-color: #BCD7DB !important;
  border-color: #A6C7CC !important;
}

/* Compact the file uploader: hide the limit caption, shrink to the button */
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploader"] { min-width: 0 !important; width: fit-content !important; }
[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"] {
  padding: 6px 10px !important;
  min-height: 0 !important;
  min-width: 0 !important;
  width: fit-content !important;
}

/* Buttons — solid teal, thick border, sized to content */
[data-testid="stButton"] button {
  border-radius: 6px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 8px 24px !important;
}
[data-testid="stButton"] button[kind="primary"] {
  background: #006B7A !important;
  color: #FFFFFF !important;
  border: 2px solid #005561 !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
  background: #005561 !important;
  border-color: #003E47 !important;
  color: #FFFFFF !important;
}
[data-testid="stButton"] button[kind="secondary"] {
  background: #FFFFFF !important;
  color: #006B7A !important;
  border: 2px solid #9CC3C9 !important;
}
[data-testid="stButton"] button[kind="secondary"]:hover {
  background: #EAF3F4 !important;
  border-color: #006B7A !important;
  color: #006B7A !important;
}

/* ── Widget label text ── */
label[data-testid="stWidgetLabel"] p {
  font-family: 'Inter', sans-serif !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 1.2px !important;
  color: #2A4F57 !important;
}

/* ── Tabs — bordered, clearly-clickable boxes ── */
.stTabs [data-baseweb="tab-list"] {
  background: #FFFFFF !important;
  border-bottom: 1px solid #D2E0E0 !important;
  padding: 16px 40px !important;
  gap: 8px !important;
  flex-wrap: wrap !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Inter', sans-serif !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  color: #4A4943 !important;
  padding: 9px 16px !important;
  border-radius: 6px !important;
  border: 1px solid #CFCAC0 !important;
  background: #FFFFFF !important;
  margin: 0 !important;
  transition: border-color 0.12s ease, color 0.12s ease, background 0.12s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
  border-color: #006B7A !important;
  color: #006B7A !important;
}
.stTabs [aria-selected="true"] {
  color: #FFFFFF !important;
  background: #006B7A !important;
  border: 1px solid #006B7A !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }
.stTabs [data-baseweb="tab-panel"] {
  padding: 28px 40px 0 !important;
}

/* ── Custom HTML elements — these we own 100% ── */

/* Top bar */
.mg-topbar {
  background: #006B7A;
  height: 54px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0;
}
.mg-topbar .brand {
  font-family: 'Inter', sans-serif;
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 5px;
  color: #FFFFFF;
  text-transform: uppercase;
}
.mg-topbar .info {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #BFE4E4;
  letter-spacing: 0.5px;
}
.mg-topbar .fund {
  font-family: 'Inter', sans-serif;
  font-size: 16px;
  font-weight: 600;
  color: #FFFFFF;
  letter-spacing: 1px;
  max-width: 60vw;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Fund identity */
.mg-id {
  padding: 20px 40px 16px;
  background: #FFFFFF;
  border-bottom: 1px solid #D2E0E0;
}
.mg-id .fname {
  font-family: 'Inter', sans-serif;
  font-size: 24px;
  font-weight: 700;
  color: #1A1A1A;
  letter-spacing: -0.5px;
  margin-bottom: 4px;
}
.mg-id .fmeta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: #6A6960;
  letter-spacing: 0.3px;
}

/* Diagnostic strip */
.mg-diag {
  background: #EAF2F2;
  border-bottom: 1px solid #D2E0E0;
  padding: 8px 40px;
  display: flex;
  gap: 48px;
  align-items: center;
}
.mg-diag .d-item { display: flex; flex-direction: column; gap: 2px; }
.mg-diag .d-lbl {
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: #5F5E56;
}
.mg-diag .d-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: #1A1A1A;
  font-weight: 500;
}

/* Outlier alert */
.mg-alert {
  padding: 10px 40px;
  background: #FFFBF0;
  border-bottom: 1px solid #F0D890;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: #8A6000;
}

/* Stat banner */
.mg-stats {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  background: #006B7A;
  border-bottom: 3px solid #006B7A;
}
.mg-stat {
  padding: 20px 24px 18px;
  border-right: 1px solid #1A8290;
}
.mg-stat:last-child { border-right: none; }
.mg-stat .s-lbl {
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: #BFE4E4;
  margin-bottom: 8px;
}
.mg-stat .s-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 24px;
  font-weight: 400;
  color: #FFFFFF;
  line-height: 1;
}
.mg-stat .s-val.pos { color: #3DD68C; }
.mg-stat .s-val.neg { color: #FF6B6B; }
.mg-stat .s-val.dim { color: #CFE2E2; }

/* Section heading */
.mg-sh {
  font-family: 'Inter', sans-serif;
  font-size: 14px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: #006B7A;
  padding-bottom: 10px;
  border-bottom: 1px solid #D2E0E0;
  margin-bottom: 16px;
  margin-top: 0;
}

/* Data tables */
table.mg-tbl {
  width: 100%;
  border-collapse: collapse;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  border: 2px solid #006B7A;
  border-radius: 6px;
  overflow: hidden;
  background: #FFFFFF;
}
table.mg-tbl th {
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: #005561;
  background: #EAF2F2;
  padding: 10px 14px;
  text-align: right;
  border-bottom: 1px solid #D2E0E0;
  white-space: nowrap;
}
table.mg-tbl td {
  padding: 8px 14px;
  border-bottom: 1px solid #E4EFEF;
  color: #1A1A1A;
  text-align: right;
}
table.mg-tbl tr:last-child td { border-bottom: none; }
table.mg-tbl tr:hover td      { background: #EFF6F6; }
table.mg-tbl td.pos { color: #1A8A50; font-weight: 500; }
table.mg-tbl td.neg { color: #CC2222; font-weight: 500; }
table.mg-tbl td.gld { color: #9A6800; font-weight: 500; }
table.mg-tbl td.lbl { color: #5F5E56; text-align: left; }
table.mg-tbl th.lbl { text-align: left; }
table.mg-tbl th:first-child,
table.mg-tbl td:first-child { text-align: left; }

/* Note text */
.mg-note {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: #2A4F57;
  font-weight: 500;
  margin-top: 8px;
}

/* Outlier badge */
.mg-badge {
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: #9A6800;
  background: #FFF8E0;
  border: 1px solid #F0D890;
  padding: 2px 6px;
  border-radius: 2px;
}

/* Upload landing */
.mg-land {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
  text-align: center;
}
.mg-land h1 {
  font-family: 'Inter', sans-serif;
  font-size: 48px;
  font-weight: 700;
  letter-spacing: -1px;
  color: #1A1A1A;
  margin-bottom: 8px;
}
.mg-land p {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: #6A6960;
  letter-spacing: 0.5px;
  margin-bottom: 32px;
}
.mg-land .hint {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: #CCCCCC;
  margin-top: 12px;
}

/* Trim badge */
.trim-tag {
  display: inline-block;
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: #FFFFFF;
  background: #006B7A;
  padding: 3px 8px;
  border-radius: 2px;
  vertical-align: middle;
  margin-left: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─── PASSWORD GATE ────────────────────────────────────────────────────────────

def _check_password():
    if st.session_state.get("authenticated"):
        return True

    def _submit():
        entered = st.session_state.get("pw_input", "")
        correct = st.secrets.get("password", "")
        if hmac.compare_digest(entered, correct):
            st.session_state["authenticated"] = True
        else:
            st.session_state["pw_wrong"] = True

    st.markdown("""
    <style>
    /* Paint teal across every Streamlit container variant — selectors changed
       between versions, so we cover the class names AND the data-testids. */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    section.main,
    .main,
    .block-container {
        background: #006B7A !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding-top: 0 !important; padding-bottom: 0 !important; }
    html, body, .stApp { overflow-x: hidden !important; }
    [data-testid="stTextInput"] label p {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
    }
    [data-testid="stTextInput"] input {
        background: #004F5C !important;
        border: 2px solid #FFFFFF !important;
        border-radius: 4px !important;
        color: #FFFFFF !important;
        font-size: 20px !important;
        padding: 14px 18px !important;
    }
    [data-testid="stTextInput"] input::placeholder {
        color: #AADDDD !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("""
        <div style="background:#006B7A; min-height:40vh;
                    display:flex; flex-direction:column;
                    align-items:center; justify-content:center;
                    text-align:center;
                    margin-left:-50vw; margin-right:-50vw;
                    padding-left:50vw; padding-right:50vw;">
          <div style="font-family:'Inter',sans-serif; font-size:64px; font-weight:900;
                      letter-spacing:12px; color:#FFFFFF; text-transform:uppercase;
                      white-space:nowrap; line-height:1;
                      text-shadow:0 2px 24px rgba(0,0,0,0.18);">
            ARGUS
          </div>
          <div style="font-family:'Inter',sans-serif; font-size:18px; font-weight:600;
                      color:#FFFFFF; letter-spacing:7px; text-transform:uppercase;
                      margin-top:22px; opacity:0.92;">
            Performance Analytics
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.text_input("PASSWORD", type="password", key="pw_input",
                      on_change=_submit, label_visibility="visible",
                      placeholder="Enter password")

        if st.session_state.get("pw_wrong"):
            st.markdown("""
            <div style="color:#CC2222; font-family:'Inter',sans-serif;
                        font-size:15px; font-weight:700; text-align:center;
                        margin-top:12px; letter-spacing:1px;">
                Incorrect password — try again
            </div>
            """, unsafe_allow_html=True)
            st.session_state["pw_wrong"] = False

    return False

if not _check_password():
    st.stop()


# ─── SESSION STATE ────────────────────────────────────────────────────────────

for k, v in {
    'fund_df': None, 'fund_name': 'Fund',
    'outliers_df': None, 'trimmed': False,
    'parse_diag': None, 'bm_choice': 'MSCI World Hedged USD',
    'pdf_candidates': None, 'pdf_warns': None, 'pdf_sig': None,
    'pdf_meta': None, 'deck_meta': {},
    'report_pdf': None, 'report_name': None,
    'win_start': None, 'win_end': None, 'excl_extremes': False,
    'hfrx_choice': None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── PDF EXTRACTION HELPERS ───────────────────────────────────────────────────

def _load_pdf_file(up):
    """Extract candidate return series from a PDF, cached by file signature."""
    sig = f"{up.name}:{getattr(up, 'size', 0)}"
    if st.session_state.get('pdf_sig') != sig:
        cands, warns, meta = extract_return_series(up)
        st.session_state['pdf_candidates'] = cands
        st.session_state['pdf_warns'] = warns
        st.session_state['pdf_meta'] = meta
        st.session_state['pdf_sig'] = sig
        st.session_state['pdf_name_default'] = _name_from_file(up.name)


def _clear_pdf_state():
    for k in ('pdf_candidates', 'pdf_warns', 'pdf_sig'):
        st.session_state[k] = None


def _name_from_file(filename):
    """A readable default fund name from an uploaded file name."""
    import os
    base = os.path.splitext(os.path.basename(filename or ''))[0]
    base = base.replace('_', ' ').replace('-', ' ').strip()
    return base or 'Fund'


def _render_pdf_picker():
    """Show return series detected in a PDF; preview and load on confirm."""
    cands = st.session_state.get('pdf_candidates')
    if cands is None:
        return
    for w in (st.session_state.get('pdf_warns') or []):
        st.warning(w)
    if not cands:
        return

    st.markdown('<div class="mg-input-hdr">Returns found in PDF</div>',
                unsafe_allow_html=True)
    labels = [f"{c['label']} — {c['n']} months (page {c['page']})" for c in cands]
    idx = st.selectbox("Detected return series", range(len(cands)),
                       format_func=lambda i: labels[i], key="pdf_pick")
    sel = cands[idx]

    prev = sel['df'].copy()
    prev['ret'] = prev['ret'].map(lambda v: f"{v:+.2f}%")
    prev = prev.rename(columns={'year': 'Year', 'month': 'Month', 'ret': 'Return'})

    cprev, cmeta = st.columns([2, 3])
    with cprev:
        st.dataframe(prev, height=260, hide_index=True, use_container_width=True)
    with cmeta:
        _default_name = sel['label']
        if _default_name == 'Returns':
            _default_name = st.session_state.get('pdf_name_default') or 'Returns'
        name = st.text_input("Fund name", value=_default_name, key="pdf_name")
        st.caption(f"{sel['n']} monthly returns · "
                   f"{int(sel['df'].iloc[0]['year'])}\u2013"
                   f"{int(sel['df'].iloc[-1]['year'])}")
        lc, rc, _ = st.columns([1, 1, 2])
        with lc:
            if st.button("Load into Argus", type="primary", key="pdf_load"):
                df = sel['df'][['year', 'month', 'ret']].copy()
                st.session_state.update({
                    'fund_df': df,
                    'outliers_df': compute_outliers(df),
                    'fund_name': name,
                    'deck_meta': st.session_state.get('pdf_meta') or {}, 'report_pdf': None,
                    'parse_diag': {
                        'date_col': 'PDF deck', 'date_format': 'year \u00d7 month grid',
                        'ret_col': 'PDF deck', 'ret_format': 'percent',
                        'n_parsed': sel['n'], 'n_dropped': 0, 'failed_dates': [],
                    },
                })
                _clear_pdf_state()
                st.rerun()
        with rc:
            if st.button("Cancel", key="pdf_cancel"):
                _clear_pdf_state()
                st.rerun()


# ─── TOP BAR ──────────────────────────────────────────────────────────────────

_tb_fund = st.session_state['fund_name'] if st.session_state['fund_df'] is not None else ''
_tb_fund_html = f'<div class="fund">{_tb_fund}</div>' if _tb_fund else ''
st.markdown(f"""
<div class="mg-topbar">
  <div class="brand">Argus</div>
  {_tb_fund_html}
</div>
""", unsafe_allow_html=True)


# ─── LANDING ──────────────────────────────────────────────────────────────────

if st.session_state['fund_df'] is None:

    # Centered layout: heading + uploader + requirements box, all together
    _, col_up, _ = st.columns([1, 3, 1])
    with col_up:
        st.markdown("""
        <div style="padding: 22px 0 16px; text-align: center;">
          <div style="font-family: 'Inter', sans-serif; font-size: 36px; font-weight: 700; white-space: nowrap;
                      color: #1A1A1A; margin-bottom: 8px;">
            Upload a fund file to begin
          </div>
          <div style="font-family: 'Inter', sans-serif; font-size: 16px;
                      color: #666660; margin-bottom: 18px;">
            Drag and drop or click to select a CSV, Excel, or PDF file
          </div>
        </div>
        """, unsafe_allow_html=True)

        up = st.file_uploader("", type=['csv','xlsx','xls','pdf'], label_visibility="collapsed")

        st.markdown("""
        <div style="margin-top: 18px; border: 1px solid #D2E0E0; border-radius: 6px;
                    background: #FFFFFF; padding: 24px 28px;">
          <div style="font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 700;
                      text-transform: uppercase; letter-spacing: 1.5px; color: #1A1A1A;
                      margin-bottom: 16px; border-bottom: 1px solid #D2E0E0; padding-bottom: 10px;">
            File Requirements
          </div>
          <div style="font-family: 'Inter', sans-serif; font-size: 14px; color: #333330;
                      line-height: 2;">
            <div>1. &nbsp; Format: CSV, Excel, or PDF (.csv, .xlsx, .xls, .pdf)</div>
            <div>2. &nbsp; One column of dates (any standard date format)</div>
            <div>3. &nbsp; One column of monthly returns (% or decimal — auto-detected)</div>
            <div>4. &nbsp; PDF decks: returns are pulled from the track-record table, with a preview to confirm</div>
            <div>5. &nbsp; Minimum 6 months of data required</div>
          </div>
          <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #D2E0E0;
                      font-family: 'Inter', sans-serif; font-size: 13px; color: #5F5E56;">
            Benchmarks (MSCI World Hedged, Bloomberg Global Agg) and TB3MS risk-free rates
            are built in — no need to upload them.
          </div>
        </div>
        """, unsafe_allow_html=True)

    if up is not None:
        if up.name.lower().endswith('.pdf'):
            _load_pdf_file(up)
        else:
            df, err, diag = parse_uploaded_file(up)
            if err:
                st.error(err)
            else:
                st.session_state.update({
                    'fund_df': df,
                    'outliers_df': compute_outliers(df),
                    'parse_diag': diag,
                    'fund_name': _name_from_file(up.name),
                    'deck_meta': {}, 'report_pdf': None,
                })
                st.rerun()

    _render_pdf_picker()
    st.stop()


# ─── CONTROLS BAR ─────────────────────────────────────────────────────────────

st.markdown('<div class="mg-input-hdr">Input</div>', unsafe_allow_html=True)

_hfrx_names = list(HFRX_INDICES.keys())
_hfrx_cur = st.session_state.get('hfrx_choice', _hfrx_names[0])
if _hfrx_cur not in _hfrx_names:
    _hfrx_cur = _hfrx_names[0]

_ALIGN = '<div style="height:24px;"></div>'   # drops buttons to the input baseline
c_up, c_fund, c_hfrx, _pad = st.columns([0.95, 1.7, 1.7, 1.3])

with c_up:
    new_up = st.file_uploader("Upload Fund Data", type=['csv','xlsx','xls','pdf'])
    if new_up is not None:
        if new_up.name.lower().endswith('.pdf'):
            _load_pdf_file(new_up)
        else:
            df2, err2, diag2 = parse_uploaded_file(new_up)
            if err2:
                st.error(err2)
            else:
                st.session_state.update({
                    'fund_df': df2,
                    'outliers_df': compute_outliers(df2),
                    'parse_diag': diag2,
                    'fund_name': _name_from_file(new_up.name),
                    'deck_meta': {}, 'report_pdf': None,
                })
                st.rerun()

with c_fund:
    fn = st.text_input("Fund Name", value=st.session_state['fund_name'])
    st.session_state['fund_name'] = fn

with c_hfrx:
    st.session_state['hfrx_choice'] = st.selectbox(
        "Hedge Fund Index", _hfrx_names,
        index=_hfrx_names.index(_hfrx_cur), key='hfrx_sel')

bm3_name = st.session_state['hfrx_choice']
bm3_df = HFRX_INDICES[bm3_name]

_render_pdf_picker()


# ─── ANALYSIS PERIOD ──────────────────────────────────────────────────────────

_raw_all = st.session_state['fund_df']
_mkeys = [(int(r.year), int(r.month)) for r in _raw_all.sort_values(['year', 'month']).itertuples()]
_mlabels = [f"{MN[m - 1]} {y}" for (y, m) in _mkeys]
if st.session_state.get('win_start') not in _mkeys:
    st.session_state['win_start'] = _mkeys[0]
if st.session_state.get('win_end') not in _mkeys:
    st.session_state['win_end'] = _mkeys[-1]


def _set_window(months=None, ytd=False):
    end = _mkeys[-1]
    if ytd:
        start = next((k for k in _mkeys if k[0] == end[0]), _mkeys[0])
    elif months is None:
        start = _mkeys[0]
    else:
        start = _mkeys[max(0, len(_mkeys) - months)]
    st.session_state['win_start'] = start
    st.session_state['win_end'] = end
    st.session_state['report_pdf'] = None


st.markdown('<div class="mg-input-hdr" style="font-size:15px;padding:0;">Analysis Period</div>',
            unsafe_allow_html=True)
_presets = [('Max', dict()), ('10Y', dict(months=120)), ('5Y', dict(months=60)),
            ('3Y', dict(months=36)), ('1Y', dict(months=12)), ('YTD', dict(ytd=True))]
_r1 = st.columns([1.8, 1.8, 1.6, 1.6])
with _r1[0]:
    _s_lab = st.selectbox("Start", _mlabels, index=_mkeys.index(st.session_state['win_start']))
with _r1[1]:
    _e_lab = st.selectbox("End", _mlabels, index=_mkeys.index(st.session_state['win_end']))
st.session_state['win_start'] = _mkeys[_mlabels.index(_s_lab)]
st.session_state['win_end'] = _mkeys[_mlabels.index(_e_lab)]
with _r1[2]:
    st.markdown(_ALIGN, unsafe_allow_html=True)
    _xx = st.toggle("Exclude best/worst", value=st.session_state['excl_extremes'])
    if _xx != st.session_state['excl_extremes']:
        st.session_state['excl_extremes'] = _xx
        st.session_state['report_pdf'] = None
with _r1[3]:
    st.markdown(_ALIGN, unsafe_allow_html=True)
    st.session_state['trimmed'] = st.toggle("Exclude \u22653\u03c3 outliers",
                                            value=st.session_state['trimmed'])


def _preset_window(kw):
    end = _mkeys[-1]
    if kw.get('ytd'):
        start = next((k for k in _mkeys if k[0] == end[0]), _mkeys[0])
    elif kw.get('months') is None:
        start = _mkeys[0]
    else:
        start = _mkeys[max(0, len(_mkeys) - kw['months'])]
    return (start, end)


_cur_win = (st.session_state['win_start'], st.session_state['win_end'])
_active_preset = next((lbl for lbl, kw in _presets if _preset_window(kw) == _cur_win), None)
_r2 = st.columns(6)
for _col, (_lbl, _kw) in zip(_r2, _presets):
    with _col:
        if st.button(_lbl, key=f"preset_{_lbl}", use_container_width=True,
                     type=("primary" if _lbl == _active_preset else "secondary")):
            _set_window(**_kw)
            st.rerun()


# ─── RESOLVE DATA ─────────────────────────────────────────────────────────────

_win_s, _win_e = st.session_state['win_start'], st.session_state['win_end']
if _win_s > _win_e:
    _win_s, _win_e = _win_e, _win_s
_full_n = len(_raw_all)
_windowed = _raw_all[_raw_all.apply(
    lambda r: _win_s <= (int(r['year']), int(r['month'])) <= _win_e, axis=1)].copy()
if st.session_state['excl_extremes'] and len(_windowed) > 2:
    _windowed = _windowed.drop([_windowed['ret'].idxmax(), _windowed['ret'].idxmin()])
fund_df_raw = _windowed.sort_values(['year', 'month']).reset_index(drop=True)

outliers_df = st.session_state['outliers_df']
diag        = st.session_state.get('parse_diag') or {}
fund_name   = st.session_state['fund_name']
trimmed     = st.session_state['trimmed']
bm_choice   = st.session_state['bm_choice']
_win_active = (len(fund_df_raw) != _full_n)
_scope_sig = (_win_s, _win_e, st.session_state['excl_extremes'], trimmed,
              st.session_state['fund_name'], st.session_state.get('hfrx_choice'))
if st.session_state.get('win_sig') != _scope_sig:
    _prev_sig = st.session_state.get('win_sig')
    st.session_state['win_sig'] = _scope_sig
    st.session_state['report_pdf'] = None
    if _prev_sig is not None:
        _tlabel = _active_preset or 'Custom'
        st.toast(f"{_tlabel} \u00b7 {MN[_win_s[1]-1]} {_win_s[0]} \u2013 "
                 f"{MN[_win_e[1]-1]} {_win_e[0]} \u00b7 {len(fund_df_raw)} months")

if bm_choice == 'MSCI World Hedged USD':
    bm1_df, bm1_name = MSCI_DF, 'MSCI World Hdg'
    bm2_df, bm2_name = AGG_DF,  'Blmbrg Agg'
elif bm_choice == 'Bloomberg Global Agg':
    bm1_df, bm1_name = AGG_DF,  'Bloomberg Agg'
    bm2_df, bm2_name = MSCI_DF, 'MSCI World Hdg'
else:
    bm1_df = bm2_df = bm1_name = bm2_name = None

if trimmed and outliers_df is not None and len(outliers_df) > 0:
    out_set = set(zip(outliers_df['year'], outliers_df['month']))
    fund_df = fund_df_raw[
        ~fund_df_raw.apply(lambda r: (r['year'], r['month']) in out_set, axis=1)
    ].copy()
else:
    fund_df = fund_df_raw

_MIN_MONTHS = 12
if len(fund_df) < _MIN_MONTHS:
    st.warning(
        f"The selected analysis window leaves **{len(fund_df)}** month"
        f"{'s' if len(fund_df) != 1 else ''} of data. Argus needs at least "
        f"**{_MIN_MONTHS} months** to compute reliable analytics. "
        f"Widen the date range, or turn off the exclusions, in the Analysis Period controls above.")
    st.stop()

_band_parts = [f"{MN[_win_s[1]-1]} {_win_s[0]} \u2013 {MN[_win_e[1]-1]} {_win_e[0]}",
               f"{len(fund_df)} months"]
if _active_preset:
    _band_parts.insert(0, _active_preset)
if st.session_state['excl_extremes']:
    _band_parts.append("best/worst excluded")
if trimmed:
    _band_parts.append("\u22653\u03c3 excluded")
st.markdown(
    f'<div style="border:1.5px solid #006B7A;background:#EAF3F4;color:#004A54;'
    f'font-family:Inter,sans-serif;font-weight:700;font-size:13px;letter-spacing:.2px;'
    f'padding:7px 14px;border-radius:6px;margin:8px 0 4px;display:inline-block;">'
    f'Showing&nbsp; {"&nbsp;&middot;&nbsp; ".join(_band_parts)}</div>',
    unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _cls(v, flip=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    return ('neg' if flip else 'pos') if v > 0 else ('pos' if flip else 'neg')

def _fmt(v, dec=2, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f"{v:.{dec}f}{'%' if pct else ''}"


# ─── FUND IDENTITY ────────────────────────────────────────────────────────────

rets  = fund_df['ret'].values
n_obs = len(rets)
n_out = len(outliers_df) if outliers_df is not None else 0
first, last = fund_df.iloc[0], fund_df.iloc[-1]
date_range = (
    f"{MN[int(first['month'])-1]} {int(first['year'])} "
    f"– {MN[int(last['month'])-1]} {int(last['year'])}"
)
trim_html = '<span class="trim-tag">Trimmed</span>' if trimmed else ''
out_html  = (f'&nbsp;·&nbsp; <span style="color:#9A6800;font-weight:600">'
             f'{n_out} outlier{"s" if n_out!=1 else ""} detected</span>'
             if n_out > 0 else '')



# ─── STAT BANNER ──────────────────────────────────────────────────────────────

ar = geo_return(rets)
av = ann_vol(rets)
sh = sharpe_with_rf(fund_df)
so = sortino_with_rf(fund_df)
md = max_drawdown(rets)
ca = calmar(rets)

st.markdown(f"""
<div class="mg-stats">
  <div class="mg-stat">
    <div class="s-lbl">Ann. Return</div>
    <div class="s-val {_cls(ar)}">{_fmt(ar)}</div>
  </div>
  <div class="mg-stat">
    <div class="s-lbl">Ann. Volatility</div>
    <div class="s-val dim">{_fmt(av)}</div>
  </div>
  <div class="mg-stat">
    <div class="s-lbl">Sharpe · Rf=TB3MS</div>
    <div class="s-val dim">{_fmt(sh, 2, False)}</div>
  </div>
  <div class="mg-stat">
    <div class="s-lbl">Sortino · Rf=TB3MS</div>
    <div class="s-val dim">{_fmt(so, 2, False)}</div>
  </div>
  <div class="mg-stat">
    <div class="s-lbl">Max Drawdown</div>
    <div class="s-val neg">{_fmt(md)}</div>
  </div>
  <div class="mg-stat">
    <div class="s-lbl">Calmar Ratio</div>
    <div class="s-val dim">{_fmt(ca, 2, False)}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── TABS ─────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Exec Summary", "Summary", "Calendar", "Drawdowns", "Distribution",
    "Regression", "Significance", "Tail Risk", "De-Smoothing",
    "Co-Movement", "Waterfall", "Rolling", "Seasonality", "Multi-Period",
    "Macro Events", "Shocks", "DDQ", "Data", "Input",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — INPUT
# ══════════════════════════════════════════════════════════════════════════════

with tabs[18]:
    st.markdown('<div class="mg-sh">Input &amp; Parsing Diagnostics</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="mg-id">
  <div class="fname">{fund_name}{trim_html}</div>
  <div class="fmeta">
    {date_range} &nbsp;·&nbsp; {n_obs} monthly observations
    &nbsp;·&nbsp; {bm1_name or 'No benchmark'}
    &nbsp;·&nbsp; Rf = TB3MS (FRED){out_html}
  </div>
</div>
""", unsafe_allow_html=True)

    # Diagnostic strip
    if diag:
        failed = diag.get('failed_dates', [])
        drop_color = '#CC2222' if diag.get('n_dropped', 0) > 0 else '#1A1A1A'
        _rawfmt = diag.get('date_format', '') or ''
        fmt_disp = _rawfmt
        for _k, _v in (('%Y','YYYY'),('%y','YY'),('%m','MM'),('%b','Mon'),('%B','Month'),('%d','DD')):
            fmt_disp = fmt_disp.replace(_k, _v)
        fmt_disp = fmt_disp or '—'
        st.markdown(f"""
<div class="mg-diag">
  <div class="d-item"><div class="d-lbl">Date Column</div><div class="d-val">{diag.get('date_col','—')}</div></div>
  <div class="d-item"><div class="d-lbl">Format</div><div class="d-val">{fmt_disp}</div></div>
  <div class="d-item"><div class="d-lbl">Return Column</div><div class="d-val">{diag.get('ret_col','—')}</div></div>
  <div class="d-item"><div class="d-lbl">Return Scale</div><div class="d-val">{diag.get('ret_format','—')}</div></div>
  <div class="d-item"><div class="d-lbl">Rows Parsed</div><div class="d-val">{diag.get('n_parsed','—')}</div></div>
  <div class="d-item"><div class="d-lbl">Rows Dropped</div><div class="d-val" style="color:{drop_color}">{diag.get('n_dropped',0)}</div></div>
  {f'<div class="d-item"><div class="d-lbl">Parse Warnings</div><div class="d-val" style="color:#CC2222">{len(failed)} failed dates</div></div>' if failed else ''}
</div>
""", unsafe_allow_html=True)

    # Outlier alert
    if n_out > 0 and outliers_df is not None:
        out_list = ' · '.join(
            f"{MN[int(r.month)-1]} {int(r.year)} ({r.ret:+.2f}%)"
            for r in outliers_df.itertuples()
        )
        st.markdown(
            f'<div class="mg-alert">⚡ <strong>Outlier Alert</strong> &nbsp;—&nbsp; {out_list}'
            f'&nbsp; · &nbsp; Toggle "Exclude ≥3σ Outliers" above to recompute without these months.</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — EXEC SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    _c_pdf, _c_dl, _ = st.columns([1.1, 1.1, 4])
    with _c_pdf:
        if st.button("Generate PDF", type="primary"):
            with st.spinner("Building report\u2026"):
                try:
                    _safe = _re.sub(r'[^A-Za-z0-9]+', '_', str(fund_name)).strip('_') or 'Report'
                    _ym = f"{int(fund_df.iloc[-1]['year'])}-{int(fund_df.iloc[-1]['month']):02d}"
                    st.session_state['report_pdf'] = build_report(
                        fund_df, fund_name, MSCI_DF, 'MSCI World Hedged',
                        AGG_DF, bm3_df, bm3_name, meta=st.session_state.get('deck_meta'))
                    st.session_state['report_name'] = f"Argus_{_safe}_{_ym}.pdf"
                except Exception as _e:
                    st.session_state['report_pdf'] = None
                    st.error(f"Report failed: {_e}")
    with _c_dl:
        if st.session_state.get('report_pdf'):
            st.download_button("Download PDF", st.session_state['report_pdf'],
                               file_name=st.session_state.get('report_name', 'Argus_Report.pdf'),
                               mime="application/pdf")
    for _head, _body in build_exec_summary(fund_df, fund_name, MSCI_DF, 'MSCI World Hdg',
                                           AGG_DF, bm3_df, bm3_name,
                                           meta=st.session_state.get('deck_meta')):
        st.markdown(f'<div class="mg-sh" style="margin-top:18px;">{_head}</div>', unsafe_allow_html=True)
        st.markdown(_body, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    col_l, col_r = st.columns([1, 1], gap="large")

    full_rets = fund_df_raw['ret'].values
    trim_rets = rets
    jb_full, _ = jarque_bera(full_rets)
    jb_trim, _ = jarque_bera(trim_rets)

    with col_l:
        st.markdown('<div class="mg-sh">Full vs. Ex-Outliers Statistics</div>', unsafe_allow_html=True)

        rows = [
            ("Observations",       len(full_rets),                   len(trim_rets),                  False, 0,  False),
            ("Ann. Return",        geo_return(full_rets),            geo_return(trim_rets),            True,  2,  False),
            ("Ann. Volatility",    ann_vol(full_rets),              ann_vol(trim_rets),              False, 2,  False),
            ("Sharpe (Rf=TB3MS)",  sharpe_with_rf(fund_df_raw),     sharpe_with_rf(fund_df),         False, 2,  False),
            ("Sortino (Rf=TB3MS)", sortino_with_rf(fund_df_raw),    sortino_with_rf(fund_df),        False, 2,  False),
            ("Calmar Ratio",       calmar(full_rets),               calmar(trim_rets),               False, 2,  False),
            ("Max Drawdown",       max_drawdown(full_rets),         max_drawdown(trim_rets),         True,  2,  True),
            ("Skewness",           skewness(full_rets),             skewness(trim_rets),             False, 4,  False),
            ("Excess Kurtosis",    excess_kurtosis(full_rets),      excess_kurtosis(trim_rets),      False, 4,  False),
            ("Jarque-Bera Stat",   jb_full,                         jb_trim,                         False, 2,  False),
            ("Best Month",         float(np.max(full_rets)),        float(np.max(trim_rets)),        True,  2,  False),
            ("Worst Month",        float(np.min(full_rets)),        float(np.min(trim_rets)),        True,  2,  True),
            ("Avg Monthly Return", float(np.mean(full_rets)),       float(np.mean(trim_rets)),       True,  2,  False),
            ("Hit Rate",           float(np.sum(full_rets>0)/len(full_rets)*100),
                                   float(np.sum(trim_rets>0)/max(len(trim_rets),1)*100), False, 1, False),
            ("Up Months",          int(np.sum(full_rets > 0)),      int(np.sum(trim_rets > 0)),      False, 0,  False),
            ("Down Months",        int(np.sum(full_rets < 0)),      int(np.sum(trim_rets < 0)),      False, 0,  False),
        ]

        html = f"""<table class="mg-tbl">
<thead><tr>
  <th class="lbl">Metric</th>
  <th>Full (N={len(full_rets)})</th>
  <th>Ex-Outliers (N={len(trim_rets)})</th>
</tr></thead><tbody>"""
        for (lbl, fv, tv, pct, dec, flip) in rows:
            def _cell(v):
                if isinstance(v, int) and dec == 0:
                    return f'<td>{v}</td>'
                try:
                    c = _cls(float(v), flip)
                    return f'<td class="{c}">{_fmt(float(v), dec, pct)}</td>'
                except Exception:
                    return '<td>—</td>'
            html += f'<tr><td class="lbl">{lbl}</td>{_cell(fv)}{_cell(tv)}</tr>'
        html += '</tbody></table>'
        st.markdown(html, unsafe_allow_html=True)

    with col_r:
        if bm1_df is not None:
            st.markdown('<div class="mg-sh">Benchmark Comparison</div>', unsafe_allow_html=True)

            def _bm_stats(df):
                r = df['ret'].values
                return {
                    'Ann. Return':     geo_return(r),
                    'Ann. Volatility': ann_vol(r),
                    'Sharpe':          sharpe_with_rf(df),
                    'Sortino':         sortino_with_rf(df),
                    'Max Drawdown':    max_drawdown(r),
                    'Calmar Ratio':    calmar(r),
                    'Skewness':        skewness(r),
                    'Excess Kurtosis': excess_kurtosis(r),
                }

            bm1_al = fund_df_raw.merge(bm1_df, on=['year','month'], suffixes=('','_b'))[['year','month','ret_b']]\
                                 .rename(columns={'ret_b':'ret'})
            bm2_al = None
            if bm2_df is not None:
                bm2_al = fund_df_raw.merge(bm2_df, on=['year','month'], suffixes=('','_b'))[['year','month','ret_b']]\
                                     .rename(columns={'ret_b':'ret'})
            bm3_al = fund_df_raw.merge(bm3_df, on=['year','month'], suffixes=('','_b'))[['year','month','ret_b']]\
                                 .rename(columns={'ret_b':'ret'})

            fs = _bm_stats(fund_df_raw)
            b1 = _bm_stats(bm1_al)
            b2 = _bm_stats(bm2_al) if bm2_al is not None else {}
            b3 = _bm_stats(bm3_al) if len(bm3_al) > 0 else {}

            bm_defs = [
                ('Ann. Return',     True,  2, False),
                ('Ann. Volatility', False, 2, False),
                ('Sharpe',          False, 2, False),
                ('Sortino',         False, 2, False),
                ('Max Drawdown',    True,  2, True),
                ('Calmar Ratio',    False, 2, False),
                ('Skewness',        False, 4, False),
                ('Excess Kurtosis', False, 4, False),
            ]
            b2h = f'<th>{bm2_name}</th>' if bm2_al is not None else ''
            b3h = f'<th>{bm3_name}</th>' if b3 else ''
            html2 = f"""<table class="mg-tbl">
<thead><tr><th class="lbl">Metric</th><th>{fund_name}</th><th>{bm1_name}</th>{b2h}{b3h}</tr></thead><tbody>"""
            for (m, pct, dec, flip) in bm_defs:
                html2 += f'<tr><td class="lbl">{m}</td>'
                for st_d in ([fs, b1] + ([b2] if bm2_al is not None else []) + ([b3] if b3 else [])):
                    v = st_d.get(m, np.nan)
                    try:
                        c = _cls(float(v), flip)
                        html2 += f'<td class="{c}">{_fmt(float(v), dec, pct)}</td>'
                    except Exception:
                        html2 += '<td>—</td>'
                html2 += '</tr>'
            html2 += '</tbody></table>'
            st.markdown(html2, unsafe_allow_html=True)

        st.markdown('<div class="mg-sh" style="margin-top:28px;">Cumulative Wealth — Growth of $100</div>', unsafe_allow_html=True)
        st.plotly_chart(
            chart_cumulative(fund_df, fund_name, bm1_df, bm1_name or '', bm2_df, bm2_name or '', bm3_df, bm3_name),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CALENDAR
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown('<div class="mg-sh">Monthly Return Track Record</div>', unsafe_allow_html=True)

    _MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    def _ret_lut(df):
        return {(int(r.year), int(r.month)): float(r.ret) for r in df.itertuples()}
    def _ytd(vals):
        present = [v for v in vals if v is not None]
        if not present:
            return None
        w = 1.0
        for v in present:
            w *= (1 + v / 100)
        return (w - 1) * 100
    def _heat(v, cap):
        if v is None:
            return 'background:#FFFFFF;'
        t = max(-1.0, min(1.0, v / cap))
        if t >= 0:
            return f'background:rgba(26,138,80,{0.10 + 0.50 * t:.2f});'
        return f'background:rgba(204,34,34,{0.10 + 0.50 * (-t):.2f});'

    _cal_series = [(fund_name, fund_df), ('MSCI World Hdg', MSCI_DF), ('Bloomberg Agg', AGG_DF), (bm3_name, bm3_df)]
    _luts = [(nm, _ret_lut(d)) for nm, d in _cal_series]
    _years = sorted({int(r.year) for r in fund_df.itertuples()})

    _hd = 'background:#006B7A;color:#FFFFFF;padding:6px 6px;text-align:right;font-weight:600;'
    th = (f'<th style="{_hd}text-align:left;"></th><th style="{_hd}text-align:left;"></th>')
    for mn in _MONTHS:
        th += f'<th style="{_hd}">{mn}</th>'
    th += f'<th style="{_hd}">YTD</th>'

    rows_html = ''
    for yi, yr in enumerate(_years):
        for si, (sname, lut) in enumerate(_luts):
            tb = 'border-top:2px solid #006B7A;' if (si == 0 and yi > 0) else ''
            yr_cell = str(yr) if si == 0 else ''
            row = f'<tr><td style="text-align:left;font-weight:700;color:#1A1A1A;padding:3px 8px;{tb}">{yr_cell}</td>'
            row += f'<td style="text-align:left;color:#1A1A1A;padding:3px 8px;{tb}">{sname}</td>'
            mvals = []
            for mo in range(1, 13):
                v = lut.get((yr, mo))
                mvals.append(v)
                txt = f'{v:+.2f}%' if v is not None else ''
                row += f'<td style="text-align:right;padding:3px 5px;color:#1A1A1A;{_heat(v, 6.0)}{tb}">{txt}</td>'
            ytd = _ytd(mvals)
            ytd_txt = f'{ytd:+.2f}%' if ytd is not None else ''
            row += f'<td style="text-align:right;padding:3px 6px;font-weight:700;color:#1A1A1A;{_heat(ytd, 20.0)}{tb}">{ytd_txt}</td>'
            row += '</tr>'
            rows_html += row

    cal_html = ('<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;'
                "font-family:'JetBrains Mono',monospace;font-size:11px;\">"
                f'<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody></table></div>')
    st.markdown(cal_html, unsafe_allow_html=True)
    st.markdown('<div class="mg-sh" style="margin-top:8px;">Monthly Return Bars</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_monthly_bars(fund_df), use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DRAWDOWNS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown('<div class="mg-sh">Drawdown Series</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_drawdowns(fund_df), use_container_width=True, config={'displayModeBar': False})

    episodes = top_drawdowns(fund_df)

    # ── Drawdown Analysis: Largest / Longest / Mean / Median summary ──
    st.markdown('<div class="mg-sh" style="margin-top:8px;">Drawdown Analysis</div>', unsafe_allow_html=True)
    if episodes:
        import datetime as _dt
        _MN = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        def _pmd(s):
            if not s or s in ('Start', 'Ongoing'):
                return None
            try:
                mo, yr = s.split()
                return _dt.date(int(yr), _MN.index(mo) + 1, 1)
            except Exception:
                return None
        def _avg_date(strs, med=False):
            ords = [d.toordinal() for d in (_pmd(s) for s in strs) if d]
            if not ords:
                return '—'
            v = np.median(ords) if med else np.mean(ords)
            d = _dt.date.fromordinal(int(round(v)))
            return f"{_MN[d.month - 1]} {d.year}"
        def _dur(e, key):
            v = e[key]
            return str(v) if v is not None else 'Ongoing'

        largest = episodes[0]
        longest = max(episodes, key=lambda e: e['total_months'] if e['total_months'] is not None else e['peak_to_trough'])
        depths = [abs(e['drawdown']) for e in episodes]
        p2t = [e['peak_to_trough'] for e in episodes]
        t2r = [e['trough_to_recovery'] for e in episodes if e['trough_to_recovery'] is not None]
        p2r = [e['total_months'] for e in episodes if e['total_months'] is not None]

        da_rows = [
            ('Drawdown %',             f"{abs(largest['drawdown']):.2f}", f"{abs(longest['drawdown']):.2f}", f"{np.mean(depths):.2f}", f"{np.median(depths):.2f}"),
            ('Peak',                   largest['peak_date'],     longest['peak_date'],     _avg_date([e['peak_date'] for e in episodes]),     _avg_date([e['peak_date'] for e in episodes], med=True)),
            ('Trough',                 largest['trough_date'],   longest['trough_date'],   _avg_date([e['trough_date'] for e in episodes]),   _avg_date([e['trough_date'] for e in episodes], med=True)),
            ('Recovery',               largest['recovery_date'], longest['recovery_date'], _avg_date([e['recovery_date'] for e in episodes]), _avg_date([e['recovery_date'] for e in episodes], med=True)),
            ('Peak to Trough (mo)',    _dur(largest, 'peak_to_trough'),     _dur(longest, 'peak_to_trough'),     f"{np.mean(p2t):.2f}",                f"{np.median(p2t):.2f}"),
            ('Trough to Recovery (mo)', _dur(largest, 'trough_to_recovery'), _dur(longest, 'trough_to_recovery'), (f"{np.mean(t2r):.2f}" if t2r else '—'), (f"{np.median(t2r):.2f}" if t2r else '—')),
            ('Peak to Recovery (mo)',  _dur(largest, 'total_months'),       _dur(longest, 'total_months'),       (f"{np.mean(p2r):.2f}" if p2r else '—'), (f"{np.median(p2r):.2f}" if p2r else '—')),
        ]
        da_html = ('<div style="max-width:640px;"><table class="mg-tbl"><thead><tr>'
                   '<th></th><th>Largest</th><th>Longest</th><th>Mean</th><th>Median</th>'
                   '</tr></thead><tbody>')
        for lbl, a, b, c, d in da_rows:
            da_html += f'<tr><td class="lbl">{lbl}</td><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>'
        da_html += '</tbody></table></div>'
        st.markdown(da_html, unsafe_allow_html=True)

    st.markdown('<div class="mg-sh" style="margin-top:8px;">Top Drawdown Episodes</div>', unsafe_allow_html=True)
    ep_html = '<div style="max-width:880px;"><table class="mg-tbl"><thead><tr><th class="lbl">#</th><th>Drawdown</th><th class="lbl">Peak</th><th class="lbl">Trough</th><th class="lbl">Recovery</th><th>Peak→Trough</th><th>Trough→Rec.</th><th>Total</th></tr></thead><tbody>'
    for i, ep in enumerate(episodes):
        pt  = str(ep['peak_to_trough']) + 'm'
        tr  = str(ep['trough_to_recovery']) + 'm' if ep['trough_to_recovery'] is not None else 'Ongoing'
        tot = str(ep['total_months']) + 'm'        if ep['total_months'] is not None else 'Ongoing'
        ep_html += (f"<tr><td class='lbl'>{i+1}</td><td class='neg'>{_fmt(ep['drawdown'])}</td>"
                    f"<td class='lbl'>{ep['peak_date']}</td><td class='lbl'>{ep['trough_date']}</td>"
                    f"<td class='lbl'>{ep['recovery_date']}</td><td>{pt}</td>"
                    f"<td>{tr}</td><td>{tot}</td></tr>")
    ep_html += '</tbody></table></div>'
    st.markdown(ep_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    c_h, c_q = st.columns([3, 2], gap="large")
    with c_h:
        st.markdown('<div class="mg-sh">Return Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_histogram(fund_df, outliers_df if not trimmed else None), use_container_width=True, config={'displayModeBar': False})
    with c_q:
        st.markdown('<div class="mg-sh">Q-Q Plot vs Normal</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_qq(fund_df), use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div class="mg-sh" style="margin-top:8px;">Normality Tests</div>', unsafe_allow_html=True)
    norm = normality_tests(rets)
    n_html = '<table class="mg-tbl"><thead><tr><th class="lbl">Test</th><th>Statistic</th><th>p-value / Critical</th><th>Reject Normality?</th></tr></thead><tbody>'
    jb  = norm['jb'];  jbr = jb['pval'] < 0.05
    n_html += f"<tr><td class='lbl'>Jarque-Bera</td><td>{jb['stat']:.4f}</td><td>{jb['pval']:.4f}</td><td class='{'neg' if jbr else 'pos'}'>{'Yes ✗' if jbr else 'No ✓'}</td></tr>"
    if 'sw' in norm:
        sw = norm['sw']; swr = sw['pval'] < 0.05
        n_html += f"<tr><td class='lbl'>Shapiro-Wilk</td><td>{sw['stat']:.4f}</td><td>{sw['pval']:.4f}</td><td class='{'neg' if swr else 'pos'}'>{'Yes ✗' if swr else 'No ✓'}</td></tr>"
    ad  = norm['ad']
    n_html += f"<tr><td class='lbl'>Anderson-Darling</td><td>{ad['stat']:.4f}</td><td>Crit(5%) = {ad['critical']:.4f}</td><td class='{'neg' if ad['reject'] else 'pos'}'>{'Yes ✗' if ad['reject'] else 'No ✓'}</td></tr>"
    ks  = norm['ks'];  ksr = ks['pval'] < 0.05
    n_html += f"<tr><td class='lbl'>Kolmogorov-Smirnov</td><td>{ks['stat']:.4f}</td><td>{ks['pval']:.4f}</td><td class='{'neg' if ksr else 'pos'}'>{'Yes ✗' if ksr else 'No ✓'}</td></tr>"
    n_html += '</tbody></table>'
    st.markdown(n_html, unsafe_allow_html=True)

    st.markdown('<div class="mg-sh" style="margin-top:28px;">Return Autocorrelation — Lags 1–12</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_acf(fund_df), use_container_width=True, config={'displayModeBar': False})
    cb = acf_conf_band(len(rets))
    st.markdown(f'<div class="mg-note">95% confidence band: ±{cb:.4f} &nbsp;·&nbsp; Highlighted bars are statistically significant</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    from scipy.stats import t as _t

    def _reg_pval(t_stat, nn):
        if t_stat is None or np.isnan(t_stat):
            return np.nan
        return float(2 * _t.sf(abs(t_stat), df=nn))

    def _render_regression(reg, bm_label):
        if reg is None:
            st.warning(f"Insufficient overlapping data for regression vs {bm_label}.")
            return
        c_rl, c_rr = st.columns([3, 2], gap="large")
        with c_rl:
            st.markdown(f'<div class="mg-sh">{fund_name} vs {bm_label}</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_regression(reg, fund_name, bm_label), use_container_width=True, config={'displayModeBar': False})
        with c_rr:
            st.markdown('<div class="mg-sh">Regression Statistics</div>', unsafe_allow_html=True)
            n_reg = reg['n']
            ap = _reg_pval(reg['t_alpha'], n_reg - 2)
            bp = _reg_pval(reg['t_beta'], n_reg - 2)
            reg_rows = [
                ("N (overlapping)",      f"{n_reg}"),
                ("Alpha (monthly %)",    f"{reg['alpha']:.4f}%"),
                ("  SE(α)",          f"{reg['se_alpha']:.4f}" if not np.isnan(reg['se_alpha']) else '—'),
                ("  t(α)",           f"{reg['t_alpha']:.2f}"  if not np.isnan(reg['t_alpha'])  else '—'),
                ("  p(α)",           f"{ap:.4f}"              if not np.isnan(ap)              else '—'),
                ("  95% CI",             f"[{reg['alpha_ci'][0]:.2f}, {reg['alpha_ci'][1]:.2f}]"  if not np.isnan(reg['alpha_ci'][0]) else '—'),
                ("Beta (full period)",   f"{reg['beta']:.4f}"),
                ("  SE(β)",          f"{reg['se_beta']:.4f}"  if not np.isnan(reg['se_beta']) else '—'),
                ("  t(β)",           f"{reg['t_beta']:.2f}"   if not np.isnan(reg['t_beta'])  else '—'),
                ("  p(β)",           f"{bp:.4f}"              if not np.isnan(bp)             else '—'),
                ("  95% CI",             f"[{reg['beta_ci'][0]:.2f}, {reg['beta_ci'][1]:.2f}]"   if not np.isnan(reg['beta_ci'][0]) else '—'),
                ("R²",               f"{reg['r2']:.4f}"),
                ("Correlation",          f"{reg['corr']:.4f}"),
                ("β⁺ (BM up months)",    f"{reg['beta_up']:.4f}"   if reg['beta_up']   is not None else '—'),
                ("  N up",               f"{reg['n_up']}"),
                ("β⁻ (BM down months)",  f"{reg['beta_dn']:.4f}"   if reg['beta_dn']   is not None else '—'),
                ("  N down",             f"{reg['n_dn']}"),
                ("Convexity (β⁺ − β⁻)", f"{reg['convexity']:.4f}" if reg['convexity'] is not None else '—'),
            ]
            r_html = '<table class="mg-tbl"><thead><tr><th class="lbl">Statistic</th><th>Value</th></tr></thead><tbody>'
            for lbl, val in reg_rows:
                s = 'color:#1A1A1A;padding-left:20px;' if lbl.startswith('  ') else ''
                r_html += f'<tr><td class="lbl" style="{s}">{lbl.strip()}</td><td>{val}</td></tr>'
            r_html += '</tbody></table>'
            st.markdown(r_html, unsafe_allow_html=True)

    _render_regression(piecewise_beta_regression(fund_df, MSCI_DF), 'MSCI World Hdg')
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    _render_regression(piecewise_beta_regression(fund_df, AGG_DF), 'Bloomberg Agg')
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    _render_regression(piecewise_beta_regression(fund_df, bm3_df), bm3_name)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown('<div class="mg-sh" style="margin-top:14px;">Statistical Significance</div>', unsafe_allow_html=True)
    _sig = sharpe_significance(fund_df)
    _reg = piecewise_beta_regression(fund_df, MSCI_DF)
    if _sig.get('n', 0) < 3:
        st.markdown('<div class="mg-note">Not enough observations for inference.</div>', unsafe_allow_html=True)
    else:
        _ci = _sig['ci']
        _rows = [
            ("Observations (months)", f"{_sig['n']}"),
            ("Annualized Sharpe", f"{_sig['sharpe']:.4f}"),
            ("Sharpe std. error (Lo)", f"{_sig['se']:.4f}"),
            ("Sharpe 95% CI", f"{_ci[0]:.4f} to {_ci[1]:.4f}"),
            ("Sharpe t-stat (H\u2080: SR=0)", f"{_sig['sharpe_t']:.4f}"),
            ("Mean-return t-stat", f"{_sig['ret_t']:.4f}"),
            ("Mean-return p-value", f"{_sig['ret_p']:.4f}"),
        ]
        if _reg:
            _ac = _reg['alpha_ci']
            _rows += [
                ("Monthly alpha vs MSCI World Hdg", f"{_reg['alpha']:.4f}%"),
                ("Alpha t-stat", f"{_reg['t_alpha']:.4f}"),
                ("Alpha 95% CI", f"{_ac[0]:.4f}% to {_ac[1]:.4f}%"),
            ]
        _html = '<table style="width:100%;border-collapse:collapse;font-size:15px;color:#1A1A1A;">'
        for k, v in _rows:
            _html += (f'<tr><td style="padding:7px 10px;border-bottom:1px solid #E5ECEC;">{k}</td>'
                      f'<td style="padding:7px 10px;border-bottom:1px solid #E5ECEC;text-align:right;'
                      f'font-family:{"JetBrains Mono, monospace"};font-weight:600;">{v}</td></tr>')
        _html += '</table>'
        st.markdown(_html, unsafe_allow_html=True)
        _sig_note = ("Significant at 5% (CI excludes 0)." if _ci[0] > 0
                     else "Not significant at 5% \u2014 the 95% interval includes a Sharpe of 0.")
        st.markdown(f'<div style="color:#1A1A1A;font-size:15px;margin-top:12px;">{_sig_note} '
                    f'Standard error uses Lo (2002) under an i.i.d. assumption; serial correlation would widen it.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — TAIL RISK
# ══════════════════════════════════════════════════════════════════════════════

with tabs[7]:
    st.markdown('<div class="mg-sh" style="margin-top:14px;">Tail Risk &amp; Ratios</div>', unsafe_allow_html=True)
    _tr = tail_risk(fund_df['ret'].values)
    _cap = capture_ratios(fund_df, MSCI_DF)
    if _tr.get('n', 0) < 4:
        st.markdown('<div class="mg-note">Not enough observations.</div>', unsafe_allow_html=True)
    else:
        def _tbl(title, rows):
            h = f'<div class="mg-sh" style="margin-top:16px;">{title}</div>'
            h += '<table style="width:100%;border-collapse:collapse;font-size:15px;color:#1A1A1A;">'
            for k, v in rows:
                h += (f'<tr><td style="padding:7px 10px;border-bottom:1px solid #E5ECEC;">{k}</td>'
                      f'<td style="padding:7px 10px;border-bottom:1px solid #E5ECEC;text-align:right;'
                      f'font-family:JetBrains Mono, monospace;font-weight:600;">{v}</td></tr>')
            return h + '</table>'
        st.markdown(_tbl("Monthly Value-at-Risk", [
            ("Historical VaR 95%", f"{_tr['var95']:.2f}%"),
            ("Historical VaR 99%", f"{_tr['var99']:.2f}%"),
            ("Modified VaR 95% (Cornish-Fisher)", f"{_tr['mvar95']:.2f}%"),
            ("Modified VaR 99% (Cornish-Fisher)", f"{_tr['mvar99']:.2f}%"),
            ("Expected shortfall (CVaR) 95%", f"{_tr['cvar95']:.2f}%"),
            ("Expected shortfall (CVaR) 99%", f"{_tr['cvar99']:.2f}%"),
        ]), unsafe_allow_html=True)
        _ratio_rows = [
            ("Ulcer index", f"{_tr['ulcer']:.2f}"),
            ("Tail ratio (P95 / |P5|)", f"{_tr['tail_ratio']:.2f}"),
            ("Gain-to-pain", f"{_tr['gain_to_pain']:.2f}"),
            ("Omega (threshold 0)", f"{_tr['omega']:.2f}"),
        ]
        if _cap.get('up_capture') is not None:
            _ratio_rows.append(("Up-market capture vs MSCI World Hdg", f"{_cap['up_capture']:.2f}%"))
        if _cap.get('down_capture') is not None:
            _ratio_rows.append(("Down-market capture vs MSCI World Hdg", f"{_cap['down_capture']:.2f}%"))
        st.markdown(_tbl("Ratios &amp; Capture", _ratio_rows), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — DE-SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

with tabs[8]:
    st.markdown('<div class="mg-sh" style="margin-top:14px;">Return De-Smoothing</div>', unsafe_allow_html=True)
    _ds = desmooth_stats(fund_df, MSCI_DF)
    if _ds.get('n', 0) < 6:
        st.markdown('<div class="mg-note">Not enough observations.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color:#1A1A1A;font-size:15px;margin-bottom:10px;">'
                    f'First-order autocorrelation (\u03c1) = <b>{_ds["rho"]:.4f}</b>. The unsmoothed series removes that '
                    f'serial dependence (Geltner/Okunev-White); compare the risk figures below.</div>', unsafe_allow_html=True)
        _drows = [
            ("Annualized volatility", f"{_ds['vol_rep']:.2f}%", f"{_ds['vol_uns']:.2f}%"),
            ("Sharpe", f"{_ds['sharpe_rep']:.2f}", f"{_ds['sharpe_uns']:.2f}"),
            ("Beta vs MSCI World Hdg", f"{_ds['beta_rep']:.2f}", f"{_ds['beta_uns']:.2f}"),
        ]
        _h = ('<table style="width:100%;border-collapse:collapse;font-size:15px;color:#1A1A1A;">'
              '<tr><th style="text-align:left;padding:7px 10px;border-bottom:2px solid #006B7A;">Metric</th>'
              '<th style="text-align:right;padding:7px 10px;border-bottom:2px solid #006B7A;">Reported</th>'
              '<th style="text-align:right;padding:7px 10px;border-bottom:2px solid #006B7A;">Unsmoothed</th></tr>')
        for k, a, b in _drows:
            _h += (f'<tr><td style="padding:7px 10px;border-bottom:1px solid #E5ECEC;">{k}</td>'
                   f'<td style="padding:7px 10px;border-bottom:1px solid #E5ECEC;text-align:right;font-family:JetBrains Mono, monospace;font-weight:600;">{a}</td>'
                   f'<td style="padding:7px 10px;border-bottom:1px solid #E5ECEC;text-align:right;font-family:JetBrains Mono, monospace;font-weight:600;">{b}</td></tr>')
        _h += '</table>'
        st.markdown(_h, unsafe_allow_html=True)
        st.plotly_chart(chart_desmooth(fund_df, MSCI_DF), use_container_width=True,
                        config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — CO-MOVEMENT
# ══════════════════════════════════════════════════════════════════════════════

with tabs[9]:
    st.markdown('<div class="mg-sh" style="margin-top:14px;">Co-Movement in Market Extremes</div>', unsafe_allow_html=True)
    c_bw_l, c_bw_r = st.columns(2, gap="large")
    with c_bw_l:
        st.markdown('<div class="mg-sh">Market\'s Worst 5 Months</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_best_worst(fund_df, MSCI_DF, 'MSCI World Hdg', AGG_DF, 'Bloomberg Agg', 5, worst=True, bm3_df=bm3_df, bm3_name=bm3_name, show_legend=True, fund_name='Strategy'), use_container_width=True, config={'displayModeBar': False}, key='cw_worst')
    with c_bw_r:
        st.markdown('<div class="mg-sh">Market\'s Best 5 Months</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_best_worst(fund_df, MSCI_DF, 'MSCI World Hdg', AGG_DF, 'Bloomberg Agg', 5, worst=False, bm3_df=bm3_df, bm3_name=bm3_name, show_legend=False, fund_name='Strategy'), use_container_width=True, config={'displayModeBar': False}, key='cw_best')

    st.markdown('<div class="mg-sh" style="margin-top:14px;">Up / Down Capture</div>', unsafe_allow_html=True)
    st.markdown('<div class="mg-note">Average monthly return in up-market vs. down-market months. Up and down months are defined by the market (MSCI World Hedged) being positive or negative.</div>', unsafe_allow_html=True)
    st.plotly_chart(
        chart_up_down_capture(fund_df, 'Strategy', MSCI_DF, 'MSCI World Hdg',
                              others=[('Bloomberg Agg', AGG_DF), (bm3_name, bm3_df)]),
        use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — WATERFALL
# ══════════════════════════════════════════════════════════════════════════════

with tabs[10]:
    st.markdown('<div class="mg-sh" style="margin-top:14px;">Strategy vs Market Waterfall</div>', unsafe_allow_html=True)
    st.markdown('<div class="mg-note">Every overlapping month sorted by market return, best (left) to worst (right). Bottom: MSCI World Hedged, shaded by its own return. Top: the fund over the same ordering — blue when positive, orange/red when negative. Hover any bar for the month.</div>', unsafe_allow_html=True)
    st.plotly_chart(
        chart_waterfall(fund_df, 'Strategy', MSCI_DF, 'MSCI World Hdg'),
        use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ROLLING
# ══════════════════════════════════════════════════════════════════════════════

with tabs[11]:
    if len(fund_df) < 12:
        st.warning("Need at least 12 months of data for rolling metrics.")
    else:
        st.markdown('<div class="mg-sh">Rolling 12-Month Return</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_rolling(fund_df, 'roll_ret'), use_container_width=True, config={'displayModeBar': False})
        c_rs, c_rv = st.columns(2, gap="large")
        with c_rs:
            st.markdown('<div class="mg-sh">Rolling 12-Month Sharpe</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_rolling(fund_df, 'roll_sharpe'), use_container_width=True, config={'displayModeBar': False})
        with c_rv:
            st.markdown('<div class="mg-sh">Rolling 12-Month Volatility</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_rolling(fund_df, 'roll_vol'), use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SEASONALITY
# ══════════════════════════════════════════════════════════════════════════════

with tabs[12]:
    seas = seasonality(fund_df)
    c_sm, c_sq = st.columns(2, gap="large")
    with c_sm:
        st.markdown('<div class="mg-sh">Average Return by Month</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_seasonality_monthly(seas), use_container_width=True, config={'displayModeBar': False})
    with c_sq:
        st.markdown('<div class="mg-sh">Average Return by Quarter</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_seasonality_quarterly(seas), use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div class="mg-sh" style="margin-top:8px;">Monthly Seasonality Detail</div>', unsafe_allow_html=True)
    s_html = '<table class="mg-tbl"><thead><tr><th class="lbl">Month</th><th>Avg Return</th><th>Std Dev</th><th>N</th><th>Hit Rate</th></tr></thead><tbody>'
    for _, row in seas['monthly'].iterrows():
        m_idx = int(row['month'])
        sub   = fund_df[fund_df['month'] == m_idx]['ret']
        hit   = int(np.sum(sub > 0)) / len(sub) * 100 if len(sub) > 0 else 0
        cls   = 'pos' if row['mean'] >= 0 else 'neg'
        s_html += f"<tr><td class='lbl'>{MN[m_idx-1]}</td><td class='{cls}'>{row['mean']:.2f}%</td><td>{row['std']:.2f}%</td><td>{int(row['count'])}</td><td>{hit:.2f}%</td></tr>"
    s_html += '</tbody></table>'
    st.markdown(s_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — MULTI-PERIOD
# ══════════════════════════════════════════════════════════════════════════════

with tabs[13]:
    last_year  = int(fund_df.iloc[-1]['year'])
    last_month = int(fund_df.iloc[-1]['month'])

    def _slice(yr):
        return fund_df[
            (fund_df['year'] > last_year - yr) |
            ((fund_df['year'] == last_year - yr) & (fund_df['month'] >= last_month))
        ].copy()

    periods = [
        ('Since Inception', fund_df.copy()),
        ('5 Year',  _slice(5)),
        ('3 Year',  _slice(3)),
        ('1 Year',  _slice(1)),
    ]
    pstats = [(lbl, period_stats(df)) for lbl, df in periods]

    m_defs = [
        ('Ann. Return (%)',     'ann_ret',  True,  2, False),
        ('Ann. Volatility (%)', 'ann_vol',  False, 2, False),
        ('Sharpe (Rf=TB3MS)',   'sharpe',   False, 2, False),
        ('Sortino (Rf=TB3MS)',  'sortino',  False, 2, False),
        ('Calmar Ratio',        'calmar',   False, 2, False),
        ('Max Drawdown (%)',    'max_dd',   True,  2, True),
        ('Hit Rate (%)',        'hit_rate', False, 1, False),
        ('Best Month (%)',      'best',     True,  2, False),
        ('Worst Month (%)',     'worst',    True,  2, True),
        ('Skewness',            'skew',     False, 4, False),
        ('Excess Kurtosis',     'exkurt',   False, 4, False),
        ('N Observations',      'n',        False, 0, False),
    ]

    mp_html = '<table class="mg-tbl"><thead><tr><th class="lbl">Metric</th>'
    for lbl, _ in pstats:
        mp_html += f'<th>{lbl}</th>'
    mp_html += '</tr></thead><tbody>'

    for (m_lbl, m_key, pct, dec, flip) in m_defs:
        mp_html += f'<tr><td class="lbl">{m_lbl}</td>'
        for _, ps in pstats:
            if ps is None:
                mp_html += '<td>—</td>'; continue
            v = ps.get(m_key, np.nan)
            if m_key == 'n':
                mp_html += f'<td>{int(v)}</td>'; continue
            if v is None or (isinstance(v, float) and np.isnan(v)):
                mp_html += '<td>—</td>'; continue
            c = _cls(float(v), flip)
            mp_html += f'<td class="{c}">{_fmt(float(v), dec, pct)}</td>'
        mp_html += '</tr>'
    mp_html += '</tbody></table>'
    st.markdown(mp_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — MACRO EVENTS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[14]:
    ev_rows = macro_events_table(fund_df, bm1_df)
    if not ev_rows:
        st.info("No macro event periods overlap with this fund's history.")
    else:
        bm_hdrs = f'<th>{bm1_name}</th><th>Spread</th><th>Protected?</th>' if bm1_df is not None else ''
        ev_html = f'<table class="mg-tbl"><thead><tr><th class="lbl">Event</th><th class="lbl">Period</th><th>{fund_name}</th>{bm_hdrs}</tr></thead><tbody>'
        for ev in ev_rows:
            fc = 'pos' if ev['fund_ret'] >= 0 else 'neg'
            ev_html += f'<tr><td class="lbl"><strong>{ev["name"]}</strong></td><td class="lbl">{ev["period"]}</td><td class="{fc}">{_fmt(ev["fund_ret"])}</td>'
            if ev['bm_ret'] is not None:
                bc   = 'pos' if ev['bm_ret'] >= 0 else 'neg'
                sc   = 'pos' if ev['spread'] >= 0 else 'neg'
                prot = '<span style="color:#1A8A50;font-weight:600">✓ Yes</span>' if ev['spread'] > 0 else '<span style="color:#CC2222;">✗ No</span>'
                ev_html += f'<td class="{bc}">{_fmt(ev["bm_ret"])}</td><td class="{sc}">{_fmt(ev["spread"],2,False)} pp</td><td>{prot}</td>'
            ev_html += '</tr>'
        ev_html += '</tbody></table>'
        st.markdown(ev_html, unsafe_allow_html=True)
        st.markdown('<div class="mg-note">Compound returns over each event window &nbsp;·&nbsp; Protected = fund outperformed benchmark</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — SHOCKS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[15]:
    st.markdown('<div class="mg-sh" style="margin-top:14px;">Performance During Discrete Shock Events</div>', unsafe_allow_html=True)
    st.markdown('<div class="mg-note">The four events with the largest Strategy-vs-market return differential. Each panel shows the compound return of every series over that window — blue when positive, orange/red when negative.</div>', unsafe_allow_html=True)
    st.plotly_chart(
        chart_shocks(fund_df, 'Strategy', MSCI_DF, 'MSCI World Hdg',
                     AGG_DF, 'Bloomberg Agg', bm3_df=bm3_df, bm3_name=bm3_name),
        use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 13 — DDQ
# ══════════════════════════════════════════════════════════════════════════════

with tabs[16]:
    st.markdown('<div class="mg-sh" style="margin-top:14px;">Due-Diligence Questionnaire</div>', unsafe_allow_html=True)
    st.markdown('<div class="mg-note">Tags: Clarify · Assumptions · Evidence · Perspective · Implications · Meta</div>', unsafe_allow_html=True)
    _qno = 0
    for _ci, (_cat, _qs) in enumerate(build_ddq(fund_df, fund_name, MSCI_DF,
                                                 meta=st.session_state.get('deck_meta'))):
        _letter = chr(65 + _ci)
        st.markdown(f'<div class="mg-sh" style="margin-top:18px;">{_letter}. {_cat}</div>', unsafe_allow_html=True)
        _rows = ''
        for _q, _ty in _qs:
            _qno += 1
            _rows += (f'<div style="color:#1A1A1A;font-size:15px;line-height:1.6;margin:0 0 8px 0;">'
                      f'<span style="color:#006B7A;font-weight:700;">{_qno}.</span> {_q} '
                      f'<span style="color:#006B7A;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;">[{_ty}]</span></div>')
        st.markdown(_rows, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 14 — DATA
# ══════════════════════════════════════════════════════════════════════════════

with tabs[17]:
    _, c_s = st.columns([3, 1])
    with c_s:
        search = st.text_input("Filter rows", placeholder="e.g. 2020 or Mar")

    disp = fund_df_raw.copy()
    disp['date'] = disp.apply(lambda r: f"{MN[int(r['month'])-1]} {int(r['year'])}", axis=1)
    if outliers_df is not None:
        out_set      = set(zip(outliers_df['year'], outliers_df['month']))
        disp['flag'] = disp.apply(lambda r: (r['year'], r['month']) in out_set, axis=1)
    else:
        disp['flag'] = False

    if search:
        mask = (
            disp['date'].str.lower().str.contains(search.lower()) |
            disp['ret'].astype(str).str.contains(search)
        )
        disp = disp[mask]

    d_html = f'<table class="mg-tbl"><thead><tr><th class="lbl">Date</th><th>{fund_name} Return</th><th></th></tr></thead><tbody>'
    for r in disp.itertuples():
        bg    = 'background:#FFFBEE;' if r.flag else ''
        cls   = 'pos' if r.ret >= 0 else 'neg'
        badge = '<span class="mg-badge">Outlier</span>' if r.flag else ''
        d_html += f"<tr style='{bg}'><td class='lbl'>{r.date}</td><td class='{cls}'>{_fmt(r.ret,2,True)}</td><td>{badge}</td></tr>"
    d_html += '</tbody></table>'
    st.markdown(d_html, unsafe_allow_html=True)
    st.markdown(f'<div class="mg-note">Showing {len(disp)} of {len(fund_df_raw)} rows &nbsp;·&nbsp; Highlighted rows are ≥3σ outliers</div>', unsafe_allow_html=True)
