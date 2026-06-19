"""
app.py — Meatgrinder Performance Analytics
Clean professional UI. Works within Streamlit's actual CSS constraints.
Light theme so native widgets blend naturally.
"""

import hmac
import numpy as np
import pandas as pd
import streamlit as st

from analytics import (
    geo_return, ann_vol, sharpe_with_rf, sortino_with_rf,
    max_drawdown, calmar, skewness, excess_kurtosis, jarque_bera,
    compute_outliers, normality_tests, top_drawdowns, period_stats,
    seasonality, macro_events_table, piecewise_beta_regression,
    acf, acf_conf_band, rolling_metrics, parse_uploaded_file,
    MSCI_DF, AGG_DF,
)
from charts import (
    chart_cumulative, chart_drawdowns, chart_monthly_bars,
    chart_histogram, chart_qq, chart_acf, chart_calendar_heatmap,
    chart_rolling, chart_regression, chart_seasonality_monthly,
    chart_seasonality_quarterly, MN,
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Meatgrinder',
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
  background: #F7F5F0 !important;
}
.main .block-container {
  padding: 0 0 80px 0 !important;
  max-width: 100% !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Controls area — give it a white strip ── */
[data-testid="stHorizontalBlock"]:first-of-type {
  background: #FFFFFF;
  padding: 12px 40px;
  border-bottom: 1px solid #E0DDD6;
  gap: 24px !important;
}

/* ── Widget label text ── */
label[data-testid="stWidgetLabel"] p {
  font-family: 'Inter', sans-serif !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 1.2px !important;
  color: #888880 !important;
}

/* ── Tabs — these selectors are confirmed in ST 1.58 ── */
.stTabs [data-baseweb="tab-list"] {
  background: #FFFFFF !important;
  border-bottom: 1px solid #E0DDD6 !important;
  padding: 0 40px !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Inter', sans-serif !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  color: #999990 !important;
  padding: 14px 20px !important;
  border-radius: 0 !important;
  border-bottom: 3px solid transparent !important;
  background: transparent !important;
  margin: 0 !important;
}
.stTabs [aria-selected="true"] {
  color: #1A1A1A !important;
  border-bottom: 3px solid #1A1A1A !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }
.stTabs [data-baseweb="tab-panel"] {
  padding: 28px 40px 0 !important;
}

/* ── Custom HTML elements — these we own 100% ── */

/* Top bar */
.mg-topbar {
  background: #1A1A1A;
  height: 54px;
  padding: 0 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0;
}
.mg-topbar .brand {
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 4px;
  color: #FFFFFF;
  text-transform: uppercase;
}
.mg-topbar .info {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #666;
  letter-spacing: 0.5px;
}

/* Fund identity */
.mg-id {
  padding: 20px 40px 16px;
  background: #FFFFFF;
  border-bottom: 1px solid #E0DDD6;
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
  color: #999990;
  letter-spacing: 0.3px;
}

/* Diagnostic strip */
.mg-diag {
  background: #F0EDE6;
  border-bottom: 1px solid #E0DDD6;
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
  color: #AAA898;
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
  background: #1A1A1A;
  border-bottom: 3px solid #1A1A1A;
}
.mg-stat {
  padding: 20px 24px 18px;
  border-right: 1px solid #2A2A2A;
}
.mg-stat:last-child { border-right: none; }
.mg-stat .s-lbl {
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: #666;
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
.mg-stat .s-val.dim { color: #AAAAAA; }

/* Section heading */
.mg-sh {
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: #999990;
  padding-bottom: 10px;
  border-bottom: 1px solid #E0DDD6;
  margin-bottom: 16px;
  margin-top: 0;
}

/* Data tables */
table.mg-tbl {
  width: 100%;
  border-collapse: collapse;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  border: 1px solid #E0DDD6;
  border-radius: 4px;
  overflow: hidden;
  background: #FFFFFF;
}
table.mg-tbl th {
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: #999990;
  background: #F7F5F0;
  padding: 10px 14px;
  text-align: left;
  border-bottom: 1px solid #E0DDD6;
  white-space: nowrap;
}
table.mg-tbl td {
  padding: 8px 14px;
  border-bottom: 1px solid #F0EDE6;
  color: #1A1A1A;
}
table.mg-tbl tr:last-child td { border-bottom: none; }
table.mg-tbl tr:hover td      { background: #FAFAF8; }
table.mg-tbl td.pos { color: #1A8A50; font-weight: 500; }
table.mg-tbl td.neg { color: #CC2222; font-weight: 500; }
table.mg-tbl td.gld { color: #9A6800; font-weight: 500; }
table.mg-tbl td.lbl { color: #888880; }

/* Note text */
.mg-note {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #AAAAAA;
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
  color: #999990;
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
  background: #1A1A1A;
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

    # Force the entire page black via CSS, then use columns to center content
    st.markdown("""
    <style>
    .stApp { background: #1A1A1A !important; }
    .main .block-container { background: #1A1A1A !important; padding-top: 15vh !important; }
    /* Make the password label and input visible on dark background */
    [data-testid="stTextInput"] label p {
        color: #999999 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }
    [data-testid="stTextInput"] input {
        background: #2A2A2A !important;
        border: 1px solid #444444 !important;
        border-radius: 3px !important;
        color: #FFFFFF !important;
        font-size: 16px !important;
        padding: 12px 16px !important;
    }
    [data-testid="stTextInput"] input::placeholder { color: #666666 !important; }
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.markdown("""
        <div style="text-align:center; padding-bottom: 48px;">
          <div style="font-family:'Inter',sans-serif; font-size:42px; font-weight:700;
                      letter-spacing:8px; color:#FFFFFF; text-transform:uppercase;
                      margin-bottom:14px;">
            MEATGRINDER
          </div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:12px;
                      color:#AAAAAA; letter-spacing:4px;">
            PERFORMANCE ANALYTICS
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.text_input("PASSWORD", type="password", key="pw_input",
                      on_change=_submit, label_visibility="visible",
                      placeholder="Enter password")

        if st.session_state.get("pw_wrong"):
            st.error("Incorrect password")
            st.session_state["pw_wrong"] = False

    return False

if not _check_password():
    st.stop()


# ─── SESSION STATE ────────────────────────────────────────────────────────────

for k, v in {
    'fund_df': None, 'fund_name': 'Fund',
    'outliers_df': None, 'trimmed': False,
    'parse_diag': None, 'bm_choice': 'MSCI World Hedged USD',
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── TOP BAR ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="mg-topbar">
  <div class="brand">Meatgrinder</div>
</div>
""", unsafe_allow_html=True)


# ─── LANDING ──────────────────────────────────────────────────────────────────

if st.session_state['fund_df'] is None:

    # Centered layout: heading + uploader + requirements box, all together
    _, col_up, _ = st.columns([1, 1.4, 1])
    with col_up:
        st.markdown("""
        <div style="padding: 60px 0 32px; text-align: center;">
          <div style="font-family: 'Inter', sans-serif; font-size: 36px; font-weight: 700;
                      color: #1A1A1A; margin-bottom: 8px;">
            Upload a fund file to begin
          </div>
          <div style="font-family: 'Inter', sans-serif; font-size: 16px;
                      color: #666660; margin-bottom: 32px;">
            Drag and drop or click to select a CSV or Excel file
          </div>
        </div>
        """, unsafe_allow_html=True)

        up = st.file_uploader("", type=['csv','xlsx','xls'], label_visibility="collapsed")

        st.markdown("""
        <div style="margin-top: 32px; border: 1px solid #E0DDD6; border-radius: 6px;
                    background: #FFFFFF; padding: 24px 28px;">
          <div style="font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 700;
                      text-transform: uppercase; letter-spacing: 1.5px; color: #1A1A1A;
                      margin-bottom: 16px; border-bottom: 1px solid #E0DDD6; padding-bottom: 10px;">
            File Requirements
          </div>
          <div style="font-family: 'Inter', sans-serif; font-size: 14px; color: #333330;
                      line-height: 2;">
            <div>1. &nbsp; Format: CSV or Excel (.csv, .xlsx, .xls)</div>
            <div>2. &nbsp; One column of dates (any standard date format)</div>
            <div>3. &nbsp; One column of monthly returns (% or decimal — auto-detected)</div>
            <div>4. &nbsp; Column headers optional — app detects columns automatically</div>
            <div>5. &nbsp; Minimum 6 months of data required</div>
          </div>
          <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #E0DDD6;
                      font-family: 'Inter', sans-serif; font-size: 13px; color: #888880;">
            Benchmarks (MSCI World Hedged, Bloomberg Global Agg) and TB3MS risk-free rates
            are built in — no need to upload them.
          </div>
        </div>
        """, unsafe_allow_html=True)

    if up is not None:
        df, err, diag = parse_uploaded_file(up)
        if err:
            st.error(err)
        else:
            st.session_state.update({
                'fund_df': df,
                'outliers_df': compute_outliers(df),
                'parse_diag': diag,
            })
            st.rerun()
    st.stop()


# ─── CONTROLS BAR ─────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns([2, 1.4, 1.6, 1.2, 0.8])

with c1:
    new_up = st.file_uploader("Load New File", type=['csv','xlsx','xls'])
    if new_up is not None:
        df2, err2, diag2 = parse_uploaded_file(new_up)
        if err2:
            st.error(err2)
        else:
            st.session_state.update({
                'fund_df': df2,
                'outliers_df': compute_outliers(df2),
                'parse_diag': diag2,
            })
            st.rerun()

with c2:
    fn = st.text_input("Fund Name", value=st.session_state['fund_name'])
    st.session_state['fund_name'] = fn

with c3:
    opts = ['MSCI World Hedged USD', 'Bloomberg Global Agg', 'None']
    bm = st.selectbox("Benchmark", opts,
                      index=opts.index(st.session_state['bm_choice']))
    st.session_state['bm_choice'] = bm

with c4:
    tr = st.toggle("Exclude ≥3σ Outliers", value=st.session_state['trimmed'])
    st.session_state['trimmed'] = tr

with c5:
    st.write("")
    st.write("")
    if st.button("Clear Data", use_container_width=True):
        for k in ['fund_df','outliers_df','parse_diag']:
            st.session_state[k] = None
        st.rerun()


# ─── RESOLVE DATA ─────────────────────────────────────────────────────────────

fund_df_raw = st.session_state['fund_df']
outliers_df = st.session_state['outliers_df']
diag        = st.session_state.get('parse_diag') or {}
fund_name   = st.session_state['fund_name']
trimmed     = st.session_state['trimmed']
bm_choice   = st.session_state['bm_choice']

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
    st.markdown(f"""
<div class="mg-diag">
  <div class="d-item"><div class="d-lbl">Date Column</div><div class="d-val">{diag.get('date_col','—')}</div></div>
  <div class="d-item"><div class="d-lbl">Format</div><div class="d-val">{diag.get('date_format','—')}</div></div>
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
        f"{MN[int(r.month)-1]} {int(r.year)} ({r.ret:+.1f}%)"
        for r in outliers_df.itertuples()
    )
    st.markdown(
        f'<div class="mg-alert">⚡ <strong>Outlier Alert</strong> &nbsp;—&nbsp; {out_list}'
        f'&nbsp; · &nbsp; Toggle "Exclude ≥3σ Outliers" above to recompute without these months.</div>',
        unsafe_allow_html=True
    )


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
    <div class="s-val dim">{_fmt(sh, 3, False)}</div>
  </div>
  <div class="mg-stat">
    <div class="s-lbl">Sortino · Rf=TB3MS</div>
    <div class="s-val dim">{_fmt(so, 3, False)}</div>
  </div>
  <div class="mg-stat">
    <div class="s-lbl">Max Drawdown</div>
    <div class="s-val neg">{_fmt(md)}</div>
  </div>
  <div class="mg-stat">
    <div class="s-lbl">Calmar Ratio</div>
    <div class="s-val dim">{_fmt(ca, 3, False)}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── TABS ─────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Summary", "Calendar", "Drawdowns", "Distribution",
    "Regression", "Rolling", "Seasonality", "Multi-Period",
    "Macro Events", "Data",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
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
            ("Sharpe (Rf=TB3MS)",  sharpe_with_rf(fund_df_raw),     sharpe_with_rf(fund_df),         False, 3,  False),
            ("Sortino (Rf=TB3MS)", sortino_with_rf(fund_df_raw),    sortino_with_rf(fund_df),        False, 3,  False),
            ("Calmar Ratio",       calmar(full_rets),               calmar(trim_rets),               False, 3,  False),
            ("Max Drawdown",       max_drawdown(full_rets),         max_drawdown(trim_rets),         True,  2,  True),
            ("Skewness",           skewness(full_rets),             skewness(trim_rets),             False, 4,  False),
            ("Excess Kurtosis",    excess_kurtosis(full_rets),      excess_kurtosis(trim_rets),      False, 4,  False),
            ("Jarque-Bera Stat",   jb_full,                         jb_trim,                         False, 2,  False),
            ("Best Month",         float(np.max(full_rets)),        float(np.max(trim_rets)),        True,  2,  False),
            ("Worst Month",        float(np.min(full_rets)),        float(np.min(trim_rets)),        True,  2,  True),
            ("Avg Monthly Return", float(np.mean(full_rets)),       float(np.mean(trim_rets)),       True,  3,  False),
            ("Hit Rate",           float(np.sum(full_rets>0)/len(full_rets)*100),
                                   float(np.sum(trim_rets>0)/max(len(trim_rets),1)*100), False, 1, False),
            ("Up Months",          int(np.sum(full_rets > 0)),      int(np.sum(trim_rets > 0)),      False, 0,  False),
            ("Down Months",        int(np.sum(full_rets < 0)),      int(np.sum(trim_rets < 0)),      False, 0,  False),
        ]

        html = f"""<table class="mg-tbl">
<thead><tr>
  <th>Metric</th>
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

            bm1_al = fund_df_raw.merge(bm1_df, on=['year','month'], suffixes=('','_b'))\
                                 .rename(columns={'ret_b':'ret'})[['year','month','ret']]
            bm2_al = None
            if bm2_df is not None:
                bm2_al = fund_df_raw.merge(bm2_df, on=['year','month'], suffixes=('','_b'))\
                                     .rename(columns={'ret_b':'ret'})[['year','month','ret']]

            fs = _bm_stats(fund_df_raw)
            b1 = _bm_stats(bm1_al)
            b2 = _bm_stats(bm2_al) if bm2_al is not None else {}

            bm_defs = [
                ('Ann. Return',     True,  2, False),
                ('Ann. Volatility', False, 2, False),
                ('Sharpe',          False, 3, False),
                ('Sortino',         False, 3, False),
                ('Max Drawdown',    True,  2, True),
                ('Calmar Ratio',    False, 3, False),
                ('Skewness',        False, 4, False),
                ('Excess Kurtosis', False, 4, False),
            ]
            b2h = f'<th>{bm2_name}</th>' if bm2_al is not None else ''
            html2 = f"""<table class="mg-tbl">
<thead><tr><th>Metric</th><th>{fund_name}</th><th>{bm1_name}</th>{b2h}</tr></thead><tbody>"""
            for (m, pct, dec, flip) in bm_defs:
                html2 += f'<tr><td class="lbl">{m}</td>'
                for st_d in ([fs, b1] + ([b2] if bm2_al else [])):
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
            chart_cumulative(fund_df, fund_name, bm1_df, bm1_name or '', bm2_df, bm2_name or ''),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CALENDAR
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown('<div class="mg-sh">Monthly Return Calendar</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_calendar_heatmap(fund_df), use_container_width=True)
    st.markdown('<div class="mg-sh" style="margin-top:8px;">Monthly Return Bars</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_monthly_bars(fund_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DRAWDOWNS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown('<div class="mg-sh">Drawdown Series</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_drawdowns(fund_df), use_container_width=True)

    st.markdown('<div class="mg-sh" style="margin-top:8px;">Top Drawdown Episodes</div>', unsafe_allow_html=True)
    episodes = top_drawdowns(fund_df)
    ep_html = '<table class="mg-tbl"><thead><tr><th>#</th><th>Drawdown</th><th>Peak</th><th>Trough</th><th>Recovery</th><th>Peak→Trough</th><th>Trough→Rec.</th><th>Total</th></tr></thead><tbody>'
    for i, ep in enumerate(episodes):
        pt  = str(ep['peak_to_trough']) + 'm'
        tr  = str(ep['trough_to_recovery']) + 'm' if ep['trough_to_recovery'] is not None else 'Ongoing'
        tot = str(ep['total_months']) + 'm'        if ep['total_months'] is not None else 'Ongoing'
        ep_html += (f"<tr><td class='lbl'>{i+1}</td><td class='neg'>{_fmt(ep['drawdown'])}</td>"
                    f"<td class='lbl'>{ep['peak_date']}</td><td class='lbl'>{ep['trough_date']}</td>"
                    f"<td class='lbl'>{ep['recovery_date']}</td><td class='lbl'>{pt}</td>"
                    f"<td class='lbl'>{tr}</td><td class='lbl'>{tot}</td></tr>")
    ep_html += '</tbody></table>'
    st.markdown(ep_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    c_h, c_q = st.columns([3, 2], gap="large")
    with c_h:
        st.markdown('<div class="mg-sh">Return Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_histogram(fund_df, outliers_df if not trimmed else None), use_container_width=True)
    with c_q:
        st.markdown('<div class="mg-sh">Q-Q Plot vs Normal</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_qq(fund_df), use_container_width=True)

    st.markdown('<div class="mg-sh" style="margin-top:8px;">Normality Tests</div>', unsafe_allow_html=True)
    norm = normality_tests(rets)
    n_html = '<table class="mg-tbl"><thead><tr><th>Test</th><th>Statistic</th><th>p-value / Critical</th><th>Reject Normality?</th></tr></thead><tbody>'
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
    st.plotly_chart(chart_acf(fund_df), use_container_width=True)
    cb = acf_conf_band(len(rets))
    st.markdown(f'<div class="mg-note">95% confidence band: ±{cb:.4f} &nbsp;·&nbsp; Highlighted bars are statistically significant</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    if bm1_df is None:
        st.info("Select a benchmark in the controls bar above.")
    else:
        reg = piecewise_beta_regression(fund_df, bm1_df)
        if reg is None:
            st.warning("Insufficient overlapping data for regression.")
        else:
            c_rl, c_rr = st.columns([3, 2], gap="large")
            with c_rl:
                st.markdown(f'<div class="mg-sh">{fund_name} vs {bm1_name}</div>', unsafe_allow_html=True)
                st.plotly_chart(chart_regression(reg, fund_name, bm1_name), use_container_width=True)
            with c_rr:
                st.markdown('<div class="mg-sh">Regression Statistics</div>', unsafe_allow_html=True)
                from scipy.stats import t as _t
                def _pval(t_stat, n):
                    if t_stat is None or np.isnan(t_stat): return np.nan
                    return float(2 * _t.sf(abs(t_stat), df=n))
                n_reg = reg['n']
                ap = _pval(reg['t_alpha'], n_reg-2)
                bp = _pval(reg['t_beta'],  n_reg-2)
                reg_rows = [
                    ("N (overlapping)",      f"{n_reg}"),
                    ("Alpha (monthly %)",    f"{reg['alpha']:.4f}%"),
                    ("  SE(α)",              f"{reg['se_alpha']:.4f}" if not np.isnan(reg['se_alpha']) else '—'),
                    ("  t(α)",               f"{reg['t_alpha']:.3f}"  if not np.isnan(reg['t_alpha'])  else '—'),
                    ("  p(α)",               f"{ap:.4f}"              if not np.isnan(ap)              else '—'),
                    ("  95% CI",             f"[{reg['alpha_ci'][0]:.3f}, {reg['alpha_ci'][1]:.3f}]"  if not np.isnan(reg['alpha_ci'][0]) else '—'),
                    ("Beta (full period)",   f"{reg['beta']:.4f}"),
                    ("  SE(β)",              f"{reg['se_beta']:.4f}"  if not np.isnan(reg['se_beta']) else '—'),
                    ("  t(β)",               f"{reg['t_beta']:.3f}"   if not np.isnan(reg['t_beta'])  else '—'),
                    ("  p(β)",               f"{bp:.4f}"              if not np.isnan(bp)             else '—'),
                    ("  95% CI",             f"[{reg['beta_ci'][0]:.3f}, {reg['beta_ci'][1]:.3f}]"   if not np.isnan(reg['beta_ci'][0]) else '—'),
                    ("R²",                   f"{reg['r2']:.4f}"),
                    ("Correlation",          f"{reg['corr']:.4f}"),
                    ("β⁺ (BM up months)",    f"{reg['beta_up']:.4f}"   if reg['beta_up']   is not None else '—'),
                    ("  N up",               f"{reg['n_up']}"),
                    ("β⁻ (BM down months)",  f"{reg['beta_dn']:.4f}"   if reg['beta_dn']   is not None else '—'),
                    ("  N down",             f"{reg['n_dn']}"),
                    ("Convexity (β⁺ − β⁻)", f"{reg['convexity']:.4f}" if reg['convexity'] is not None else '—'),
                ]
                r_html = '<table class="mg-tbl"><thead><tr><th>Statistic</th><th>Value</th></tr></thead><tbody>'
                for lbl, val in reg_rows:
                    s = 'color:#BBBBBB;padding-left:20px;' if lbl.startswith('  ') else ''
                    r_html += f'<tr><td class="lbl" style="{s}">{lbl.strip()}</td><td>{val}</td></tr>'
                r_html += '</tbody></table>'
                st.markdown(r_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ROLLING
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    if len(fund_df) < 12:
        st.warning("Need at least 12 months of data for rolling metrics.")
    else:
        st.markdown('<div class="mg-sh">Rolling 12-Month Return</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_rolling(fund_df, 'roll_ret'), use_container_width=True)
        c_rs, c_rv = st.columns(2, gap="large")
        with c_rs:
            st.markdown('<div class="mg-sh">Rolling 12-Month Sharpe</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_rolling(fund_df, 'roll_sharpe'), use_container_width=True)
        with c_rv:
            st.markdown('<div class="mg-sh">Rolling 12-Month Volatility</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_rolling(fund_df, 'roll_vol'), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SEASONALITY
# ══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    seas = seasonality(fund_df)
    c_sm, c_sq = st.columns(2, gap="large")
    with c_sm:
        st.markdown('<div class="mg-sh">Average Return by Month</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_seasonality_monthly(seas), use_container_width=True)
    with c_sq:
        st.markdown('<div class="mg-sh">Average Return by Quarter</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_seasonality_quarterly(seas), use_container_width=True)

    st.markdown('<div class="mg-sh" style="margin-top:8px;">Monthly Seasonality Detail</div>', unsafe_allow_html=True)
    s_html = '<table class="mg-tbl"><thead><tr><th>Month</th><th>Avg Return</th><th>Std Dev</th><th>N</th><th>Hit Rate</th></tr></thead><tbody>'
    for _, row in seas['monthly'].iterrows():
        m_idx = int(row['month'])
        sub   = fund_df[fund_df['month'] == m_idx]['ret']
        hit   = int(np.sum(sub > 0)) / len(sub) * 100 if len(sub) > 0 else 0
        cls   = 'pos' if row['mean'] >= 0 else 'neg'
        s_html += f"<tr><td class='lbl'>{MN[m_idx-1]}</td><td class='{cls}'>{row['mean']:.2f}%</td><td>{row['std']:.2f}%</td><td>{int(row['count'])}</td><td>{hit:.0f}%</td></tr>"
    s_html += '</tbody></table>'
    st.markdown(s_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — MULTI-PERIOD
# ══════════════════════════════════════════════════════════════════════════════

with tabs[7]:
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
        ('Sharpe (Rf=TB3MS)',   'sharpe',   False, 3, False),
        ('Sortino (Rf=TB3MS)',  'sortino',  False, 3, False),
        ('Calmar Ratio',        'calmar',   False, 3, False),
        ('Max Drawdown (%)',    'max_dd',   True,  2, True),
        ('Hit Rate (%)',        'hit_rate', False, 1, False),
        ('Best Month (%)',      'best',     True,  2, False),
        ('Worst Month (%)',     'worst',    True,  2, True),
        ('Skewness',            'skew',     False, 4, False),
        ('Excess Kurtosis',     'exkurt',   False, 4, False),
        ('N Observations',      'n',        False, 0, False),
    ]

    mp_html = '<table class="mg-tbl"><thead><tr><th>Metric</th>'
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

with tabs[8]:
    ev_rows = macro_events_table(fund_df, bm1_df)
    if not ev_rows:
        st.info("No macro event periods overlap with this fund's history.")
    else:
        bm_hdrs = f'<th>{bm1_name}</th><th>Spread</th><th>Protected?</th>' if bm1_df is not None else ''
        ev_html = f'<table class="mg-tbl"><thead><tr><th>Event</th><th>Period</th><th>{fund_name}</th>{bm_hdrs}</tr></thead><tbody>'
        for ev in ev_rows:
            fc = 'pos' if ev['fund_ret'] >= 0 else 'neg'
            ev_html += f'<tr><td><strong>{ev["name"]}</strong></td><td class="lbl">{ev["period"]}</td><td class="{fc}">{_fmt(ev["fund_ret"])}</td>'
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
# TAB 9 — DATA
# ══════════════════════════════════════════════════════════════════════════════

with tabs[9]:
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

    d_html = f'<table class="mg-tbl"><thead><tr><th>Date</th><th>{fund_name} Return</th><th></th></tr></thead><tbody>'
    for r in disp.itertuples():
        bg    = 'background:#FFFBEE;' if r.flag else ''
        cls   = 'pos' if r.ret >= 0 else 'neg'
        badge = '<span class="mg-badge">Outlier</span>' if r.flag else ''
        d_html += f"<tr style='{bg}'><td class='lbl'>{r.date}</td><td class='{cls}'>{_fmt(r.ret,2,True)}</td><td>{badge}</td></tr>"
    d_html += '</tbody></table>'
    st.markdown(d_html, unsafe_allow_html=True)
    st.markdown(f'<div class="mg-note">Showing {len(disp)} of {len(fund_df_raw)} rows &nbsp;·&nbsp; Highlighted rows are ≥3σ outliers</div>', unsafe_allow_html=True)
