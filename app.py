"""
app.py — Meatgrinder Performance Analytics
Clean full-width UI. No sidebar. Top navigation bar. Minimal palette.
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

# ─── DESIGN SYSTEM ────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & palette ─────────────────────────────── */
:root {
  --bg:       #F5F0E8;   /* warm parchment */
  --surface:  #FDFAF4;   /* off-white cards */
  --border:   #DDD8CC;
  --text:     #1A1A1A;   /* near-black */
  --muted:    #888070;
  --accent:   #1A1A1A;   /* black accent */
  --green:    #1B6B3A;
  --red:      #B01C2E;
  --gold:     #9A6B00;
  --ui: 'Inter', -apple-system, sans-serif;
  --mono: 'JetBrains Mono', 'Menlo', monospace;
}

/* ── Hide Streamlit chrome ───────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
.main .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Top navigation bar ──────────────────────────── */
.mg-topbar {
  background: #1A1A1A;
  padding: 0 40px;
  height: 52px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 999;
}
.mg-topbar .brand {
  font-family: var(--ui);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 3px;
  color: #FFFFFF;
  text-transform: uppercase;
}
.mg-topbar .meta {
  font-family: var(--mono);
  font-size: 10px;
  color: #888;
  letter-spacing: 0.5px;
}

/* ── Page body ───────────────────────────────────── */
.mg-body {
  background: var(--bg);
  padding: 28px 40px 60px;
  min-height: 100vh;
}

/* ── Fund identity bar ───────────────────────────── */
.mg-identity {
  margin-bottom: 24px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
}
.mg-identity .fund-name {
  font-family: var(--ui);
  font-size: 22px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.3px;
  margin: 0 0 4px;
}
.mg-identity .fund-meta {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  letter-spacing: 0.3px;
}

/* ── Stat banner ─────────────────────────────────── */
.stat-row {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 10px;
  margin-bottom: 24px;
}
.stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 14px 16px 12px;
}
.stat-card .s-label {
  font-family: var(--ui);
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: var(--muted);
  margin-bottom: 6px;
}
.stat-card .s-value {
  font-family: var(--mono);
  font-size: 22px;
  font-weight: 500;
  color: var(--text);
  line-height: 1;
}
.s-value.pos { color: var(--green); }
.s-value.neg { color: var(--red); }
.s-value.gld { color: var(--gold); }

/* ── Alert strip ─────────────────────────────────── */
.mg-alert {
  background: #FEF9EC;
  border: 1px solid #E8D48A;
  border-left: 3px solid var(--gold);
  border-radius: 3px;
  padding: 10px 14px;
  font-family: var(--ui);
  font-size: 12px;
  color: #5A4A00;
  margin-bottom: 20px;
  line-height: 1.5;
}

/* ── Tabs ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  background: transparent;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--ui) !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.8px !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  padding: 10px 18px !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  background: transparent !important;
  margin-right: 0 !important;
}
.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  border-bottom: 2px solid var(--text) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text) !important;
  background: rgba(0,0,0,0.03) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* ── Section headings ────────────────────────────── */
.sec-head {
  font-family: var(--ui);
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--muted);
  margin: 0 0 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

/* ── Data tables ─────────────────────────────────── */
table.mg-tbl {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--mono);
  font-size: 11px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
}
table.mg-tbl th {
  font-family: var(--ui);
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--muted);
  background: #F0EBE0;
  padding: 8px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
table.mg-tbl td {
  padding: 6px 12px;
  border-bottom: 1px solid #EDE8DE;
  color: var(--text);
}
table.mg-tbl tr:last-child td { border-bottom: none; }
table.mg-tbl tr:hover td { background: #F8F4EC; }
table.mg-tbl td.pos { color: var(--green); font-weight: 500; }
table.mg-tbl td.neg { color: var(--red);   font-weight: 500; }
table.mg-tbl td.gld { color: var(--gold);  font-weight: 500; }
table.mg-tbl td.lbl { color: var(--muted); font-family: var(--ui); font-size: 11px; }

/* ── Diagnostic card ─────────────────────────────── */
.diag-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 14px 18px;
  margin-bottom: 20px;
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr;
  gap: 12px 24px;
}
.diag-item .d-label {
  font-family: var(--ui);
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--muted);
  margin-bottom: 3px;
}
.diag-item .d-value {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text);
  font-weight: 500;
}

/* ── Upload zone ─────────────────────────────────── */
.upload-zone {
  background: var(--surface);
  border: 1.5px dashed var(--border);
  border-radius: 6px;
  padding: 48px 32px;
  text-align: center;
  max-width: 560px;
  margin: 80px auto 0;
}
.upload-zone h2 {
  font-family: var(--ui);
  font-size: 18px;
  font-weight: 600;
  color: var(--text);
  margin: 0 0 8px;
  letter-spacing: -0.2px;
}
.upload-zone p {
  font-family: var(--ui);
  font-size: 12px;
  color: var(--muted);
  margin: 0 0 24px;
  line-height: 1.6;
}

/* ── Controls bar ────────────────────────────────── */
.ctrl-bar {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 20px;
  padding: 12px 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
}

/* ── Misc ────────────────────────────────────────── */
.note {
  font-family: var(--ui);
  font-size: 11px;
  color: var(--muted);
  margin-top: 8px;
}
.badge-outlier {
  display: inline-block;
  background: #FEF3CD;
  border: 1px solid #E8D48A;
  color: var(--gold);
  font-family: var(--mono);
  font-size: 9px;
  font-weight: 600;
  padding: 1px 5px;
  border-radius: 2px;
}

/* ── Login page ──────────────────────────────────── */
.login-wrap {
  min-height: 100vh;
  background: #1A1A1A;
  display: flex;
  align-items: center;
  justify-content: center;
}
.login-box {
  text-align: center;
}
.login-box .lg-title {
  font-family: 'Inter', sans-serif;
  font-size: 32px;
  font-weight: 700;
  letter-spacing: 6px;
  color: #FFFFFF;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.login-box .lg-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #666;
  letter-spacing: 2px;
  margin-bottom: 32px;
}

/* ── Streamlit widget overrides ──────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
  font-family: var(--ui) !important;
  font-size: 12px !important;
  background: var(--surface) !important;
  border-color: var(--border) !important;
  border-radius: 3px !important;
}
[data-testid="stFileUploader"] {
  font-family: var(--ui) !important;
  font-size: 12px !important;
}
label[data-testid="stWidgetLabel"] p {
  font-family: var(--ui) !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 1px !important;
  color: var(--muted) !important;
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
    <div style="min-height:100vh;background:#1A1A1A;display:flex;
                align-items:center;justify-content:center;flex-direction:column;">
      <div style="text-align:center;margin-bottom:32px;">
        <div style="font-family:'Inter',sans-serif;font-size:28px;font-weight:700;
                    letter-spacing:6px;color:#fff;text-transform:uppercase;">
          MEATGRINDER
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    color:#555;letter-spacing:2px;margin-top:6px;">
          PERFORMANCE ANALYTICS
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 0.6, 1])
    with col:
        st.text_input("password", type="password", key="pw_input",
                      on_change=_submit, label_visibility="collapsed",
                      placeholder="Password")
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
  <div class="meta">Performance Analytics Engine &nbsp;·&nbsp; Rf = TB3MS (FRED)</div>
</div>
""", unsafe_allow_html=True)


# ─── CONTROLS (always visible above fold) ─────────────────────────────────────

with st.container():
    c1, c2, c3, c4, c5 = st.columns([2.2, 1.8, 1.8, 1.4, 0.8])

    with c1:
        uploaded = st.file_uploader(
            "LOAD FUND DATA",
            type=['csv', 'xlsx', 'xls'],
            label_visibility="visible",
        )

    with c2:
        fund_name = st.text_input(
            "FUND NAME",
            value=st.session_state['fund_name'],
        )
        st.session_state['fund_name'] = fund_name

    with c3:
        bm_choice = st.selectbox(
            "BENCHMARK",
            ['MSCI World Hedged USD', 'Bloomberg Global Agg', 'None'],
            index=['MSCI World Hedged USD','Bloomberg Global Agg','None'].index(
                st.session_state['bm_choice']
            ),
        )
        st.session_state['bm_choice'] = bm_choice

    with c4:
        trimmed = st.toggle(
            "EXCLUDE ≥3σ OUTLIERS",
            value=st.session_state['trimmed'],
        )
        st.session_state['trimmed'] = trimmed

    with c5:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.session_state['fund_df'] is not None:
            if st.button("✕ Clear", use_container_width=True):
                st.session_state['fund_df'] = None
                st.session_state['outliers_df'] = None
                st.session_state['parse_diag'] = None
                st.rerun()

# Handle file upload
if uploaded is not None:
    df, err, diag = parse_uploaded_file(uploaded)
    if err:
        st.error(f"Upload error: {err}")
    else:
        st.session_state['fund_df'] = df
        st.session_state['outliers_df'] = compute_outliers(df)
        st.session_state['parse_diag'] = diag

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ─── RESOLVE WORKING DATA ─────────────────────────────────────────────────────

fund_df_raw = st.session_state['fund_df']
outliers_df = st.session_state['outliers_df']
diag        = st.session_state.get('parse_diag')

# Benchmark setup
if bm_choice == 'MSCI World Hedged USD':
    bm1_df, bm1_name = MSCI_DF, 'MSCI World Hdg'
    bm2_df, bm2_name = AGG_DF,  'Blmbrg Agg'
elif bm_choice == 'Bloomberg Global Agg':
    bm1_df, bm1_name = AGG_DF,  'Bloomberg Agg'
    bm2_df, bm2_name = MSCI_DF, 'MSCI World Hdg'
else:
    bm1_df = bm2_df = bm1_name = bm2_name = None

# Apply outlier trim
if fund_df_raw is not None and trimmed and outliers_df is not None and len(outliers_df) > 0:
    out_set  = set(zip(outliers_df['year'], outliers_df['month']))
    fund_df  = fund_df_raw[
        ~fund_df_raw.apply(lambda r: (r['year'], r['month']) in out_set, axis=1)
    ].copy()
else:
    fund_df = fund_df_raw


# ─── UPLOAD PROMPT ────────────────────────────────────────────────────────────

if fund_df is None:
    st.markdown("""
    <div class="upload-zone">
      <h2>Upload a fund file to begin</h2>
      <p>CSV or Excel with two columns: date and monthly return.<br>
         Percent or decimal format is detected automatically.<br>
         Benchmarks and TB3MS risk-free rates are built in.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def _cls(v, flip=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    return ('neg' if flip else 'pos') if v > 0 else ('pos' if flip else 'neg')

def _fmt(v, dec=2, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f"{v:.{dec}f}{'%' if pct else ''}"


# ─── FUND IDENTITY BAR ────────────────────────────────────────────────────────

rets  = fund_df['ret'].values
n_obs = len(rets)
n_out = len(outliers_df) if outliers_df is not None else 0
first = fund_df.iloc[0]
last  = fund_df.iloc[-1]
date_range = (
    f"{MN[int(first['month'])-1]} {int(first['year'])} "
    f"– {MN[int(last['month'])-1]} {int(last['year'])}"
)
trim_tag = ' &nbsp;<span style="background:#1A1A1A;color:#fff;font-size:9px;padding:2px 7px;border-radius:2px;letter-spacing:1px;">TRIMMED</span>' if trimmed else ''

st.markdown(f"""
<div class="mg-identity">
  <div class="fund-name">{fund_name}{trim_tag}</div>
  <div class="fund-meta">
    {date_range} &nbsp;·&nbsp; {n_obs} monthly observations
    &nbsp;·&nbsp; {bm1_name or 'No benchmark'} &nbsp;·&nbsp; Rf = TB3MS (FRED)
    {f' &nbsp;·&nbsp; {n_out} outlier(s) detected' if n_out > 0 else ''}
  </div>
</div>
""", unsafe_allow_html=True)

# Diagnostic card — shown once after upload
if diag:
    failed = diag.get('failed_dates', [])
    st.markdown(f"""
    <div class="diag-card">
      <div class="diag-item">
        <div class="d-label">File</div>
        <div class="d-value">{diag.get('date_col','?')} · {diag.get('ret_col','?')}</div>
      </div>
      <div class="diag-item">
        <div class="d-label">Date Format Detected</div>
        <div class="d-value">{diag.get('date_format','?')}</div>
      </div>
      <div class="diag-item">
        <div class="d-label">Return Scale</div>
        <div class="d-value">{diag.get('ret_format','?')}</div>
      </div>
      <div class="diag-item">
        <div class="d-label">Rows Parsed / Dropped</div>
        <div class="d-value">{diag.get('n_parsed','?')} / {diag.get('n_dropped',0)}
        {f' &nbsp;⚠ {len(failed)} failed' if failed else ''}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Outlier alert
if n_out > 0 and outliers_df is not None:
    out_list = ', '.join(
        f"{MN[int(r.month)-1]} {int(r.year)} ({r.ret:+.1f}%)"
        for r in outliers_df.itertuples()
    )
    st.markdown(
        f'<div class="mg-alert"><strong>Outlier Alert</strong> &nbsp;—&nbsp; '
        f'{n_out} month{"s" if n_out > 1 else ""} ≥3σ from mean: '
        f'<strong>{out_list}</strong>. '
        f'Toggle "Exclude ≥3σ Outliers" above to recompute without these months.</div>',
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
<div class="stat-row">
  <div class="stat-card">
    <div class="s-label">Ann. Return</div>
    <div class="s-value {_cls(ar)}">{_fmt(ar)}</div>
  </div>
  <div class="stat-card">
    <div class="s-label">Ann. Volatility</div>
    <div class="s-value">{_fmt(av)}</div>
  </div>
  <div class="stat-card">
    <div class="s-label">Sharpe (Rf=TB3MS)</div>
    <div class="s-value">{_fmt(sh, 3, False)}</div>
  </div>
  <div class="stat-card">
    <div class="s-label">Sortino (Rf=TB3MS)</div>
    <div class="s-value">{_fmt(so, 3, False)}</div>
  </div>
  <div class="stat-card">
    <div class="s-label">Max Drawdown</div>
    <div class="s-value neg">{_fmt(md)}</div>
  </div>
  <div class="stat-card">
    <div class="s-label">Calmar Ratio</div>
    <div class="s-value">{_fmt(ca, 3, False)}</div>
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
    sk_f = skewness(full_rets)
    ek_f = excess_kurtosis(full_rets)
    sk_t = skewness(trim_rets)
    ek_t = excess_kurtosis(trim_rets)

    with col_l:
        st.markdown('<div class="sec-head">Full vs. Ex-Outliers Statistics</div>', unsafe_allow_html=True)

        stat_rows = [
            ("Observations",      len(full_rets),                    len(trim_rets),                   False, 0,  False),
            ("Ann. Return",       geo_return(full_rets),             geo_return(trim_rets),             True,  2,  False),
            ("Ann. Volatility",   ann_vol(full_rets),               ann_vol(trim_rets),               False, 2,  False),
            ("Sharpe (Rf=TB3MS)", sharpe_with_rf(fund_df_raw),      sharpe_with_rf(fund_df),          False, 3,  False),
            ("Sortino (Rf=TB3MS)",sortino_with_rf(fund_df_raw),     sortino_with_rf(fund_df),         False, 3,  False),
            ("Calmar Ratio",      calmar(full_rets),                calmar(trim_rets),                False, 3,  False),
            ("Max Drawdown",      max_drawdown(full_rets),          max_drawdown(trim_rets),          True,  2,  True),
            ("Skewness",          sk_f,                             sk_t,                             False, 4,  False),
            ("Excess Kurtosis",   ek_f,                             ek_t,                             False, 4,  False),
            ("Jarque-Bera",       jb_full,                          jb_trim,                          False, 2,  False),
            ("Best Month",        float(np.max(full_rets)),         float(np.max(trim_rets)),         True,  2,  False),
            ("Worst Month",       float(np.min(full_rets)),         float(np.min(trim_rets)),         True,  2,  True),
            ("Avg Monthly Ret",   float(np.mean(full_rets)),        float(np.mean(trim_rets)),        True,  3,  False),
            ("Hit Rate",          float(np.sum(full_rets>0)/len(full_rets)*100),
                                  float(np.sum(trim_rets>0)/max(len(trim_rets),1)*100), False, 1, False),
            ("Up Months",         int(np.sum(full_rets > 0)),       int(np.sum(trim_rets > 0)),       False, 0,  False),
            ("Down Months",       int(np.sum(full_rets < 0)),       int(np.sum(trim_rets < 0)),       False, 0,  False),
        ]

        html = f"""<table class="mg-tbl">
          <thead><tr>
            <th>Metric</th>
            <th>Full (N={len(full_rets)})</th>
            <th>Ex Outliers (N={len(trim_rets)})</th>
          </tr></thead><tbody>"""

        for (lbl, fv, tv, pct, dec, flip) in stat_rows:
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
            st.markdown('<div class="sec-head">Benchmark Comparison</div>', unsafe_allow_html=True)

            def _bm_stats(df):
                r = df['ret'].values
                return {
                    'Ann. Return':      geo_return(r),
                    'Ann. Volatility':  ann_vol(r),
                    'Sharpe':           sharpe_with_rf(df),
                    'Sortino':          sortino_with_rf(df),
                    'Max Drawdown':     max_drawdown(r),
                    'Calmar Ratio':     calmar(r),
                    'Skewness':         skewness(r),
                    'Excess Kurtosis':  excess_kurtosis(r),
                }

            bm1_al = fund_df_raw.merge(bm1_df, on=['year','month'], suffixes=('','_b'))\
                                 .rename(columns={'ret_b':'ret'})[['year','month','ret']]
            bm2_al = None
            if bm2_df is not None:
                bm2_al = fund_df_raw.merge(bm2_df, on=['year','month'], suffixes=('','_b'))\
                                     .rename(columns={'ret_b':'ret'})[['year','month','ret']]

            fund_stats = _bm_stats(fund_df_raw)
            bm1_stats  = _bm_stats(bm1_al)
            bm2_stats  = _bm_stats(bm2_al) if bm2_al is not None else {}

            bm_metrics = [
                ('Ann. Return',     True,  2, False),
                ('Ann. Volatility', False, 2, False),
                ('Sharpe',          False, 3, False),
                ('Sortino',         False, 3, False),
                ('Max Drawdown',    True,  2, True),
                ('Calmar Ratio',    False, 3, False),
                ('Skewness',        False, 4, False),
                ('Excess Kurtosis', False, 4, False),
            ]

            bm2_hdr = f'<th>{bm2_name}</th>' if bm2_al is not None else ''
            html2 = f"""<table class="mg-tbl">
              <thead><tr>
                <th>Metric</th><th>{fund_name}</th><th>{bm1_name}</th>{bm2_hdr}
              </tr></thead><tbody>"""
            for (m, pct, dec, flip) in bm_metrics:
                html2 += f'<tr><td class="lbl">{m}</td>'
                for stats in ([fund_stats, bm1_stats] + ([bm2_stats] if bm2_al else [])):
                    v = stats.get(m, np.nan)
                    try:
                        c = _cls(float(v), flip)
                        html2 += f'<td class="{c}">{_fmt(float(v), dec, pct)}</td>'
                    except Exception:
                        html2 += '<td>—</td>'
                html2 += '</tr>'
            html2 += '</tbody></table>'
            st.markdown(html2, unsafe_allow_html=True)

        st.markdown('<div class="sec-head" style="margin-top:24px;">Cumulative Wealth (Growth of $100)</div>', unsafe_allow_html=True)
        fig_cum = chart_cumulative(fund_df, fund_name, bm1_df, bm1_name or '', bm2_df, bm2_name or '')
        st.plotly_chart(fig_cum, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CALENDAR
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown('<div class="sec-head">Monthly Return Calendar</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_calendar_heatmap(fund_df), use_container_width=True)
    st.markdown('<div class="sec-head" style="margin-top:8px;">Monthly Return Bars</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_monthly_bars(fund_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DRAWDOWNS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown('<div class="sec-head">Drawdown Series</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_drawdowns(fund_df), use_container_width=True)

    st.markdown('<div class="sec-head" style="margin-top:8px;">Top Drawdown Episodes</div>', unsafe_allow_html=True)
    episodes = top_drawdowns(fund_df)
    html_ep = """<table class="mg-tbl"><thead><tr>
      <th>#</th><th>Drawdown</th><th>Peak</th><th>Trough</th><th>Recovery</th>
      <th>Peak→Trough</th><th>Trough→Rec.</th><th>Total</th>
    </tr></thead><tbody>"""
    for i, ep in enumerate(episodes):
        pt  = str(ep['peak_to_trough']) + 'm'
        tr  = (str(ep['trough_to_recovery']) + 'm') if ep['trough_to_recovery'] is not None else 'Ongoing'
        tot = (str(ep['total_months']) + 'm')        if ep['total_months'] is not None else 'Ongoing'
        html_ep += (f"<tr><td class='lbl'>{i+1}</td>"
                    f"<td class='neg'>{_fmt(ep['drawdown'])}</td>"
                    f"<td class='lbl'>{ep['peak_date']}</td>"
                    f"<td class='lbl'>{ep['trough_date']}</td>"
                    f"<td class='lbl'>{ep['recovery_date']}</td>"
                    f"<td class='lbl'>{pt}</td><td class='lbl'>{tr}</td><td class='lbl'>{tot}</td></tr>")
    html_ep += '</tbody></table>'
    st.markdown(html_ep, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    col_h, col_q = st.columns([3, 2], gap="large")
    with col_h:
        st.markdown('<div class="sec-head">Return Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_histogram(fund_df, outliers_df if not trimmed else None), use_container_width=True)
    with col_q:
        st.markdown('<div class="sec-head">Q-Q Plot vs Normal</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_qq(fund_df), use_container_width=True)

    st.markdown('<div class="sec-head" style="margin-top:8px;">Normality Tests</div>', unsafe_allow_html=True)
    norm = normality_tests(rets)

    html_norm = """<table class="mg-tbl"><thead><tr>
      <th>Test</th><th>Statistic</th><th>p-value / Critical</th><th>Reject Normality?</th>
    </tr></thead><tbody>"""

    jb = norm['jb']
    jbr = jb['pval'] < 0.05
    html_norm += (f"<tr><td class='lbl'>Jarque-Bera</td>"
                  f"<td>{jb['stat']:.4f}</td><td>{jb['pval']:.4f}</td>"
                  f"<td class='{'neg' if jbr else 'pos'}'>{'Yes ✗' if jbr else 'No ✓'}</td></tr>")

    if 'sw' in norm:
        sw = norm['sw']
        swr = sw['pval'] < 0.05
        html_norm += (f"<tr><td class='lbl'>Shapiro-Wilk</td>"
                      f"<td>{sw['stat']:.4f}</td><td>{sw['pval']:.4f}</td>"
                      f"<td class='{'neg' if swr else 'pos'}'>{'Yes ✗' if swr else 'No ✓'}</td></tr>")

    ad = norm['ad']
    html_norm += (f"<tr><td class='lbl'>Anderson-Darling</td>"
                  f"<td>{ad['stat']:.4f}</td><td>Crit(5%) = {ad['critical']:.4f}</td>"
                  f"<td class='{'neg' if ad['reject'] else 'pos'}'>{'Yes ✗' if ad['reject'] else 'No ✓'}</td></tr>")

    ks = norm['ks']
    ksr = ks['pval'] < 0.05
    html_norm += (f"<tr><td class='lbl'>Kolmogorov-Smirnov</td>"
                  f"<td>{ks['stat']:.4f}</td><td>{ks['pval']:.4f}</td>"
                  f"<td class='{'neg' if ksr else 'pos'}'>{'Yes ✗' if ksr else 'No ✓'}</td></tr>")

    html_norm += '</tbody></table>'
    st.markdown(html_norm, unsafe_allow_html=True)

    st.markdown('<div class="sec-head" style="margin-top:24px;">Return Autocorrelation — Lags 1–12</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_acf(fund_df), use_container_width=True)
    cb = acf_conf_band(len(rets))
    st.markdown(f'<div class="note">95% confidence band: ±{cb:.4f}. Coloured bars exceed the band (statistically significant).</div>', unsafe_allow_html=True)


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
            col_rl, col_rr = st.columns([3, 2], gap="large")
            with col_rl:
                st.markdown(f'<div class="sec-head">{fund_name} vs {bm1_name}</div>', unsafe_allow_html=True)
                st.plotly_chart(chart_regression(reg, fund_name, bm1_name), use_container_width=True)

            with col_rr:
                st.markdown('<div class="sec-head">Regression Statistics</div>', unsafe_allow_html=True)
                from scipy.stats import t as _t
                def _pval(t_stat, df):
                    if t_stat is None or np.isnan(t_stat): return np.nan
                    return float(2 * _t.sf(abs(t_stat), df=df))

                n_reg = reg['n']
                ap = _pval(reg['t_alpha'], n_reg - 2)
                bp = _pval(reg['t_beta'],  n_reg - 2)

                reg_rows = [
                    ("N (overlapping)",       f"{n_reg}"),
                    ("Alpha (monthly %)",     f"{reg['alpha']:.4f}%"),
                    ("  SE(α)",               f"{reg['se_alpha']:.4f}" if not np.isnan(reg['se_alpha']) else '—'),
                    ("  t(α)",                f"{reg['t_alpha']:.3f}"  if not np.isnan(reg['t_alpha'])  else '—'),
                    ("  p(α)",                f"{ap:.4f}"              if not np.isnan(ap)              else '—'),
                    ("  95% CI α",            f"[{reg['alpha_ci'][0]:.3f}, {reg['alpha_ci'][1]:.3f}]" if not np.isnan(reg['alpha_ci'][0]) else '—'),
                    ("Beta (full period)",    f"{reg['beta']:.4f}"),
                    ("  SE(β)",               f"{reg['se_beta']:.4f}"  if not np.isnan(reg['se_beta']) else '—'),
                    ("  t(β)",                f"{reg['t_beta']:.3f}"   if not np.isnan(reg['t_beta'])  else '—'),
                    ("  p(β)",                f"{bp:.4f}"              if not np.isnan(bp)             else '—'),
                    ("  95% CI β",            f"[{reg['beta_ci'][0]:.3f}, {reg['beta_ci'][1]:.3f}]"  if not np.isnan(reg['beta_ci'][0]) else '—'),
                    ("R²",                    f"{reg['r2']:.4f}"),
                    ("Correlation",           f"{reg['corr']:.4f}"),
                    ("β⁺ (BM up months)",     f"{reg['beta_up']:.4f}"  if reg['beta_up']  is not None else '—'),
                    (f"  N up",               f"{reg['n_up']}"),
                    ("β⁻ (BM down months)",   f"{reg['beta_dn']:.4f}"  if reg['beta_dn']  is not None else '—'),
                    (f"  N down",             f"{reg['n_dn']}"),
                    ("Convexity (β⁺ − β⁻)",  f"{reg['convexity']:.4f}" if reg['convexity'] is not None else '—'),
                ]
                html_reg = '<table class="mg-tbl"><thead><tr><th>Statistic</th><th>Value</th></tr></thead><tbody>'
                for lbl, val in reg_rows:
                    indent = lbl.startswith("  ")
                    style = 'color:var(--muted);padding-left:20px' if indent else ''
                    html_reg += f'<tr><td class="lbl" style="{style}">{lbl.strip()}</td><td>{val}</td></tr>'
                html_reg += '</tbody></table>'
                st.markdown(html_reg, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ROLLING
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    if len(fund_df) < 12:
        st.warning("Need at least 12 months of data for rolling metrics.")
    else:
        st.markdown('<div class="sec-head">Rolling 12-Month Return</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_rolling(fund_df, 'roll_ret'), use_container_width=True)
        c_rs, c_rv = st.columns(2, gap="large")
        with c_rs:
            st.markdown('<div class="sec-head">Rolling 12-Month Sharpe</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_rolling(fund_df, 'roll_sharpe'), use_container_width=True)
        with c_rv:
            st.markdown('<div class="sec-head">Rolling 12-Month Volatility</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_rolling(fund_df, 'roll_vol'), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SEASONALITY
# ══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    seas = seasonality(fund_df)
    c_sm, c_sq = st.columns(2, gap="large")
    with c_sm:
        st.markdown('<div class="sec-head">Average Return by Month</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_seasonality_monthly(seas), use_container_width=True)
    with c_sq:
        st.markdown('<div class="sec-head">Average Return by Quarter</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_seasonality_quarterly(seas), use_container_width=True)

    st.markdown('<div class="sec-head" style="margin-top:8px;">Monthly Detail</div>', unsafe_allow_html=True)
    html_s = '<table class="mg-tbl"><thead><tr><th>Month</th><th>Avg Return</th><th>Std Dev</th><th>N</th><th>Hit Rate</th></tr></thead><tbody>'
    for _, row in seas['monthly'].iterrows():
        m_idx = int(row['month'])
        sub   = fund_df[fund_df['month'] == m_idx]['ret']
        hit   = int(np.sum(sub > 0)) / len(sub) * 100 if len(sub) > 0 else 0
        cls   = 'pos' if row['mean'] >= 0 else 'neg'
        html_s += (f"<tr><td class='lbl'>{MN[m_idx-1]}</td>"
                   f"<td class='{cls}'>{row['mean']:.2f}%</td>"
                   f"<td>{row['std']:.2f}%</td>"
                   f"<td>{int(row['count'])}</td>"
                   f"<td>{hit:.0f}%</td></tr>")
    html_s += '</tbody></table>'
    st.markdown(html_s, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — MULTI-PERIOD
# ══════════════════════════════════════════════════════════════════════════════

with tabs[7]:
    last_year  = int(fund_df.iloc[-1]['year'])
    last_month = int(fund_df.iloc[-1]['month'])

    def _slice(yr_offset):
        sy = last_year - yr_offset
        return fund_df[
            (fund_df['year'] > sy) |
            ((fund_df['year'] == sy) & (fund_df['month'] >= last_month))
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

    html_mp = '<table class="mg-tbl"><thead><tr><th>Metric</th>'
    for lbl, _ in pstats:
        html_mp += f'<th>{lbl}</th>'
    html_mp += '</tr></thead><tbody>'

    for (m_lbl, m_key, pct, dec, flip) in m_defs:
        html_mp += f'<tr><td class="lbl">{m_lbl}</td>'
        for _, ps in pstats:
            if ps is None:
                html_mp += '<td>—</td>'
                continue
            v = ps.get(m_key, np.nan)
            if m_key == 'n':
                html_mp += f'<td>{int(v)}</td>'
                continue
            if v is None or (isinstance(v, float) and np.isnan(v)):
                html_mp += '<td>—</td>'
                continue
            c = _cls(float(v), flip)
            html_mp += f'<td class="{c}">{_fmt(float(v), dec, pct)}</td>'
        html_mp += '</tr>'
    html_mp += '</tbody></table>'
    st.markdown(html_mp, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — MACRO EVENTS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[8]:
    ev_rows = macro_events_table(fund_df, bm1_df)

    if not ev_rows:
        st.info("No macro event periods overlap with this fund's history.")
    else:
        bm_hdrs = f'<th>{bm1_name}</th><th>Spread</th><th>Protected?</th>' if bm1_df is not None else ''
        html_ev = f'<table class="mg-tbl"><thead><tr><th>Event</th><th>Period</th><th>{fund_name}</th>{bm_hdrs}</tr></thead><tbody>'
        for ev in ev_rows:
            fc = 'pos' if ev['fund_ret'] >= 0 else 'neg'
            html_ev += f'<tr><td><strong>{ev["name"]}</strong></td><td class="lbl">{ev["period"]}</td><td class="{fc}">{_fmt(ev["fund_ret"])}</td>'
            if ev['bm_ret'] is not None:
                bc  = 'pos' if ev['bm_ret'] >= 0 else 'neg'
                sc  = 'pos' if ev['spread'] >= 0 else 'neg'
                prot = '<span style="color:var(--green);font-weight:600">✓</span>' if ev['spread'] > 0 else '<span style="color:var(--red);">✗</span>'
                html_ev += (f'<td class="{bc}">{_fmt(ev["bm_ret"])}</td>'
                            f'<td class="{sc}">{_fmt(ev["spread"], 2, False)} pp</td>'
                            f'<td>{prot}</td>')
            html_ev += '</tr>'
        html_ev += '</tbody></table>'
        st.markdown(html_ev, unsafe_allow_html=True)
        st.markdown('<div class="note">Compound returns over each event window. Protected = fund outperformed benchmark.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — DATA
# ══════════════════════════════════════════════════════════════════════════════

with tabs[9]:
    col_d1, col_d2 = st.columns([3, 1])
    with col_d2:
        search = st.text_input("Filter", placeholder="e.g. 2020 or Mar")

    disp = fund_df_raw.copy()
    disp['date'] = disp.apply(lambda r: f"{MN[int(r['month'])-1]} {int(r['year'])}", axis=1)

    if outliers_df is not None:
        out_set = set(zip(outliers_df['year'], outliers_df['month']))
        disp['flag'] = disp.apply(lambda r: True if (r['year'], r['month']) in out_set else False, axis=1)
    else:
        disp['flag'] = False

    if search:
        mask = disp['date'].str.lower().str.contains(search.lower()) | \
               disp['ret'].astype(str).str.contains(search)
        disp = disp[mask]

    html_d = f'<table class="mg-tbl"><thead><tr><th>Date</th><th>{fund_name} Return</th><th></th></tr></thead><tbody>'
    for r in disp.itertuples():
        bg  = 'background:#FEFAE8;' if r.flag else ''
        cls = 'pos' if r.ret >= 0 else 'neg'
        badge = "<span class='badge-outlier'>OUTLIER</span>" if r.flag else ''
        html_d += (f"<tr style='{bg}'>"
                   f"<td class='lbl'>{r.date}</td>"
                   f"<td class='{cls}'>{_fmt(r.ret, 2, True)}</td>"
                   f"<td>{badge}</td></tr>")
    html_d += '</tbody></table>'
    st.markdown(html_d, unsafe_allow_html=True)
    st.markdown(f'<div class="note">Showing {len(disp)} of {len(fund_df_raw)} rows. Yellow rows are ≥3σ outliers.</div>', unsafe_allow_html=True)
