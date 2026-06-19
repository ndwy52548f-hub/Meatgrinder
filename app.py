"""
app.py — Meatgrinder: Institutional Performance Analytics
Streamlit front-end.  All analytics are in analytics.py (no Streamlit dependency).
All charts are in charts.py.
"""

import streamlit as st
import numpy as np
import pandas as pd
from analytics import (
    # Core stats
    geo_return, ann_vol, sharpe_with_rf, sortino_with_rf,
    max_drawdown, calmar, skewness, excess_kurtosis, jarque_bera,
    compute_outliers, normality_tests, top_drawdowns, period_stats,
    seasonality, macro_events_table, piecewise_beta_regression,
    acf, acf_conf_band, rolling_metrics, parse_uploaded_file,
    # Reference data
    MSCI_DF, AGG_DF, MACRO_EVENTS,
)
from charts import (
    chart_cumulative, chart_drawdowns, chart_monthly_bars,
    chart_histogram, chart_qq, chart_acf, chart_calendar_heatmap,
    chart_rolling, chart_regression, chart_seasonality_monthly,
    chart_seasonality_quarterly, C, MN,
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Meatgrinder | Performance Analytics',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inconsolata:wght@400;600&display=swap');

/* Root palette */
:root {
  --bg:       #f4f7fb;
  --surface:  #ffffff;
  --border:   #d0daea;
  --text:     #1a2332;
  --muted:    #7a8ea8;
  --accent:   #1a6cf5;
  --green:    #0a7a50;
  --red:      #c0182a;
  --gold:     #c47d0a;
  --mono: 'Inconsolata', 'Menlo', 'Monaco', 'Consolas', monospace;
  --ui: 'Gill Sans', 'Optima', 'Segoe UI', 'Helvetica Neue', sans-serif;
}

/* Main background */
.main .block-container { background: var(--bg); padding: 1.5rem 2rem 3rem; }
[data-testid="stSidebar"] { background: #1a2332; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: #d0daea !important;
}

/* Header / branding */
.mg-header {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 18px 24px;
  margin-bottom: 20px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.mg-header h1 {
  font-family: var(--ui);
  font-size: 22px; font-weight: 700; letter-spacing: 0.5px;
  color: var(--text); margin: 0 0 4px;
}
.mg-header .sub {
  font-family: var(--mono);
  font-size: 11px; color: var(--muted); line-height: 1.6;
}

/* Stat banner */
.stat-banner {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 12px; margin-bottom: 20px;
}
.stat-cell {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px 14px;
}
.stat-cell .lbl {
  font-family: var(--ui); font-size: 10px; font-weight: 600;
  color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px;
  margin-bottom: 4px;
}
.stat-cell .val {
  font-family: var(--mono); font-size: 20px; font-weight: 600; color: var(--text);
}
.stat-cell .val.pos  { color: var(--green); }
.stat-cell .val.neg  { color: var(--red); }
.stat-cell .val.ac   { color: var(--accent); }
.stat-cell .val.gld  { color: var(--gold); }

/* Card */
.mg-card {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 18px 20px;
  margin-bottom: 16px;
}
.section-title {
  font-family: var(--ui); font-size: 11px; font-weight: 700;
  color: var(--muted); text-transform: uppercase; letter-spacing: 1px;
  margin-bottom: 10px;
}

/* Tables */
table.mg-table {
  width: 100%; border-collapse: collapse;
  font-family: var(--mono); font-size: 11px;
}
table.mg-table th {
  font-family: var(--ui); font-size: 10px; font-weight: 700;
  color: var(--muted); text-transform: uppercase; letter-spacing: 0.7px;
  border-bottom: 1px solid var(--border); padding: 6px 10px; text-align: left;
}
table.mg-table td { padding: 5px 10px; border-bottom: 1px solid #edf1f7; }
table.mg-table tr:last-child td { border-bottom: none; }
table.mg-table td.pos { color: var(--green); font-weight: 600; }
table.mg-table td.neg { color: var(--red); font-weight: 600; }
table.mg-table td.gld { color: var(--gold); font-weight: 600; }

/* Info / warning boxes */
.info-box {
  background: #eef4ff; border: 1px solid #b8d0f8; color: #1a4a9c;
  border-radius: 5px; padding: 10px 14px; font-size: 12px; margin-bottom: 12px;
}
.warn-box {
  background: #fff8e8; border: 1px solid #f0d070; color: #7a5000;
  border-radius: 5px; padding: 10px 14px; font-size: 12px; margin-bottom: 12px;
}

/* Trim toggle area */
.trim-area {
  display: flex; align-items: center; gap: 12px;
  font-family: var(--ui); font-size: 12px; color: var(--text);
  background: #fff; border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 14px; margin-bottom: 16px;
}

/* Outlier highlight */
.outlier-badge {
  background: #fff8e8; border: 1px solid #f0d070;
  color: var(--gold); font-family: var(--mono); font-size: 10px;
  padding: 1px 6px; border-radius: 3px; font-weight: 600;
}

/* Section note */
.section-note {
  font-family: var(--ui); font-size: 11px; color: var(--muted);
  margin-top: 6px;
}

@media (max-width: 900px) {
  .stat-banner { grid-template-columns: repeat(3, 1fr); }
}
</style>
""", unsafe_allow_html=True)


# ─── PASSWORD GATE ────────────────────────────────────────────────────────────
# Password is stored in Streamlit Cloud "Secrets" as:  password = "yourpassword"
# Locally, create a file .streamlit/secrets.toml with the same line.

def _check_password() -> bool:
    """Returns True once the correct password has been entered."""
    import hmac

    def _submit():
        entered = st.session_state.get("pw_input", "")
        correct = st.secrets.get("password", "")
        if hmac.compare_digest(entered, correct):
            st.session_state["authenticated"] = True
        else:
            st.session_state["pw_wrong"] = True

    if st.session_state.get("authenticated"):
        return True

    # Center the login box
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div style="text-align:center; padding: 40px 0 20px;">
          <div style="font-family:'Gill Sans','Optima','Segoe UI',sans-serif;
                      font-size:28px; font-weight:700; letter-spacing:1px;
                      color:#1a2332;">MEATGRINDER</div>
          <div style="font-family:'Menlo','Consolas',monospace; font-size:11px;
                      color:#7a8ea8; margin-top:6px;">
                      Performance Analytics Engine</div>
        </div>
        """, unsafe_allow_html=True)
        st.text_input("Password", type="password", key="pw_input",
                      on_change=_submit, label_visibility="collapsed",
                      placeholder="Enter password…")
        if st.session_state.get("pw_wrong"):
            st.error("Incorrect password.")
            st.session_state["pw_wrong"] = False
    return False

if not _check_password():
    st.stop()


# ─── SESSION STATE ────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        'fund_df': None,
        'fund_name': 'Fund',
        'outliers_df': None,
        'trimmed': False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 MEATGRINDER")
    st.markdown("*Universal Performance Analytics*")
    st.markdown("---")

    st.markdown("### Load Fund Data")

    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=['csv', 'xlsx', 'xls'],
        help="Date column + monthly return column. Percent or decimal format auto-detected."
    )

    if uploaded is not None:
        df, err, diag = parse_uploaded_file(uploaded)
        if err:
            st.error(err)
            if diag:
                with st.expander("Parse diagnostics"):
                    st.json(diag)
        else:
            st.session_state['fund_df'] = df
            st.session_state['outliers_df'] = compute_outliers(df)
            st.session_state['parse_diag'] = diag
            st.success(f"✓ {len(df)} monthly observations loaded")
            # Show staging summary
            st.markdown(
                f"<small style='color:#7a8ea8;'>"
                f"Date col: <b>{diag.get('date_col','?')}</b> · "
                f"Format: <b>{diag.get('date_format','?')}</b><br>"
                f"Return col: <b>{diag.get('ret_col','?')}</b> · "
                f"Scale: <b>{diag.get('ret_format','?')}</b><br>"
                f"Parsed: <b>{diag.get('n_parsed','?')}</b> rows · "
                f"Dropped: <b>{diag.get('n_dropped',0)}</b>"
                + (f"<br>⚠ Failed dates: {diag['failed_dates'][:5]}"
                   if diag.get('failed_dates') else "")
                + "</small>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("### Settings")

    fund_name = st.text_input("Fund Name", value=st.session_state.get('fund_name', 'Fund'))
    st.session_state['fund_name'] = fund_name

    bm_choice = st.selectbox(
        "Primary Benchmark",
        ['MSCI World Hedged USD', 'Bloomberg Global Agg', 'None']
    )

    trimmed = st.toggle(
        "Exclude ≥3σ outlier months",
        value=st.session_state['trimmed']
    )
    st.session_state['trimmed'] = trimmed

    st.markdown("---")
    st.markdown(
        "<small style='color:#7a8ea8;'>Benchmarks: MSCI World Hdg USD<br>"
        "Bloomberg Global Agg<br>Rf: TB3MS (FRED) Jul 2005–Dec 2025</small>",
        unsafe_allow_html=True
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────

fund_df_raw = st.session_state['fund_df']
outliers_df = st.session_state['outliers_df']

# Apply trim filter
if fund_df_raw is not None and trimmed and outliers_df is not None and len(outliers_df) > 0:
    out_set = set(zip(outliers_df['year'], outliers_df['month']))
    fund_df = fund_df_raw[
        ~fund_df_raw.apply(lambda r: (r['year'], r['month']) in out_set, axis=1)
    ].copy()
else:
    fund_df = fund_df_raw

# Select benchmark DataFrames
if bm_choice == 'MSCI World Hedged USD':
    bm1_df = MSCI_DF
    bm1_name = 'MSCI World Hdg'
    bm2_df = AGG_DF
    bm2_name = 'Blmbrg Agg'
elif bm_choice == 'Bloomberg Global Agg':
    bm1_df = AGG_DF
    bm1_name = 'Bloomberg Agg'
    bm2_df = MSCI_DF
    bm2_name = 'MSCI World Hdg'
else:
    bm1_df = None
    bm1_name = None
    bm2_df = None
    bm2_name = None


# ─── UPLOAD PROMPT ────────────────────────────────────────────────────────────

if fund_df is None:
    st.markdown("""
    <div class="mg-header">
      <h1>MEATGRINDER</h1>
      <div class="sub">Universal Performance Analytics Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="mg-card">
    <div class="section-title">Getting Started</div>
    <p style="font-size:13px; color:#1a2332; margin:0 0 10px;">
    Upload a CSV or Excel file with monthly returns using the sidebar.
    </p>
    <ul style="font-size:12px; color:#7a8ea8; line-height:1.8;">
      <li>First column: date (any standard format)</li>
      <li>Second column: monthly returns (% or decimal — auto-detected)</li>
      <li>Headers optional; first numeric non-date column is used</li>
      <li>Benchmarks (MSCI World Hdg, Bloomberg Agg) and TB3MS risk-free rates are built-in</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── DASHBOARD ────────────────────────────────────────────────────────────────

rets = fund_df['ret'].values
n_obs = len(rets)
n_out = len(outliers_df) if outliers_df is not None else 0

# Build date range string
first = fund_df.iloc[0]
last  = fund_df.iloc[-1]
date_range = f"{MN[int(first['month'])-1]} {int(first['year'])} – {MN[int(last['month'])-1]} {int(last['year'])}"
trim_note  = f" · {'✂ ' if trimmed else ''}{n_obs} obs" + (f" ({n_out} outlier(s) excluded)" if trimmed else f" · {n_out} outlier(s) detected")

# Header
bm_note = f"· Benchmarks: {bm1_name or 'None'}" if bm1_df is not None else ""
st.markdown(f"""
<div class="mg-header">
  <h1>PERFORMANCE ANALYTICS{"&nbsp;<span style='font-size:12px;background:#f0f5ff;color:#1a6cf5;border:1px solid #c0d4f8;padding:2px 8px;border-radius:3px;vertical-align:middle;'>{'✂ TRIMMED' if trimmed else ''}</span>" if trimmed else ""}</h1>
  <div class="sub">
    {st.session_state['fund_name'].upper()} &nbsp;·&nbsp; {date_range} &nbsp;·&nbsp; N = {n_obs} Monthly Observations{' ' + bm_note if bm_note else ''} &nbsp;·&nbsp; Rf = TB3MS (FRED)
  </div>
</div>
""", unsafe_allow_html=True)

# Outlier info box
if n_out > 0 and outliers_df is not None:
    out_list = ', '.join(
        f"{MN[int(r.month)-1]} {int(r.year)} ({r.ret:+.1f}%)"
        for r in outliers_df.itertuples()
    )
    st.markdown(
        f'<div class="info-box">⚡ <strong>Outlier Alert:</strong> '
        f'{n_out} month{"s" if n_out > 1 else ""} detected ≥3σ from the mean: '
        f'<span style="color:#c47d0a;font-weight:600">{out_list}</span>. '
        f'Toggle "Exclude ≥3σ outlier months" in the sidebar to see trimmed analytics.</div>',
        unsafe_allow_html=True
    )


# ─── STAT BANNER ──────────────────────────────────────────────────────────────

ar  = geo_return(rets)
av  = ann_vol(rets)
sh  = sharpe_with_rf(fund_df)
md  = max_drawdown(rets)
sk  = skewness(rets)
ek  = excess_kurtosis(rets)

def _cls(v, flip=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    return ('neg' if flip else 'pos') if v > 0 else ('pos' if flip else 'neg')

def _fmt(v, dec=2, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f"{v:.{dec}f}{'%' if pct else ''}"

st.markdown(f"""
<div class="stat-banner">
  <div class="stat-cell"><div class="lbl">Ann. Return</div><div class="val {_cls(ar)}">{_fmt(ar)}</div></div>
  <div class="stat-cell"><div class="lbl">Ann. Volatility</div><div class="val ac">{_fmt(av)}</div></div>
  <div class="stat-cell"><div class="lbl">Sharpe (Rf=TB3MS)</div><div class="val ac">{_fmt(sh, 3, False)}</div></div>
  <div class="stat-cell"><div class="lbl">Max Drawdown</div><div class="val neg">{_fmt(md)}</div></div>
  <div class="stat-cell"><div class="lbl">Skewness</div><div class="val gld">{_fmt(sk, 3, False)}</div></div>
  <div class="stat-cell"><div class="lbl">Ex. Kurtosis</div><div class="val gld">{_fmt(ek, 3, False)}</div></div>
</div>
""", unsafe_allow_html=True)


# ─── TABS ─────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Summary", "Calendar", "Drawdowns", "Distribution",
    "Regression", "Rolling", "Seasonality", "Multi-Period",
    "Macro Events", "Data"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<div class="section-title">Full vs. Ex-Outliers Comparison</div>', unsafe_allow_html=True)

        full_rets = fund_df_raw['ret'].values
        trim_rets = rets  # already filtered if trimmed toggle on

        # When toggle is OFF we still show both columns; when ON the "full" is same as trimmed
        full_df_for_stats  = fund_df_raw
        trim_df_for_stats  = fund_df

        jb_full, _ = jarque_bera(full_rets)
        jb_trim, _ = jarque_bera(trim_rets)

        def _v(val, dec=2, pct=True):
            return _fmt(val, dec, pct)

        trim_n   = len(trim_rets)
        trim_hdr = f"Ex Outliers (N={trim_n})"

        rows = [
            ("Observations",        len(full_rets),                       trim_n,                                      False, 0,  False),
            ("Ann. Return",         geo_return(full_rets),                 geo_return(trim_rets),                       True,  2,  False),
            ("Ann. Volatility",     ann_vol(full_rets),                    ann_vol(trim_rets),                          True,  2,  False),
            ("Sharpe (Rf=TB3MS)",   sharpe_with_rf(fund_df_raw),           sharpe_with_rf(trim_df_for_stats),           False, 3,  False),
            ("Sortino (Rf=TB3MS)",  sortino_with_rf(fund_df_raw),          sortino_with_rf(trim_df_for_stats),          False, 3,  False),
            ("Calmar Ratio",        calmar(full_rets),                     calmar(trim_rets),                           False, 3,  False),
            ("Max Drawdown",        max_drawdown(full_rets),               max_drawdown(trim_rets),                     True,  2,  True),
            ("Skewness",            skewness(full_rets),                   skewness(trim_rets),                         False, 4,  False),
            ("Excess Kurtosis",     excess_kurtosis(full_rets),            excess_kurtosis(trim_rets),                  False, 4,  False),
            ("Jarque-Bera Stat",    jb_full,                               jb_trim,                                     False, 2,  False),
            ("Best Month",          float(np.max(full_rets)),              float(np.max(trim_rets)),                    True,  2,  False),
            ("Worst Month",         float(np.min(full_rets)),              float(np.min(trim_rets)),                    True,  2,  True),
            ("Avg Monthly Return",  float(np.mean(full_rets)),             float(np.mean(trim_rets)),                   True,  3,  False),
            ("Hit Rate (%)",        float(np.sum(full_rets>0)/len(full_rets)*100),
                                    float(np.sum(trim_rets>0)/len(trim_rets)*100) if len(trim_rets)>0 else 0, False, 1, False),
            ("# Positive Months",   int(np.sum(full_rets > 0)),            int(np.sum(trim_rets > 0)),                  False, 0,  False),
            ("# Negative Months",   int(np.sum(full_rets < 0)),            int(np.sum(trim_rets < 0)),                  False, 0,  False),
        ]

        html = f"""
        <table class="mg-table">
          <thead><tr><th>Metric</th><th>Full (N={len(full_rets)})</th><th>{trim_hdr}</th></tr></thead>
          <tbody>
        """
        for (lbl, fv, tv, pct, dec, flip) in rows:
            def cell(v):
                if isinstance(v, int) and dec == 0:
                    return str(v)
                try:
                    cls = _cls(float(v), flip) if (not isinstance(v, int) or dec > 0) else ''
                    return f'<td class="{cls}">{_fmt(float(v), dec, pct)}</td>'
                except:
                    return f'<td>—</td>'
            html += f"<tr><td>{lbl}</td>{cell(fv)}{cell(tv)}</tr>"
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)

    with col_r:
        if bm1_df is not None:
            st.markdown('<div class="section-title">Benchmark Comparison</div>', unsafe_allow_html=True)

            # Align fund and benchmark on common months
            fund_bm_merged = fund_df_raw.merge(bm1_df, on=['year','month'], suffixes=('','_bm'))
            bm1_aligned = fund_bm_merged.rename(columns={'ret_bm': 'ret'})[['year','month','ret']]

            bm2_aligned = None
            if bm2_df is not None:
                fund_bm2_merged = fund_df_raw.merge(bm2_df, on=['year','month'], suffixes=('','_bm'))
                bm2_aligned = fund_bm2_merged.rename(columns={'ret_bm': 'ret'})[['year','month','ret']]

            bm_rows = [
                ("Ann. Return",    True,  2, False),
                ("Ann. Volatility",True,  2, False),
                ("Sharpe (Rf=TB3MS)",False,3, False),
                ("Sortino (Rf=TB3MS)",False,3,False),
                ("Max Drawdown",   True,  2, True),
                ("Calmar Ratio",   False, 3, False),
                ("Skewness",       False, 4, False),
                ("Excess Kurtosis",False, 4, False),
            ]

            def bm_val(stat_name, df):
                r = df['ret'].values
                lookup = {
                    "Ann. Return": geo_return(r),
                    "Ann. Volatility": ann_vol(r),
                    "Sharpe (Rf=TB3MS)": sharpe_with_rf(df),
                    "Sortino (Rf=TB3MS)": sortino_with_rf(df),
                    "Max Drawdown": max_drawdown(r),
                    "Calmar Ratio": calmar(r),
                    "Skewness": skewness(r),
                    "Excess Kurtosis": excess_kurtosis(r),
                }
                return lookup.get(stat_name, np.nan)

            bm2_hdr = bm2_name if bm2_df is not None else ''
            html2 = f"""
            <table class="mg-table">
              <thead><tr><th>Metric</th><th>{fund_name}</th><th>{bm1_name}</th>
              {'<th>' + bm2_hdr + '</th>' if bm2_df is not None else ''}</tr></thead>
              <tbody>
            """
            for (lbl, pct, dec, flip) in bm_rows:
                fv = bm_val(lbl, fund_df_raw)
                b1v = bm_val(lbl, bm1_aligned)
                html2 += f"<tr><td>{lbl}</td>"
                for v in ([fv, b1v] + ([bm_val(lbl, bm2_aligned)] if bm2_aligned is not None else [])):
                    try:
                        c = _cls(float(v), flip)
                        html2 += f'<td class="{c}">{_fmt(float(v), dec, pct)}</td>'
                    except:
                        html2 += '<td>—</td>'
                html2 += "</tr>"
            html2 += "</tbody></table>"
            st.markdown(html2, unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-title">Cumulative Performance</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:20px;">Cumulative Wealth (Growth of $100)</div>', unsafe_allow_html=True)
        fig_cum = chart_cumulative(fund_df, fund_name, bm1_df, bm1_name or '', bm2_df, bm2_name or '')
        st.plotly_chart(fig_cum, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CALENDAR HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown('<div class="section-title">Monthly Return Calendar</div>', unsafe_allow_html=True)
    fig_cal = chart_calendar_heatmap(fund_df)
    st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:16px;">Monthly Return Bars</div>', unsafe_allow_html=True)
    fig_bars = chart_monthly_bars(fund_df)
    st.plotly_chart(fig_bars, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DRAWDOWNS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown('<div class="section-title">Drawdown Series</div>', unsafe_allow_html=True)
    fig_dd = chart_drawdowns(fund_df)
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:16px;">Top 7 Drawdown Episodes</div>', unsafe_allow_html=True)
    episodes = top_drawdowns(fund_df)

    html_ep = """
    <table class="mg-table">
      <thead><tr>
        <th>#</th><th>Drawdown</th><th>Peak</th><th>Trough</th><th>Recovery</th>
        <th>Peak→Trough</th><th>Trough→Recovery</th><th>Total</th>
      </tr></thead><tbody>
    """
    for i, ep in enumerate(episodes):
        dd_str = _fmt(ep['drawdown'])
        pt_str = str(ep['peak_to_trough']) + 'm'
        tr_str = str(ep['trough_to_recovery']) + 'm' if ep['trough_to_recovery'] is not None else 'Ongoing'
        tot_str = str(ep['total_months']) + 'm' if ep['total_months'] is not None else 'Ongoing'
        html_ep += (
            f"<tr>"
            f"<td>{i+1}</td>"
            f"<td class='neg'>{dd_str}</td>"
            f"<td>{ep['peak_date']}</td>"
            f"<td>{ep['trough_date']}</td>"
            f"<td>{ep['recovery_date']}</td>"
            f"<td>{pt_str}</td>"
            f"<td>{tr_str}</td>"
            f"<td>{tot_str}</td>"
            f"</tr>"
        )
    html_ep += "</tbody></table>"
    st.markdown(html_ep, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    col_h, col_q = st.columns([3, 2])

    with col_h:
        st.markdown('<div class="section-title">Return Distribution</div>', unsafe_allow_html=True)
        fig_hist = chart_histogram(fund_df, outliers_df if not trimmed else None)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_q:
        st.markdown('<div class="section-title">Q-Q Plot vs Normal</div>', unsafe_allow_html=True)
        fig_qq = chart_qq(fund_df)
        st.plotly_chart(fig_qq, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:4px;">Normality Tests</div>', unsafe_allow_html=True)

    norm = normality_tests(rets)
    html_norm = """
    <table class="mg-table">
      <thead><tr><th>Test</th><th>Statistic</th><th>p-value / Critical</th><th>Reject H₀ Normality?</th></tr></thead>
      <tbody>
    """
    # Jarque-Bera
    jb = norm['jb']
    jb_reject = jb['pval'] < 0.05
    html_norm += (f"<tr><td>Jarque-Bera</td><td class='gld'>{jb['stat']:.4f}</td>"
                  f"<td>{jb['pval']:.4f}</td>"
                  f"<td class='{'neg' if jb_reject else 'green'}'>{'Yes ✗' if jb_reject else 'No ✓'}</td></tr>")

    if 'sw' in norm:
        sw = norm['sw']
        sw_reject = sw['pval'] < 0.05
        html_norm += (f"<tr><td>Shapiro-Wilk</td><td class='gld'>{sw['stat']:.4f}</td>"
                      f"<td>{sw['pval']:.4f}</td>"
                      f"<td class='{'neg' if sw_reject else 'green'}'>{'Yes ✗' if sw_reject else 'No ✓'}</td></tr>")

    ad = norm['ad']
    html_norm += (f"<tr><td>Anderson-Darling</td><td class='gld'>{ad['stat']:.4f}</td>"
                  f"<td>Crit(5%)={ad['critical']:.4f}</td>"
                  f"<td class='{'neg' if ad['reject'] else 'green'}'>{'Yes ✗' if ad['reject'] else 'No ✓'}</td></tr>")

    ks = norm['ks']
    ks_reject = ks['pval'] < 0.05
    html_norm += (f"<tr><td>Kolmogorov-Smirnov</td><td class='gld'>{ks['stat']:.4f}</td>"
                  f"<td>{ks['pval']:.4f}</td>"
                  f"<td class='{'neg' if ks_reject else 'green'}'>{'Yes ✗' if ks_reject else 'No ✓'}</td></tr>")

    html_norm += "</tbody></table>"
    st.markdown(html_norm, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:16px;">Return Autocorrelation (Lags 1–12)</div>', unsafe_allow_html=True)
    fig_acf = chart_acf(fund_df)
    st.plotly_chart(fig_acf, use_container_width=True)
    conf_b = acf_conf_band(len(rets))
    st.markdown(
        f'<div class="section-note">95% confidence band: ±{conf_b:.4f}. '
        f'Bars exceeding the band (blue = sig, grey = insig) indicate serial correlation.</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    if bm1_df is None:
        st.info("Select a benchmark in the sidebar to view regression analysis.")
    else:
        reg = piecewise_beta_regression(
            fund_df.rename(columns={'ret':'ret'}),
            bm1_df.rename(columns={'ret':'ret'})
        )

        if reg is None:
            st.warning("Insufficient overlapping data for regression.")
        else:
            col_reg_l, col_reg_r = st.columns([3, 2])

            with col_reg_l:
                st.markdown(f'<div class="section-title">Fund vs {bm1_name}: Scatter & Regression</div>', unsafe_allow_html=True)
                fig_reg = chart_regression(reg, fund_name, bm1_name)
                st.plotly_chart(fig_reg, use_container_width=True)

            with col_reg_r:
                st.markdown('<div class="section-title">Regression Statistics</div>', unsafe_allow_html=True)

                def pval_from_t(t, df):
                    if np.isnan(t):
                        return np.nan
                    from scipy.stats import t as t_dist
                    return float(2 * t_dist.sf(abs(t), df=df))

                n_reg = reg['n']
                alpha_p = pval_from_t(reg['t_alpha'], n_reg - 2)
                beta_p  = pval_from_t(reg['t_beta'],  n_reg - 2)

                reg_table = [
                    ("N (overlapping)", n_reg, False),
                    ("Alpha (monthly %)", f"{reg['alpha']:.4f}%", False),
                    ("  SE(Alpha)",      f"{reg['se_alpha']:.4f}" if not np.isnan(reg['se_alpha']) else '—', False),
                    ("  t(Alpha)",       f"{reg['t_alpha']:.3f}" if not np.isnan(reg['t_alpha']) else '—', False),
                    ("  p(Alpha)",       f"{alpha_p:.4f}" if not np.isnan(alpha_p) else '—', False),
                    ("  95% CI Alpha",   f"[{reg['alpha_ci'][0]:.3f}, {reg['alpha_ci'][1]:.3f}]"
                                        if not np.isnan(reg['alpha_ci'][0]) else '—', False),
                    ("Beta (full)",      f"{reg['beta']:.4f}", False),
                    ("  SE(Beta)",       f"{reg['se_beta']:.4f}" if not np.isnan(reg['se_beta']) else '—', False),
                    ("  t(Beta)",        f"{reg['t_beta']:.3f}" if not np.isnan(reg['t_beta']) else '—', False),
                    ("  p(Beta)",        f"{beta_p:.4f}" if not np.isnan(beta_p) else '—', False),
                    ("  95% CI Beta",    f"[{reg['beta_ci'][0]:.3f}, {reg['beta_ci'][1]:.3f}]"
                                        if not np.isnan(reg['beta_ci'][0]) else '—', False),
                    ("R²",              f"{reg['r2']:.4f}", False),
                    ("Correlation",     f"{reg['corr']:.4f}", False),
                    ("β⁺ (BM Up months)", f"{reg['beta_up']:.4f}" if reg['beta_up'] is not None else '—', False),
                    (f"  N Up",          f"{reg['n_up']}", False),
                    ("β⁻ (BM Down months)", f"{reg['beta_dn']:.4f}" if reg['beta_dn'] is not None else '—', False),
                    (f"  N Down",        f"{reg['n_dn']}", False),
                    ("Convexity (β⁺ − β⁻)", f"{reg['convexity']:.4f}" if reg['convexity'] is not None else '—', False),
                ]

                html_reg = "<table class='mg-table'><thead><tr><th>Statistic</th><th>Value</th></tr></thead><tbody>"
                for (lbl, val, is_pct) in reg_table:
                    lbl_style = "font-weight:400;color:var(--muted);" if lbl.startswith("  ") else ""
                    html_reg += f"<tr><td style='{lbl_style}'>{lbl}</td><td style='font-family:var(--mono)'>{val}</td></tr>"
                html_reg += "</tbody></table>"
                st.markdown(html_reg, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ROLLING
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    if len(fund_df) < 12:
        st.warning("Need at least 12 months for rolling metrics.")
    else:
        st.markdown('<div class="section-title">Rolling 12-Month Return</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_rolling(fund_df, 'roll_ret'), use_container_width=True)

        col_rs, col_rv = st.columns(2)
        with col_rs:
            st.markdown('<div class="section-title">Rolling 12-Month Sharpe</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_rolling(fund_df, 'roll_sharpe'), use_container_width=True)
        with col_rv:
            st.markdown('<div class="section-title">Rolling 12-Month Volatility</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_rolling(fund_df, 'roll_vol'), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SEASONALITY
# ══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    seas = seasonality(fund_df)
    col_sm, col_sq = st.columns(2)
    with col_sm:
        st.markdown('<div class="section-title">Average Return by Month</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_seasonality_monthly(seas), use_container_width=True)
    with col_sq:
        st.markdown('<div class="section-title">Average Return by Quarter</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_seasonality_quarterly(seas), use_container_width=True)

    # Monthly detail table
    st.markdown('<div class="section-title" style="margin-top:12px;">Monthly Seasonality Detail</div>', unsafe_allow_html=True)
    html_seas = """
    <table class="mg-table">
      <thead><tr><th>Month</th><th>Avg Return</th><th>Std Dev</th><th>N</th><th>Hit Rate</th></tr></thead>
      <tbody>
    """
    for _, row in seas['monthly'].iterrows():
        m_idx = int(row['month'])
        m_name = MN[m_idx - 1]
        sub = fund_df[fund_df['month'] == m_idx]['ret']
        hit = int(np.sum(sub > 0)) / len(sub) * 100 if len(sub) > 0 else 0
        cls = 'pos' if row['mean'] >= 0 else 'neg'
        html_seas += (
            f"<tr><td>{m_name}</td>"
            f"<td class='{cls}'>{row['mean']:.2f}%</td>"
            f"<td>{row['std']:.2f}%</td>"
            f"<td>{int(row['count'])}</td>"
            f"<td>{hit:.0f}%</td></tr>"
        )
    html_seas += "</tbody></table>"
    st.markdown(html_seas, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — MULTI-PERIOD
# ══════════════════════════════════════════════════════════════════════════════

with tabs[7]:
    # Determine last year/month in data
    last_year  = int(fund_df.iloc[-1]['year'])
    last_month = int(fund_df.iloc[-1]['month'])

    def slice_from(start_year, start_month):
        return fund_df[(fund_df['year'] > start_year) |
                       ((fund_df['year'] == start_year) & (fund_df['month'] >= start_month))].copy()

    periods = [
        ('Since Inception', fund_df.copy()),
        ('5 Year',  slice_from(last_year - 5, last_month)),
        ('3 Year',  slice_from(last_year - 3, last_month)),
        ('1 Year',  slice_from(last_year - 1, last_month)),
    ]

    metrics_def = [
        ('Ann. Return (%)',    'ann_ret',  True),
        ('Ann. Volatility (%)', 'ann_vol', True),
        ('Sharpe (Rf=TB3MS)',  'sharpe',   False),
        ('Sortino (Rf=TB3MS)', 'sortino',  False),
        ('Calmar Ratio',       'calmar',   False),
        ('Max Drawdown (%)',   'max_dd',   True),
        ('Hit Rate (%)',       'hit_rate', False),
        ('Best Month (%)',     'best',     True),
        ('Worst Month (%)',    'worst',    True),
        ('Skewness',           'skew',     False),
        ('Excess Kurtosis',    'exkurt',   False),
        ('N Observations',     'n',        False),
    ]

    period_stats_list = [(lbl, period_stats(df)) for lbl, df in periods]

    html_mp = "<table class='mg-table'><thead><tr><th>Metric</th>"
    for lbl, _ in period_stats_list:
        html_mp += f"<th>{lbl}</th>"
    html_mp += "</tr></thead><tbody>"

    flip_metrics = {'max_dd', 'worst'}

    for (m_lbl, m_key, pct) in metrics_def:
        html_mp += f"<tr><td>{m_lbl}</td>"
        for _, ps in period_stats_list:
            if ps is None:
                html_mp += "<td>—</td>"
                continue
            v = ps.get(m_key, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                html_mp += "<td>—</td>"
                continue
            if m_key == 'n':
                html_mp += f"<td>{int(v)}</td>"
                continue
            flip = m_key in flip_metrics
            cls  = _cls(float(v), flip)
            dec  = 0 if m_key == 'n' else (2 if pct else 3)
            html_mp += f"<td class='{cls}'>{_fmt(float(v), dec, pct)}</td>"
        html_mp += "</tr>"

    html_mp += "</tbody></table>"
    st.markdown(html_mp, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — MACRO EVENTS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[8]:
    if bm1_df is None:
        st.info("Select a benchmark in the sidebar to see fund vs benchmark performance during macro events.")

    ev_rows = macro_events_table(fund_df, bm1_df)

    if not ev_rows:
        st.warning("No macro event periods overlap with the fund's return history.")
    else:
        html_ev = f"""
        <table class="mg-table">
          <thead><tr>
            <th>Event</th><th>Period</th>
            <th>{fund_name}</th>
            {'<th>' + (bm1_name or '') + '</th><th>Spread</th><th>Protected?</th>' if bm1_df is not None else ''}
          </tr></thead><tbody>
        """
        for ev in ev_rows:
            f_cls = 'pos' if ev['fund_ret'] >= 0 else 'neg'
            html_ev += f"<tr><td><strong>{ev['name']}</strong></td><td style='color:var(--muted)'>{ev['period']}</td>"
            html_ev += f"<td class='{f_cls}'>{_fmt(ev['fund_ret'])}</td>"
            if ev['bm_ret'] is not None:
                b_cls = 'pos' if ev['bm_ret'] >= 0 else 'neg'
                sp_cls = 'pos' if ev['spread'] >= 0 else 'neg'
                prot = '<span style="color:var(--green);font-weight:600">✓ Yes</span>' if ev['spread'] > 0 else '<span style="color:var(--red);">✗ No</span>'
                html_ev += (f"<td class='{b_cls}'>{_fmt(ev['bm_ret'])}</td>"
                            f"<td class='{sp_cls}'>{'+' if ev['spread']>=0 else ''}{_fmt(ev['spread'], 2, False)} pp</td>"
                            f"<td>{prot}</td>")
            html_ev += "</tr>"
        html_ev += "</tbody></table>"
        st.markdown(html_ev, unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">Returns are compound for multi-month events. '
            '"Protected?" = fund spread vs benchmark is positive.</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — DATA
# ══════════════════════════════════════════════════════════════════════════════

with tabs[9]:
    st.markdown('<div class="section-title">Raw Monthly Data</div>', unsafe_allow_html=True)

    # Search filter
    search = st.text_input("Filter by year or month", placeholder="e.g. 2020 or Mar")

    display_df = fund_df_raw.copy()
    display_df['date'] = display_df.apply(
        lambda r: f"{MN[int(r['month'])-1]} {int(r['year'])}", axis=1
    )

    if outliers_df is not None:
        out_set = set(zip(outliers_df['year'], outliers_df['month']))
        display_df['flag'] = display_df.apply(
            lambda r: '⭐ Outlier' if (r['year'], r['month']) in out_set else '', axis=1
        )
    else:
        display_df['flag'] = ''

    if search:
        mask = (
            display_df['date'].str.lower().str.contains(search.lower()) |
            display_df['ret'].astype(str).str.contains(search)
        )
        display_df = display_df[mask]

    html_data = f"""
    <table class="mg-table">
      <thead><tr><th>Date</th><th class="ch">{fund_name} (%)</th><th>Flag</th></tr></thead>
      <tbody>
    """
    for r in display_df.itertuples():
        out_style = 'background:#fff8e8;' if r.flag else ''
        cls = 'pos' if r.ret >= 0 else 'neg'
        html_data += (
            f"<tr style='{out_style}'>"
            f"<td>{r.date}</td>"
            f"<td class='{cls}'>{_fmt(r.ret, 2, True)}</td>"
            f"<td><span class='outlier-badge'>{r.flag}</span></td>"
            f"</tr>"
        )
    html_data += f"</tbody></table>"
    st.markdown(html_data, unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-note">Showing {len(display_df)} of {len(fund_df_raw)} rows. '
        f'Gold rows = ≥3σ outlier months.</div>',
        unsafe_allow_html=True
    )
