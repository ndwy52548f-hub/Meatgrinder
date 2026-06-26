"""Argus PDF report generator.

Renders a print-tuned, multi-page US-Letter report for a single loaded deck:
cover + deck terms, executive summary, performance / risk / market / crisis
visuals and tables, methodology, and the full DDQ. Charts are static matplotlib
re-creations of the on-screen Plotly figures (white background, same palette),
sized for 8.5x11 paper. Pure-pip dependencies only (reportlab, matplotlib).
"""

import io
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors as rlc
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether)

from analytics import (
    geo_return, ann_vol, sharpe_with_rf, sortino_with_rf, calmar, max_drawdown,
    drawdown_series, skewness, excess_kurtosis, jarque_bera, top_drawdowns,
    piecewise_beta_regression, macro_events_table, seasonality, rolling_metrics,
    compound_return, sharpe_significance, tail_risk, capture_ratios,
    desmooth_stats, build_exec_summary, build_ddq, MN,
)

# ── palette (matches the dashboard) ───────────────────────────────────────────
TEAL = '#006B7A'; GOLD = '#E8B23A'; SKY = '#5B9BD5'; GREENc = '#43B581'
REDc = '#A93420'; AXIS = '#1A1A1A'; GRID = '#E5ECEC'; MUTED = '#8A9BA0'
PAGE_W, PAGE_H = letter
MARGIN = 0.6 * inch
USABLE_W = PAGE_W - 2 * MARGIN

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.edgecolor': AXIS, 'axes.labelcolor': AXIS,
    'xtick.color': AXIS, 'ytick.color': AXIS, 'text.color': AXIS,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def _wf_hex(a, b, t):
    t = max(0.0, min(1.0, t)); a = a.lstrip('#'); b = b.lstrip('#')
    ar, ag, ab = int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)
    br, bg, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
    return '#%02X%02X%02X' % (round(ar+(br-ar)*t), round(ag+(bg-ag)*t), round(ab+(bb-ab)*t))


def _sign_color(v, scale):
    if scale <= 0 or v == 0:
        return '#C9D1D6'
    t = min(abs(v) / scale, 1.0)
    return _wf_hex('#CFE0F0', '#2C5F8A', t) if v > 0 else _wf_hex('#F6C89A', REDc, t)


def _style(ax):
    ax.set_facecolor('white')
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def _pct(ax, axis='y'):
    fmt = lambda v, _: f"{v:.0f}%"
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(plt.FuncFormatter(fmt))


def _img(fig, width=USABLE_W):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); buf.seek(0)
    iw, ih = fig.get_size_inches()
    return Image(buf, width=width, height=width * ih / iw)


def _date_labels(df):
    return [f"{MN[int(r.month)-1]} {str(int(r.year))[2:]}" for r in df.itertuples()]


def _sparse_ticks(ax, labels, n=10):
    step = max(1, len(labels) // n)
    idx = list(range(0, len(labels), step))
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=0, fontsize=7)


# ── chart renderers ───────────────────────────────────────────────────────────

def fig_cumulative(fund_df, fund_name, series):
    fig, ax = plt.subplots(figsize=(7.3, 3.0))
    _style(ax)
    lab = _date_labels(fund_df)
    x = range(len(fund_df))
    ax.plot(x, 100 * np.cumprod(1 + fund_df['ret'].values / 100), color=TEAL, lw=2, label=fund_name, zorder=5)
    for nm, d, col in series:
        m = fund_df[['year', 'month']].merge(d[['year', 'month', 'ret']], on=['year', 'month'], how='left')
        r = m['ret'].fillna(0).values
        ax.plot(x, 100 * np.cumprod(1 + r / 100), color=col, lw=1.2, ls=':', label=nm)
    _sparse_ticks(ax, lab); ax.set_ylabel('Growth of $100')
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    return _img(fig)


def _heat_hex(v, cap, pos=(26, 138, 80), neg=(204, 34, 34)):
    if v is None:
        return '#FFFFFF'
    t = max(-1.0, min(1.0, v / cap))
    a = 0.10 + 0.50 * abs(t)
    base = pos if t >= 0 else neg
    return '#%02X%02X%02X' % tuple(round(255 * (1 - a) + base[i] * a) for i in range(3))


def calendar_table(fund_name, series):
    """Multi-series monthly return track record with YTD — mirrors the dashboard
    Calendar tab (green/red shading, year grouping)."""
    def lut(d):
        return {(int(r.year), int(r.month)): float(r.ret) for r in d.itertuples()}
    luts = [(nm, lut(d)) for nm, d in series if d is not None]
    years = sorted({int(r.year) for r in series[0][1].itertuples()})

    def ytd(vals):
        present = [v for v in vals if v is not None]
        if not present:
            return None
        w = 1.0
        for v in present:
            w *= (1 + v / 100)
        return (w - 1) * 100

    header = ['', 'Series'] + MN + ['YTD']
    data = [header]
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), rlc.HexColor(TEAL)),
        ('TEXTCOLOR', (0, 0), (-1, 0), rlc.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 6.5),
        ('FONTNAME', (2, 1), (-1, -1), 'Courier'),
        ('FONTNAME', (0, 1), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 6.3),
        ('TEXTCOLOR', (0, 1), (-1, -1), rlc.HexColor(AXIS)),
        ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 2), ('RIGHTPADDING', (0, 0), (-1, -1), 2),
    ]
    r = 0
    for yi, yr in enumerate(years):
        for si, (sname, L) in enumerate(luts):
            r += 1
            mvals = [L.get((yr, mo)) for mo in range(1, 13)]
            cells = [f"{v:+.2f}" if v is not None else '' for v in mvals]
            yv = ytd(mvals)
            row = [str(yr) if si == 0 else '', sname] + cells + [f"{yv:+.2f}" if yv is not None else '']
            data.append(row)
            for ci, v in enumerate(mvals):
                style.append(('BACKGROUND', (2 + ci, r), (2 + ci, r), rlc.HexColor(_heat_hex(v, 6.0))))
            style.append(('BACKGROUND', (14, r), (14, r), rlc.HexColor(_heat_hex(yv, 20.0))))
            style.append(('FONTNAME', (14, r), (14, r), 'Courier-Bold'))
            if si == 0:
                style.append(('FONTNAME', (0, r), (0, r), 'Helvetica-Bold'))
                if yi > 0:
                    style.append(('LINEABOVE', (0, r), (-1, r), 1.0, rlc.HexColor(TEAL)))
    name_w = 1.15 * inch
    yr_w = 0.30 * inch
    mw = (USABLE_W - name_w - yr_w) / 13
    t = Table(data, colWidths=[yr_w, name_w] + [mw] * 13, repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


def fig_calendar(fund_df):
    piv = fund_df.pivot_table(index='year', columns='month', values='ret')
    piv = piv.reindex(columns=range(1, 13))
    years = list(piv.index)
    fig, ax = plt.subplots(figsize=(7.3, max(1.4, 0.34 * len(years) + 0.6)))
    cmap = LinearSegmentedColormap.from_list('div', [REDc, '#F6C89A', '#FFFFFF', '#CFE0F0', '#2C5F8A'])
    vmax = np.nanmax(np.abs(piv.values)) or 1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    ax.imshow(piv.values, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(range(12)); ax.set_xticklabels(MN, fontsize=7.5)
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years, fontsize=7.5)
    ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(years), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5); ax.tick_params(which='minor', length=0)
    if len(years) <= 14:
        for i in range(len(years)):
            for j in range(12):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha='center', va='center', fontsize=6,
                            color='white' if abs(v) > vmax * 0.55 else AXIS)
    return _img(fig)


def fig_drawdowns(fund_df):
    fig, ax = plt.subplots(figsize=(7.3, 2.4)); _style(ax)
    dd = drawdown_series(fund_df['ret'].values)
    x = range(len(dd))
    ax.fill_between(x, dd, 0, color=REDc, alpha=0.16)
    ax.plot(x, dd, color=REDc, lw=1.3)
    _sparse_ticks(ax, _date_labels(fund_df)); _pct(ax); ax.set_ylabel('Drawdown')
    return _img(fig)


def fig_histogram(fund_df):
    r = fund_df['ret'].values
    fig, ax = plt.subplots(figsize=(7.3, 2.6)); _style(ax)
    counts, edges = np.histogram(r, bins=30)
    centers = (edges[:-1] + edges[1:]) / 2; w = (edges[-1] - edges[0]) / 30
    sc = np.nanmax(np.abs(centers)) or 1
    ax.bar(centers, counts, width=w, color=[_sign_color(c, sc) for c in centers], edgecolor='white', lw=0.4)
    from scipy.stats import norm
    xs = np.linspace(r.min(), r.max(), 200)
    ax.plot(xs, norm.pdf(xs, r.mean(), r.std(ddof=1)) * len(r) * w, color=AXIS, lw=1.2, ls=':')
    _pct(ax, 'x'); ax.set_xlabel('Monthly Return'); ax.set_ylabel('Frequency')
    return _img(fig)


def fig_regression(fund_df, series):
    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(7.3, 2.5))
    if n == 1:
        axes = [axes]
    for ax, (nm, d) in zip(axes, series):
        _style(ax)
        reg = piecewise_beta_regression(fund_df, d)
        if not reg:
            continue
        x, y = reg['x'], reg['y']
        up = x > 0
        ax.scatter(x[up], y[up], s=9, color=GREENc, alpha=0.65, edgecolors='none')
        ax.scatter(x[~up], y[~up], s=9, color=REDc, alpha=0.65, edgecolors='none')
        ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, reg['alpha'] + reg['beta'] * xs, color=AXIS, lw=1.4)
        a_pw = reg.get('alpha_pw', reg['alpha'])
        xu = np.linspace(0, x.max(), 30); xd = np.linspace(x.min(), 0, 30)
        ax.plot(xu, a_pw + reg['beta_up'] * xu, color=GREENc, lw=1.2, ls='--')
        ax.plot(xd, a_pw + reg['beta_dn'] * xd, color=REDc, lw=1.2, ls='--')
        ax.set_title(f"{nm}\n\u03b2={reg['beta']:.2f}  \u03b1={reg['alpha']:.2f}%  R\u00b2={reg['r2']:.2f}", fontsize=7.5)
        _pct(ax, 'x'); _pct(ax, 'y')
    fig.tight_layout()
    return _img(fig)


def _comovement_panel(ax, m, labels, series, title):
    _style(ax)
    widths = [0.78, 0.56, 0.40, 0.22][:len(series)]
    y = np.arange(len(labels))
    for (nm, vals, col), w in zip(series, widths):
        ax.barh(y, vals, height=w, color=col, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7); ax.invert_yaxis()
    ax.axvline(0, color=AXIS, lw=1); _pct(ax, 'x'); ax.set_title(title, fontsize=8.5, color=TEAL, weight='bold')


def fig_comovement(fund_df, mkt_df, mkt_name, alt_df, alt_name, bm3_df, bm3_name, n=5):
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0))
    base = fund_df[['year', 'month', 'ret']].rename(columns={'ret': 'fund'})
    base = base.merge(mkt_df[['year', 'month', 'ret']].rename(columns={'ret': 'mkt'}), on=['year', 'month'])
    base = base.merge(alt_df[['year', 'month', 'ret']].rename(columns={'ret': 'alt'}), on=['year', 'month'])
    if bm3_df is not None:
        base = base.merge(bm3_df[['year', 'month', 'ret']].rename(columns={'ret': 'bm3'}), on=['year', 'month'])
    for ax, worst, title in ((axes[0], True, f"Worst {n} Months"), (axes[1], False, f"Best {n} Months")):
        mm = base.sort_values('mkt', ascending=worst).head(n)
        labels = [f"{MN[int(r.month)-1]} {int(r.year)}" for r in mm.itertuples()]
        series = [(mkt_name, mm['mkt'].values, GOLD), (alt_name, mm['alt'].values, SKY)]
        if 'bm3' in mm:
            series.append((bm3_name, mm['bm3'].values, GREENc))
        series.append(('Strategy', mm['fund'].values, TEAL))
        _comovement_panel(ax, mm, labels, series, title)
    fig.tight_layout()
    return _img(fig)


def fig_updown(fund_df, mkt_df, others):
    base = fund_df[['year', 'month', 'ret']].rename(columns={'ret': 'fund'})
    base = base.merge(mkt_df[['year', 'month', 'ret']].rename(columns={'ret': 'mkt'}), on=['year', 'month'])
    up, dn = base['mkt'] > 0, base['mkt'] < 0
    rows = [('Strategy', base['fund'], TEAL), ('MSCI World Hdg', base['mkt'], GOLD)]
    cols = [SKY, GREENc]
    for i, (nm, d) in enumerate(others):
        mm = base[['year', 'month']].merge(d[['year', 'month', 'ret']], on=['year', 'month'], how='left')
        rows.append((nm, mm['ret'], cols[i % len(cols)]))
    fig, ax = plt.subplots(figsize=(7.3, 2.8)); _style(ax)
    x = np.arange(2); w = 0.8 / len(rows)
    for i, (nm, s, col) in enumerate(rows):
        ax.bar(x + i * w, [s[up].mean(), s[dn].mean()], width=w, label=nm, color=col, zorder=3)
    ax.set_xticks(x + w * (len(rows) - 1) / 2); ax.set_xticklabels(['Up Months', 'Down Months'])
    ax.axhline(0, color=AXIS, lw=1); _pct(ax); ax.set_ylabel('Avg Monthly Return')
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    return _img(fig)


def fig_waterfall(fund_df, mkt_df):
    m = fund_df[['year', 'month', 'ret']].rename(columns={'ret': 'fund'})
    m = m.merge(mkt_df[['year', 'month', 'ret']].rename(columns={'ret': 'mkt'}), on=['year', 'month'])
    m = m.sort_values('mkt', ascending=False).reset_index(drop=True)
    x = np.arange(len(m))
    fs = np.nanmax(np.abs(m['fund'])) or 1; ks = np.nanmax(np.abs(m['mkt'])) or 1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.3, 3.4), sharex=True)
    for ax in (a1, a2):
        _style(ax)
    a1.bar(x, m['fund'], color=[_sign_color(v, fs) for v in m['fund']], width=1.0)
    a2.bar(x, m['mkt'], color=[_sign_color(v, ks) for v in m['mkt']], width=1.0)
    a1.set_ylabel('Strategy'); a2.set_ylabel('MSCI World Hdg')
    a1.axhline(0, color=AXIS, lw=0.8); a2.axhline(0, color=AXIS, lw=0.8)
    _pct(a1); _pct(a2); a2.set_xticks([])
    fig.tight_layout()
    return _img(fig)


def fig_shocks(fund_df, mkt_df, alt_df, alt_name, bm3_df, bm3_name):
    evs = []
    for ev in macro_events_table.__globals__['MACRO_EVENTS']:
        months = ev['months']
        if not any(len(fund_df[(fund_df.year == y) & (fund_df.month == mo)]) for (y, mo) in months):
            continue
        f = compound_return(fund_df, months); k = compound_return(mkt_df, months)
        if f is None or k is None:
            continue
        ser = [('Strategy', f), ('MSCI World Hdg', k), (alt_name, compound_return(alt_df, months))]
        if bm3_df is not None:
            ser.append((bm3_name, compound_return(bm3_df, months)))
        evs.append({'name': ev['name'], 'period': ev['period'], 'diff': abs(f - k), 'ser': ser})
    evs.sort(key=lambda e: e['diff'], reverse=True); evs = evs[:4]
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 4.2))
    for ax, e in zip(axes.flat, evs):
        _style(ax)
        names = [n for n, v in e['ser'] if v is not None]
        vals = [v for n, v in e['ser'] if v is not None]
        sc = max((abs(v) for v in vals), default=1) or 1
        y = np.arange(len(names))
        ax.barh(y, vals, color=[_sign_color(v, sc) for v in vals], zorder=3)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7); ax.invert_yaxis()
        ax.axvline(0, color=AXIS, lw=0.8); _pct(ax, 'x')
        ax.set_title(f"{e['name']} · {e['period']}", fontsize=7.5, color=TEAL, weight='bold')
    for ax in axes.flat[len(evs):]:
        ax.set_visible(False)
    fig.tight_layout()
    return _img(fig)


def fig_rolling(fund_df, window=12):
    rm = rolling_metrics(fund_df, window)
    if rm.empty:
        return None
    lab = _date_labels(rm); x = range(len(rm))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.3, 3.4), sharex=True)
    for ax in (a1, a2):
        _style(ax)
    a1.plot(x, rm['roll_ret'], color=TEAL, lw=1.5); a1.set_ylabel(f'{window}m Return'); _pct(a1)
    a1.axhline(0, color=MUTED, lw=0.6)
    a2.plot(x, rm['roll_sharpe'], color=SKY, lw=1.5); a2.set_ylabel(f'{window}m Sharpe')
    a2.axhline(0, color=MUTED, lw=0.6)
    _sparse_ticks(a2, lab)
    fig.tight_layout()
    return _img(fig)


def fig_seasonality(fund_df):
    seas = seasonality(fund_df)
    mo = seas['monthly']; q = seas['quarterly']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.3, 2.4), gridspec_kw={'width_ratios': [3, 1]})
    for ax in (a1, a2):
        _style(ax)
    sc = np.nanmax(np.abs(mo['mean'])) or 1
    a1.bar(range(len(mo)), mo['mean'], color=[_sign_color(v, sc) for v in mo['mean']], zorder=3)
    a1.set_xticks(range(len(mo))); a1.set_xticklabels([MN[int(x)-1] for x in mo['month']], fontsize=7)
    a1.axhline(0, color=AXIS, lw=0.8); _pct(a1); a1.set_title('By Month', fontsize=8.5, color=TEAL, weight='bold')
    scq = np.nanmax(np.abs(q['mean'])) or 1
    a2.bar(range(len(q)), q['mean'], color=[_sign_color(v, scq) for v in q['mean']], zorder=3)
    a2.set_xticks(range(len(q))); a2.set_xticklabels([f"Q{int(x)}" for x in q['quarter']], fontsize=7)
    a2.axhline(0, color=AXIS, lw=0.8); _pct(a2); a2.set_title('By Quarter', fontsize=8.5, color=TEAL, weight='bold')
    fig.tight_layout()
    return _img(fig)


def fig_desmooth(fund_df, mkt_df):
    ds = desmooth_stats(fund_df, mkt_df)
    rep = fund_df.iloc[1:].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.3, 2.6)); _style(ax)
    x = range(len(rep))
    ax.plot(x, 100 * np.cumprod(1 + rep['ret'].values / 100), color=TEAL, lw=2, label='Reported')
    if 'unsmoothed_df' in ds:
        ax.plot(x, 100 * np.cumprod(1 + ds['unsmoothed_df']['ret'].values / 100),
                color=REDc, lw=1.5, ls=':', label='Unsmoothed')
    _sparse_ticks(ax, _date_labels(rep)); ax.set_ylabel('Growth of $100')
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    return _img(fig)


# ── reportlab styles & helpers ────────────────────────────────────────────────

_H = ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=12, textColor=rlc.HexColor(TEAL),
                    spaceBefore=12, spaceAfter=6, leading=15)
_SUB = ParagraphStyle('SUB', fontName='Helvetica-Bold', fontSize=9, textColor=rlc.HexColor(TEAL),
                      spaceBefore=8, spaceAfter=4)
_BODY = ParagraphStyle('BODY', fontName='Helvetica', fontSize=9.5, textColor=rlc.HexColor(AXIS),
                       leading=14, spaceAfter=6, alignment=TA_LEFT)
_Q = ParagraphStyle('Q', fontName='Helvetica', fontSize=9, textColor=rlc.HexColor(AXIS),
                    leading=13, spaceAfter=4, leftIndent=4)


def _kv_table(rows, widths=(USABLE_W * 0.62, USABLE_W * 0.38)):
    data = [[Paragraph(str(k), _BODY),
             Paragraph(f"<font name='Helvetica-Bold'>{v}</font>", _BODY)] for k, v in rows]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, -1), 0.4, rlc.HexColor(GRID)),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


def _grid_table(header, rows):
    data = [[Paragraph(f"<b>{h}</b>", _BODY) for h in header]]
    data += [[Paragraph(str(c), _BODY) for c in r] for r in rows]
    t = Table(data, colWidths=[USABLE_W / len(header)] * len(header))
    style = [('LINEBELOW', (0, 0), (-1, 0), 1.0, rlc.HexColor(TEAL)),
             ('LINEBELOW', (0, 1), (-1, -1), 0.4, rlc.HexColor(GRID)),
             ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    for c in range(1, len(header)):
        style.append(('ALIGN', (c, 0), (c, -1), 'RIGHT'))
    t.setStyle(TableStyle(style))
    return t


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.5); canvas.setFillColor(rlc.HexColor(MUTED))
    y = 0.42 * inch
    canvas.drawString(MARGIN, y, f"Generated by Argus \u00b7 {doc._argus_date}")
    canvas.drawCentredString(PAGE_W / 2, y, "Confidential \u2014 for the intended recipient only")
    canvas.drawRightString(PAGE_W - MARGIN, y, f"Page {doc.page}")
    canvas.restoreState()


def _cover(fund_name, span, gen_date, meta):
    band = Table([[Paragraph("<font color='white' size=18><b>ARGUS</b></font>", _BODY),
                   Paragraph(f"<font color='white' size=14><b>{fund_name}</b></font>", 
                             ParagraphStyle('r', parent=_BODY, alignment=2))]],
                 colWidths=[USABLE_W * 0.4, USABLE_W * 0.6])
    band.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), rlc.HexColor(TEAL)),
                              ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                              ('TOPPADDING', (0, 0), (-1, -1), 12), ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                              ('LEFTPADDING', (0, 0), (0, 0), 12), ('RIGHTPADDING', (-1, 0), (-1, 0), 12)]))
    out = [band, Spacer(1, 6),
           Paragraph(f"Performance &amp; Risk Report &nbsp;&middot;&nbsp; {span} &nbsp;&middot;&nbsp; "
                     f"Generated {gen_date}", _BODY), Spacer(1, 10)]
    terms = []
    if meta.get('portfolio_manager'):
        terms.append(("Portfolio Manager", meta['portfolio_manager']))
    if meta.get('strategy_type'):
        terms.append(("Strategy", ', '.join(meta['strategy_type'][:3])))
    for a in meta.get('aum', [])[:2]:
        terms.append(("AUM", a))
    if meta.get('management_fee'):
        terms.append(("Management Fee", meta['management_fee']))
    if meta.get('incentive_fee'):
        terms.append(("Incentive Fee", meta['incentive_fee'] + (f" over {meta['hurdle']} hurdle" if meta.get('hurdle') else "")))
    liq = ', '.join(meta[k] for k in ('redemption', 'notice', 'lockup', 'gate') if meta.get(k))
    if liq:
        terms.append(("Liquidity", liq))
    sp = ', '.join(f"{meta[k]}" for k in ('prime_broker', 'administrator', 'auditor', 'legal_counsel') if meta.get(k))
    if sp:
        terms.append(("Service Providers", sp))
    if terms:
        out += [Paragraph("Deck Terms", _SUB), _kv_table(terms)]
    return out


# ── main entry ────────────────────────────────────────────────────────────────

def _drawdown_tables(fund_df):
    """Largest/Longest/Mean/Median analysis + top episodes — mirrors the dashboard."""
    eps = top_drawdowns(fund_df)
    if not eps:
        return []
    import datetime as _dt

    def _pmd(s):
        if not s or s in ('Start', 'Ongoing'):
            return None
        try:
            mo, yr = s.split()
            return _dt.date(int(yr), MN.index(mo) + 1, 1)
        except Exception:
            return None

    def _avg_date(strs, med=False):
        ords = [d.toordinal() for d in (_pmd(s) for s in strs) if d]
        if not ords:
            return '\u2013'
        v = np.median(ords) if med else np.mean(ords)
        d = _dt.date.fromordinal(int(round(v)))
        return f"{MN[d.month - 1]} {d.year}"

    def _dur(e, k):
        return str(e[k]) if e[k] is not None else 'Ongoing'

    largest = eps[0]
    longest = max(eps, key=lambda e: e['total_months'] if e['total_months'] is not None else e['peak_to_trough'])
    depths = [abs(e['drawdown']) for e in eps]
    p2t = [e['peak_to_trough'] for e in eps]
    t2r = [e['trough_to_recovery'] for e in eps if e['trough_to_recovery'] is not None]
    p2r = [e['total_months'] for e in eps if e['total_months'] is not None]

    analysis = [
        ['Drawdown %', f"{abs(largest['drawdown']):.2f}", f"{abs(longest['drawdown']):.2f}", f"{np.mean(depths):.2f}", f"{np.median(depths):.2f}"],
        ['Peak', largest['peak_date'], longest['peak_date'], _avg_date([e['peak_date'] for e in eps]), _avg_date([e['peak_date'] for e in eps], med=True)],
        ['Trough', largest['trough_date'], longest['trough_date'], _avg_date([e['trough_date'] for e in eps]), _avg_date([e['trough_date'] for e in eps], med=True)],
        ['Recovery', largest['recovery_date'], longest['recovery_date'], _avg_date([e['recovery_date'] for e in eps]), _avg_date([e['recovery_date'] for e in eps], med=True)],
        ['Peak to Trough (mo)', _dur(largest, 'peak_to_trough'), _dur(longest, 'peak_to_trough'), f"{np.mean(p2t):.2f}", f"{np.median(p2t):.2f}"],
        ['Trough to Recovery (mo)', _dur(largest, 'trough_to_recovery'), _dur(longest, 'trough_to_recovery'), (f"{np.mean(t2r):.2f}" if t2r else '\u2013'), (f"{np.median(t2r):.2f}" if t2r else '\u2013')],
        ['Peak to Recovery (mo)', _dur(largest, 'total_months'), _dur(longest, 'total_months'), (f"{np.mean(p2r):.2f}" if p2r else '\u2013'), (f"{np.median(p2r):.2f}" if p2r else '\u2013')],
    ]
    ep_rows = []
    for i, e in enumerate(eps):
        tr = f"{e['trough_to_recovery']}m" if e['trough_to_recovery'] is not None else 'Ongoing'
        tot = f"{e['total_months']}m" if e['total_months'] is not None else 'Ongoing'
        ep_rows.append([str(i + 1), f"{e['drawdown']:.2f}%", e['peak_date'], e['trough_date'],
                        e['recovery_date'], f"{e['peak_to_trough']}m", tr, tot])
    return [
        Spacer(1, 6),
        Paragraph("Drawdown Analysis", _SUB),
        _grid_table(['', 'Largest', 'Longest', 'Mean', 'Median'], analysis),
        Spacer(1, 8),
        Paragraph("Top Drawdown Episodes", _SUB),
        _grid_table(['#', 'Drawdown', 'Peak', 'Trough', 'Recovery', 'Pk\u2192Tr', 'Tr\u2192Rec', 'Total'], ep_rows),
    ]


def build_report(fund_df, fund_name, mkt_df, mkt_name, agg_df, bm3_df, bm3_name, meta=None):
    meta = meta or {}
    fd = fund_df.reset_index(drop=True)
    span = f"{MN[int(fd.iloc[0].month)-1]} {int(fd.iloc[0].year)} \u2013 {MN[int(fd.iloc[-1].month)-1]} {int(fd.iloc[-1].year)}"
    gen_date = datetime.date.today().strftime('%d %b %Y')
    story = []

    # Cover + deck terms
    story += _cover(fund_name, span, gen_date, meta)

    # Executive Summary
    story.append(Paragraph("Executive Summary", _H))
    import re
    for head, body_html in build_exec_summary(fund_df, fund_name, mkt_df, mkt_name, agg_df, bm3_df, bm3_name, meta=meta):
        text = re.sub(r'<[^>]+>', '', body_html)
        story.append(Paragraph(f"<b>{head}.</b> {text}", _BODY))

    # Headline metrics
    r = fd['ret'].values
    story.append(Paragraph("Headline Metrics", _SUB))
    story.append(_kv_table([
        ("Annualized return", f"{geo_return(r):.2f}%"),
        ("Annualized volatility", f"{ann_vol(r):.2f}%"),
        ("Sharpe (rf = 3M T-bills)", f"{sharpe_with_rf(fd):.2f}"),
        ("Sortino", f"{sortino_with_rf(fd):.2f}"),
        ("Max drawdown", f"{max_drawdown(r):.2f}%"),
        ("Calmar", f"{calmar(r):.2f}"),
    ]))

    bms = [(mkt_name, mkt_df, GOLD), (agg_df is not None and 'Bloomberg Agg' or '', agg_df, SKY)]
    bms = [(n, d, c) for n, d, c in bms if d is not None]
    if bm3_df is not None:
        bms.append((bm3_name, bm3_df, GREENc))

    # Performance
    story.append(Paragraph("Performance", _H))
    story.append(KeepTogether([Paragraph("Cumulative Growth of $100", _SUB), fig_cumulative(fund_df, fund_name, bms)]))
    story.append(Paragraph("Monthly Return Track Record (%)", _SUB))
    story.append(calendar_table(fund_name, [(fund_name, fund_df), ('MSCI World Hdg', mkt_df),
                                            ('Bloomberg Agg', agg_df), (bm3_name, bm3_df)]))
    story.append(Spacer(1, 8))
    story.append(KeepTogether([Paragraph("Drawdowns", _SUB), fig_drawdowns(fund_df)]))
    story += _drawdown_tables(fund_df)

    # Risk & significance
    story.append(PageBreak())
    story.append(Paragraph("Risk &amp; Significance", _H))
    story.append(KeepTogether([Paragraph("Distribution of Monthly Returns", _SUB), fig_histogram(fund_df)]))
    sig = sharpe_significance(fund_df); reg_m = piecewise_beta_regression(fund_df, mkt_df)
    if sig.get('n', 0) >= 3:
        srows = [("Annualized Sharpe", f"{sig['sharpe']:.4f}"),
                 ("Sharpe std. error (Lo)", f"{sig['se']:.4f}"),
                 ("Sharpe 95% CI", f"{sig['ci'][0]:.4f} to {sig['ci'][1]:.4f}"),
                 ("Mean-return t-stat", f"{sig['ret_t']:.4f}"),
                 ("Mean-return p-value", f"{sig['ret_p']:.4f}")]
        if reg_m:
            srows += [("Monthly alpha vs " + mkt_name, f"{reg_m['alpha']:.4f}%"),
                      ("Alpha t-stat", f"{reg_m['t_alpha']:.4f}"),
                      ("Alpha 95% CI", f"{reg_m['alpha_ci'][0]:.4f}% to {reg_m['alpha_ci'][1]:.4f}%")]
        story.append(Paragraph("Statistical Significance", _SUB)); story.append(_kv_table(srows))
    tr = tail_risk(r); cap = capture_ratios(fund_df, mkt_df)
    if tr.get('n', 0) >= 4:
        story.append(Paragraph("Tail Risk &amp; Ratios", _SUB))
        story.append(_kv_table([
            ("Historical VaR 95% / 99%", f"{tr['var95']:.2f}% / {tr['var99']:.2f}%"),
            ("Modified VaR 95% / 99% (Cornish-Fisher)", f"{tr['mvar95']:.2f}% / {tr['mvar99']:.2f}%"),
            ("Expected shortfall 95% / 99%", f"{tr['cvar95']:.2f}% / {tr['cvar99']:.2f}%"),
            ("Ulcer index", f"{tr['ulcer']:.2f}"),
            ("Tail ratio / Gain-to-pain / Omega", f"{tr['tail_ratio']:.2f} / {tr['gain_to_pain']:.2f} / {tr['omega']:.2f}"),
            ("Up / Down capture vs " + mkt_name,
             f"{(cap.get('up_capture') or 0):.2f}% / {(cap.get('down_capture') or 0):.2f}%"),
        ]))
    ds = desmooth_stats(fund_df, mkt_df)
    if ds.get('n', 0) >= 6:
        story.append(KeepTogether([
            Paragraph(f"De-Smoothing (AR1 \u03c1 = {ds['rho']:.4f})", _SUB),
            _grid_table(["Metric", "Reported", "Unsmoothed"], [
                ["Annualized volatility", f"{ds['vol_rep']:.2f}%", f"{ds['vol_uns']:.2f}%"],
                ["Sharpe", f"{ds['sharpe_rep']:.2f}", f"{ds['sharpe_uns']:.2f}"],
                [f"Beta vs {mkt_name}", f"{ds['beta_rep']:.2f}", f"{ds['beta_uns']:.2f}"]]),
            fig_desmooth(fund_df, mkt_df)]))

    # Market
    story.append(PageBreak())
    story.append(Paragraph("Market Relationship", _H))
    story.append(KeepTogether([Paragraph("Regression vs Benchmarks", _SUB),
                               fig_regression(fund_df, [(n, d) for n, d, c in bms])]))
    story.append(KeepTogether([Paragraph("Co-Movement in Market Extremes", _SUB),
                               fig_comovement(fund_df, mkt_df, mkt_name, agg_df, 'Bloomberg Agg', bm3_df, bm3_name)]))
    story.append(KeepTogether([Paragraph("Up / Down Capture", _SUB),
                               fig_updown(fund_df, mkt_df, [('Bloomberg Agg', agg_df)] + ([(bm3_name, bm3_df)] if bm3_df is not None else []))]))
    story.append(KeepTogether([Paragraph("Strategy vs Market Waterfall", _SUB), fig_waterfall(fund_df, mkt_df)]))

    # Crisis
    story.append(PageBreak())
    story.append(Paragraph("Crisis &amp; Cyclical Behaviour", _H))
    story.append(KeepTogether([Paragraph("Performance During Discrete Shock Events", _SUB),
                               fig_shocks(fund_df, mkt_df, agg_df, 'Bloomberg Agg', bm3_df, bm3_name)]))
    ev_rows = macro_events_table(fund_df, mkt_df)
    if ev_rows:
        story.append(Paragraph("Macro Events", _SUB))
        story.append(_grid_table(["Event", "Period", "Strategy", mkt_name, "Spread"],
            [[e['name'], e['period'], f"{e['fund_ret']:.2f}%" if e.get('fund_ret') is not None else "\u2013",
              f"{e['bm_ret']:.2f}%" if e.get('bm_ret') is not None else "\u2013",
              f"{e['spread']:+.2f}%" if e.get('spread') is not None else "\u2013"] for e in ev_rows]))
    rfig = fig_rolling(fund_df)
    if rfig is not None:
        story.append(KeepTogether([Paragraph("Rolling 12-Month Return &amp; Sharpe", _SUB), rfig]))
    story.append(KeepTogether([Paragraph("Seasonality", _SUB), fig_seasonality(fund_df)]))

    # Methodology
    story.append(PageBreak())
    story.append(Paragraph("Methodology &amp; Notes", _H))
    for note in [
        "Returns are the monthly net series ingested from the deck. Annualized figures use geometric compounding; volatility is the sample standard deviation annualized by \u221a12.",
        "Risk-free rate is the 3-month US Treasury bill (TB3MS, FRED), applied monthly to Sharpe and Sortino.",
        "Value-at-Risk is reported on a monthly basis: historical VaR is the empirical 5th/1st percentile; modified VaR applies the Cornish-Fisher expansion using sample skew and excess kurtosis; expected shortfall (CVaR) is the mean of returns beyond the VaR threshold.",
        "De-smoothed returns use a first-order Geltner/Okunev-White unsmoothing with the estimated AR(1) coefficient, removing serial dependence that can understate volatility and inflate Sharpe.",
        "The Sharpe standard error follows Lo (2002) under an i.i.d. assumption; serial correlation would widen the interval. Alpha and beta t-statistics are from OLS against the stated benchmark.",
        "Benchmarks: MSCI World Hedged USD (equity), Bloomberg Global Aggregate (fixed income), and the selected HFRX hedge-fund index. Index returns do not reflect fees.",
    ]:
        story.append(Paragraph("\u2022 " + note, _BODY))

    # DDQ
    story.append(PageBreak())
    story.append(Paragraph("Due-Diligence Questionnaire", _H))
    story.append(Paragraph("<i>Tags: Clarify \u00b7 Assumptions \u00b7 Evidence \u00b7 Perspective \u00b7 Implications \u00b7 Meta</i>", _BODY))
    qno = 0
    for i, (cat, qs) in enumerate(build_ddq(fund_df, fund_name, mkt_df, meta=meta)):
        block = [Paragraph(f"{chr(65+i)}. {cat}", _SUB)]
        for q, ty in qs:
            qno += 1
            block.append(Paragraph(f"<font color='{TEAL}'><b>{qno}.</b></font> {q} "
                                   f"<font color='{TEAL}' size=7><b>[{ty}]</b></font>", _Q))
        story.append(KeepTogether(block) if len(block) <= 6 else block[0])
        if len(block) > 6:
            for b in block[1:]:
                story.append(b)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=0.7 * inch,
                            title=f"Argus Report \u2014 {fund_name}")
    doc._argus_date = gen_date
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buf.seek(0)
    return buf.getvalue()
