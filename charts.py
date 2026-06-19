"""
charts.py — Meatgrinder Chart Definitions
All Plotly figures, matching the HTML version's visual design language.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from analytics import (
    geo_return, ann_vol, drawdown_series, acf, acf_conf_band,
    qq_data, skewness, excess_kurtosis, rolling_metrics,
    MSCI_DF, AGG_DF
)

# ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
C = {
    'bg':        '#F5F0E8',
    'surface':   '#FDFAF4',
    'border':    '#DDD8CC',
    'text':      '#1A1A1A',
    'muted':     '#888070',
    'accent':    '#1A1A1A',
    'green':     '#1B6B3A',
    'red':       '#B01C2E',
    'gold':      '#9A6B00',
    'grid':      'rgba(200,192,176,0.5)',
    'accent_lt': 'rgba(26,26,26,0.07)',
    'green_lt':  'rgba(27,107,58,0.10)',
    'red_lt':    'rgba(176,28,46,0.10)',
}

FONT_UI   = 'Inter, -apple-system, Segoe UI, sans-serif'
FONT_MONO = 'JetBrains Mono, Menlo, Monaco, Consolas, monospace'

LAYOUT_BASE = dict(
    paper_bgcolor = C['bg'],
    plot_bgcolor  = C['surface'],
    font          = dict(family=FONT_UI, size=11, color=C['text']),
    margin        = dict(l=48, r=24, t=36, b=36),
    xaxis         = dict(gridcolor=C['grid'], tickfont=dict(family=FONT_MONO, size=10, color=C['muted']),
                         linecolor=C['border'], zeroline=False),
    yaxis         = dict(gridcolor=C['grid'], tickfont=dict(family=FONT_MONO, size=10, color=C['muted']),
                         linecolor=C['border'], zeroline=True, zerolinecolor=C['border']),
    showlegend    = False,
    hovermode     = 'x unified',
)

MN = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


def _date_labels(df: pd.DataFrame) -> list[str]:
    return [f"{MN[int(r.month)-1]} {int(r.year)}" for r in df.itertuples()]


def _apply_base(fig: go.Figure, **overrides) -> go.Figure:
    layout = {**LAYOUT_BASE, **overrides}
    fig.update_layout(**layout)
    return fig


# ─── CUMULATIVE WEALTH ────────────────────────────────────────────────────────

def chart_cumulative(fund_df: pd.DataFrame,
                     fund_name: str,
                     bm1_df: pd.DataFrame | None = None,
                     bm1_name: str = 'MSCI World Hdg',
                     bm2_df: pd.DataFrame | None = None,
                     bm2_name: str = 'Bloomberg Agg') -> go.Figure:
    """Cumulative wealth (growth of $100) chart."""
    fig = go.Figure()

    labels = _date_labels(fund_df)
    wealth = 100 * np.cumprod(1 + fund_df['ret'].values / 100)
    fig.add_trace(go.Scatter(
        x=labels, y=wealth, name=fund_name,
        line=dict(color=C['accent'], width=2.5),
        hovertemplate='%{y:.1f}<extra>' + fund_name + '</extra>'
    ))

    if bm1_df is not None:
        merged = fund_df.merge(bm1_df, on=['year','month'])
        if len(merged) > 0:
            bm_labels = [f"{MN[int(r.month_x)-1]} {int(r.year_x)}" for r in merged.itertuples()] \
                        if 'month_x' in merged.columns else _date_labels(merged)
            # Handle merge result column naming
            bm_ret_col = 'ret_y' if 'ret_y' in merged.columns else 'ret'
            w = 100 * np.cumprod(1 + merged[bm_ret_col].values / 100)
            fig.add_trace(go.Scatter(
                x=_date_labels(fund_df.merge(bm1_df, on=['year','month'])),
                y=w, name=bm1_name,
                line=dict(color=C['muted'], width=1.5, dash='dot'),
                hovertemplate='%{y:.1f}<extra>' + bm1_name + '</extra>'
            ))

    if bm2_df is not None:
        merged2 = fund_df.merge(bm2_df, on=['year','month'])
        if len(merged2) > 0:
            bm2_ret_col = 'ret_y' if 'ret_y' in merged2.columns else 'ret'
            w2 = 100 * np.cumprod(1 + merged2[bm2_ret_col].values / 100)
            fig.add_trace(go.Scatter(
                x=_date_labels(merged2),
                y=w2, name=bm2_name,
                line=dict(color=C['gold'], width=1.5, dash='dot'),
                hovertemplate='%{y:.1f}<extra>' + bm2_name + '</extra>'
            ))

    fig.update_layout(showlegend=True)
    _apply_base(fig,
        yaxis_title='Growth of $100',
        yaxis=dict(tickformat=',.0f', gridcolor=C['grid'],
                   tickfont=dict(family=FONT_MONO, size=10, color=C['muted'])),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
                    font=dict(size=10))
    )
    return fig


# ─── DRAWDOWN SERIES ──────────────────────────────────────────────────────────

def chart_drawdowns(fund_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    dd = drawdown_series(fund_df['ret'].values)
    labels = _date_labels(fund_df)
    fig.add_trace(go.Scatter(
        x=labels, y=dd,
        fill='tozeroy',
        fillcolor='rgba(192,24,42,0.15)',
        line=dict(color=C['red'], width=1.5),
        hovertemplate='%{y:.2f}%<extra>Drawdown</extra>'
    ))
    _apply_base(fig, yaxis_title='Drawdown (%)',
                yaxis=dict(ticksuffix='%', gridcolor=C['grid'],
                           tickfont=dict(family=FONT_MONO, size=10, color=C['muted'])))
    return fig


# ─── MONTHLY RETURN BARS ──────────────────────────────────────────────────────

def chart_monthly_bars(fund_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    rets = fund_df['ret'].values
    labels = _date_labels(fund_df)
    colors = [C['green'] if r >= 0 else C['red'] for r in rets]
    fig.add_trace(go.Bar(
        x=labels, y=rets,
        marker_color=colors,
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    _apply_base(fig, yaxis_title='Monthly Return (%)',
                yaxis=dict(ticksuffix='%', gridcolor=C['grid'],
                           tickfont=dict(family=FONT_MONO, size=10, color=C['muted'])),
                bargap=0.1)
    return fig


# ─── HISTOGRAM + NORMAL FIT ───────────────────────────────────────────────────

def chart_histogram(fund_df: pd.DataFrame, outliers_df: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    rets = fund_df['ret'].values
    m, s = np.mean(rets), np.std(rets, ddof=1)

    # Histogram
    fig.add_trace(go.Histogram(
        x=rets,
        nbinsx=30,
        marker_color=C['accent_lt'],
        marker_line_color=C['accent'],
        marker_line_width=0.8,
        name='Monthly Returns',
        hovertemplate='%{x:.1f}% : %{y} months<extra></extra>'
    ))

    # Normal overlay
    x_range = np.linspace(m - 4*s, m + 4*s, 200)
    from scipy.stats import norm
    scale = len(rets) * (rets.max() - rets.min()) / 30  # approximate bin width scale
    bin_width = (rets.max() - rets.min()) / 30
    y_normal = norm.pdf(x_range, m, s) * len(rets) * bin_width
    fig.add_trace(go.Scatter(
        x=x_range, y=y_normal,
        line=dict(color=C['red'], width=1.5),
        name='Normal Fit',
        hoverinfo='skip'
    ))

    # Mark outliers
    if outliers_df is not None and len(outliers_df) > 0:
        for r in outliers_df['ret']:
            fig.add_vline(x=r, line_color=C['gold'], line_dash='dot', line_width=1.2)

    fig.update_layout(showlegend=True, barmode='overlay')
    _apply_base(fig,
        xaxis_title='Monthly Return (%)', yaxis_title='Frequency',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, x=0),
    )
    return fig


# ─── Q-Q PLOT ─────────────────────────────────────────────────────────────────

def chart_qq(fund_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    qq = qq_data(fund_df['ret'].values)
    th = qq['theoretical']
    obs = qq['data']

    fig.add_trace(go.Scatter(
        x=th, y=obs, mode='markers',
        marker=dict(color=C['accent'], size=6, opacity=0.7),
        hovertemplate='Theoretical: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>'
    ))

    # 45° reference line
    lo, hi = min(th.min(), obs.min()), max(th.max(), obs.max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        line=dict(color=C['red'], dash='dash', width=1.2),
        hoverinfo='skip', mode='lines'
    ))

    _apply_base(fig, xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')
    return fig


# ─── AUTOCORRELATION ──────────────────────────────────────────────────────────

def chart_acf(fund_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    rets = fund_df['ret'].values
    lags = list(range(1, 13))
    ac = acf(rets, max_lag=12)
    conf = acf_conf_band(len(rets))

    # Confidence bands
    fig.add_hrect(y0=-conf, y1=conf, fillcolor=C['accent_lt'],
                  line_width=0, annotation_text='95% CI')

    colors = [C['accent'] if abs(a) > conf else C['muted'] for a in ac]
    fig.add_trace(go.Bar(
        x=lags, y=ac,
        marker_color=colors,
        hovertemplate='Lag %{x}: %{y:.4f}<extra></extra>'
    ))
    fig.add_hline(y=0, line_color=C['border'], line_width=1)

    _apply_base(fig,
        xaxis_title='Lag (months)', yaxis_title='Autocorrelation',
        xaxis=dict(tickvals=lags, gridcolor=C['grid'],
                   tickfont=dict(family=FONT_MONO, size=10, color=C['muted'])),
    )
    return fig


# ─── CALENDAR HEATMAP ─────────────────────────────────────────────────────────

def chart_calendar_heatmap(fund_df: pd.DataFrame) -> go.Figure:
    """Monthly return heatmap: rows=month, cols=year."""
    years = sorted(fund_df['year'].unique())
    z, text = [], []
    for m_idx in range(1, 13):
        row_z, row_t = [], []
        for y in years:
            sub = fund_df[(fund_df['year'] == y) & (fund_df['month'] == m_idx)]
            if len(sub) > 0:
                v = float(sub.iloc[0]['ret'])
                row_z.append(v)
                row_t.append(f"{v:+.2f}%")
            else:
                row_z.append(None)
                row_t.append('')
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(y) for y in years],
        y=MN,
        text=text,
        texttemplate='%{text}',
        textfont=dict(family=FONT_MONO, size=9),
        colorscale=[[0.0, '#c0182a'], [0.5, '#f4f7fb'], [1.0, '#0a7a50']],
        zmid=0,
        showscale=True,
        hovertemplate='%{y} %{x}: %{text}<extra></extra>',
        colorbar=dict(ticksuffix='%', tickfont=dict(family=FONT_MONO, size=9))
    ))

    fig.update_layout(
        paper_bgcolor=C['bg'],
        plot_bgcolor=C['surface'],
        font=dict(family=FONT_UI, size=10, color=C['text']),
        margin=dict(l=48, r=24, t=24, b=36),
        xaxis=dict(tickfont=dict(family=FONT_MONO, size=9, color=C['muted'])),
        yaxis=dict(tickfont=dict(family=FONT_UI, size=10, color=C['text'])),
        height=400,
    )
    return fig


# ─── ROLLING METRICS ──────────────────────────────────────────────────────────

def chart_rolling(fund_df: pd.DataFrame, metric: str = 'roll_ret') -> go.Figure:
    """Rolling 12-month metric chart."""
    roll = rolling_metrics(fund_df)
    if roll.empty:
        return go.Figure()

    fig = go.Figure()
    labels = [f"{MN[int(r.month)-1]} {int(r.year)}" for r in roll.itertuples()]
    y = roll[metric].values
    colors = [C['green'] if v >= 0 else C['red'] for v in y]

    labels_map = {
        'roll_ret': ('Rolling 12M Return (%)', '%'),
        'roll_vol': ('Rolling 12M Volatility (%)', '%'),
        'roll_sharpe': ('Rolling 12M Sharpe', ''),
    }
    title, suffix = labels_map.get(metric, (metric, ''))

    if metric == 'roll_ret':
        fig.add_trace(go.Bar(x=labels, y=y, marker_color=colors,
                             hovertemplate=f'%{{y:.2f}}{suffix}<extra></extra>'))
    else:
        fig.add_trace(go.Scatter(x=labels, y=y,
                                 line=dict(color=C['accent'], width=2),
                                 hovertemplate=f'%{{y:.3f}}{suffix}<extra></extra>'))
        fig.add_hline(y=0, line_color=C['border'], line_width=1)

    _apply_base(fig, yaxis_title=title,
                yaxis=dict(ticksuffix=suffix, gridcolor=C['grid'],
                           tickfont=dict(family=FONT_MONO, size=10, color=C['muted'])))
    return fig


# ─── REGRESSION SCATTER ───────────────────────────────────────────────────────

def chart_regression(reg_result: dict, fund_name: str, bm_name: str) -> go.Figure:
    fig = go.Figure()
    x, y = reg_result['x'], reg_result['y']

    # Up months
    fig.add_trace(go.Scatter(
        x=reg_result['up_x'], y=reg_result['up_y'],
        mode='markers',
        marker=dict(color=C['green'], size=6, opacity=0.75),
        name='BM Up',
        hovertemplate=f'{bm_name}: %{{x:.2f}}%<br>{fund_name}: %{{y:.2f}}%<extra></extra>'
    ))

    # Down months
    fig.add_trace(go.Scatter(
        x=reg_result['dn_x'], y=reg_result['dn_y'],
        mode='markers',
        marker=dict(color=C['red'], size=6, opacity=0.75),
        name='BM Down',
        hovertemplate=f'{bm_name}: %{{x:.2f}}%<br>{fund_name}: %{{y:.2f}}%<extra></extra>'
    ))

    # OLS fit line
    x_range = np.linspace(x.min(), x.max(), 100)
    y_hat = reg_result['alpha'] + reg_result['beta'] * x_range
    fig.add_trace(go.Scatter(
        x=x_range, y=y_hat,
        mode='lines',
        line=dict(color=C['accent'], width=2),
        name='OLS Fit',
        hoverinfo='skip'
    ))

    # Zero lines
    fig.add_hline(y=0, line_color=C['border'], line_width=1)
    fig.add_vline(x=0, line_color=C['border'], line_width=1)

    fig.update_layout(showlegend=True,
                      legend=dict(orientation='h', yanchor='bottom', y=1.01, x=0))
    _apply_base(fig,
        xaxis_title=f'{bm_name} Monthly Return (%)',
        yaxis_title=f'{fund_name} Monthly Return (%)',
    )
    return fig


# ─── SEASONALITY ──────────────────────────────────────────────────────────────

def chart_seasonality_monthly(seas: dict) -> go.Figure:
    fig = go.Figure()
    monthly = seas['monthly']
    months = [MN[int(m)-1] for m in monthly['month']]
    means = monthly['mean'].values
    colors = [C['green'] if v >= 0 else C['red'] for v in means]

    fig.add_trace(go.Bar(
        x=months, y=means,
        marker_color=colors,
        error_y=dict(
            type='data',
            array=(monthly['std'] / np.sqrt(monthly['count'])).values,
            visible=True, color=C['muted'], thickness=1.5
        ),
        hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
    ))
    fig.add_hline(y=0, line_color=C['border'], line_width=1)
    _apply_base(fig, yaxis_title='Average Monthly Return (%)',
                yaxis=dict(ticksuffix='%', gridcolor=C['grid'],
                           tickfont=dict(family=FONT_MONO, size=10, color=C['muted'])))
    return fig


def chart_seasonality_quarterly(seas: dict) -> go.Figure:
    fig = go.Figure()
    quarterly = seas['quarterly']
    quarters = [f"Q{int(q)}" for q in quarterly['quarter']]
    means = quarterly['mean'].values
    colors = [C['green'] if v >= 0 else C['red'] for v in means]

    fig.add_trace(go.Bar(
        x=quarters, y=means,
        marker_color=colors,
        hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
    ))
    fig.add_hline(y=0, line_color=C['border'], line_width=1)
    _apply_base(fig, yaxis_title='Average Quarterly Return (%)',
                yaxis=dict(ticksuffix='%', gridcolor=C['grid'],
                           tickfont=dict(family=FONT_MONO, size=10, color=C['muted'])))
    return fig
