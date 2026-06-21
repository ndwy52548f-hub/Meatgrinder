"""
pdf_extract.py — pull monthly return series out of a manager's PDF deck.

The dominant institutional format is a year-by-month grid (rows = year, often
with a strategy/fund name, columns = Jan…Dec plus a YTD), exactly like a
typical track-record page. This module finds those tables, parses each one into
Argus's golden (year, month, ret) shape, and returns one candidate per
fund/strategy so the user can pick the right series and eyeball it before it
loads. A simple two-column Date/Return table is handled too.

Extraction is heuristic and never silently trusted — the app always shows a
preview for confirmation.
"""

import re
import pandas as pd

try:
    import pdfplumber
    _HAVE_PDFPLUMBER = True
except Exception:               # pragma: no cover
    _HAVE_PDFPLUMBER = False

_MONTHS = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
           'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
_YEAR_RE = re.compile(r'(?:19|20)\d{2}')


def _to_pct(cell):
    """Parse a percentage-ish cell to a float (percent units), else None."""
    if cell is None:
        return None
    s = str(cell).strip()
    if s == '' or s.lower() in ('-', '—', 'n/a', 'na', 'nm', 'n.m.', '–'):
        return None
    neg = s.startswith('(') and s.endswith(')')      # accounting negatives
    s = s.strip('()').replace('%', '').replace(',', '').replace('\u2212', '-').strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _month_columns(row):
    """Map column index -> month number for any header cell naming a month."""
    out = {}
    for i, c in enumerate(row):
        if c is None:
            continue
        key = str(c).strip().lower()[:3]
        if key in _MONTHS:
            out[i] = _MONTHS[key]
    return out


def _find_year(row):
    """First 4-digit year found in a row, or None."""
    for c in row:
        if c is None:
            continue
        m = _YEAR_RE.search(str(c))
        if m:
            return int(m.group())
    return None


def _label_for_row(row, year_idx, month_idxs):
    """The strategy/fund name in a row: first text cell that isn't the year
    column or a month/value column."""
    skip = set(month_idxs) | ({year_idx} if year_idx is not None else set())
    for i, c in enumerate(row):
        if i in skip or c is None:
            continue
        txt = str(c).strip()
        if not txt:
            continue
        if _to_pct(txt) is not None and not re.search(r'[A-Za-z]', txt):
            continue                                  # pure number, not a label
        if _YEAR_RE.fullmatch(txt):
            continue
        return txt
    return ''


def _parse_grid_table(table):
    """Parse one extracted table assumed to be a year×month grid.

    Returns a dict {label: [(year, month, ret), ...]} or {} if not a grid.
    """
    # locate the header row = the row naming the most months (need >= 6)
    header_idx, month_idxs = None, {}
    for i, row in enumerate(table[:6]):
        mc = _month_columns(row)
        if len(mc) >= 6 and len(mc) > len(month_idxs):
            header_idx, month_idxs = i, mc
    if header_idx is None:
        return {}

    # which column holds the year? the one with the most year-looking cells
    year_idx, best = None, 0
    ncol = max(len(r) for r in table)
    for ci in range(ncol):
        hits = 0
        for row in table[header_idx + 1:]:
            if ci < len(row) and row[ci] and _YEAR_RE.search(str(row[ci])):
                hits += 1
        if hits > best:
            best, year_idx = hits, ci

    series = {}
    last_year = None
    for row in table[header_idx + 1:]:
        yr = None
        if year_idx is not None and year_idx < len(row):
            yr = _find_year([row[year_idx]])
        if yr is None:
            yr = _find_year(row)
        if yr is None:
            yr = last_year                            # grouped rows share a year
        if yr is None:
            continue
        last_year = yr

        label = _label_for_row(row, year_idx, month_idxs) or 'Returns'
        rows = []
        for ci, mo in month_idxs.items():
            if ci < len(row):
                v = _to_pct(row[ci])
                if v is not None:
                    rows.append((yr, mo, v))
        if rows:
            series.setdefault(label, []).extend(rows)
    return series


def _dedupe(records):
    """Collapse duplicate (year, month) keeping the last, sort chronologically."""
    df = pd.DataFrame(records, columns=['year', 'month', 'ret'])
    df = df.drop_duplicates(subset=['year', 'month'], keep='last')
    df = df.sort_values(['year', 'month']).reset_index(drop=True)
    return df


def extract_return_series(file_like):
    """
    Extract candidate monthly-return series from a PDF.

    Returns
    -------
    candidates : list of dicts, each {label, df(year,month,ret), n, page}
    warnings   : list of str
    """
    warnings = []
    if not _HAVE_PDFPLUMBER:
        return [], ['pdfplumber is not installed — add it to requirements.txt.']

    try:
        pdf = pdfplumber.open(file_like)
    except Exception as exc:
        return [], [f'Could not open PDF: {exc}']

    candidates = []
    with pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables() or []
            except Exception:
                continue
            for table in tables:
                if not table or len(table) < 2:
                    continue
                series = _parse_grid_table(table)
                for label, recs in series.items():
                    df = _dedupe(recs)
                    if len(df) >= 6:                  # Argus needs >= 6 months
                        candidates.append({
                            'label': label.strip()[:60] or 'Returns',
                            'df': df,
                            'n': len(df),
                            'page': pno,
                        })

    # merge candidates with the same label across pages (multi-page track records)
    merged = {}
    for c in candidates:
        key = c['label'].lower()
        if key in merged:
            merged[key]['df'] = _dedupe(
                pd.concat([merged[key]['df'], c['df']]).values.tolist())
            merged[key]['n'] = len(merged[key]['df'])
        else:
            merged[key] = c
    candidates = sorted(merged.values(), key=lambda c: c['n'], reverse=True)

    if not candidates:
        warnings.append('No monthly-return grid was found in this PDF. The table '
                        'may be an image, or in a layout the parser can\'t read.')
    return candidates, warnings
