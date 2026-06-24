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


# ── Raw-text grid parser ──────────────────────────────────────────────────────
# Some decks merge all twelve month columns into a single table cell, so the
# cell-based parser sees no grid. The raw text, however, lays each row out as
# "[year] <series name> v1 v2 … v12 [YTD]". This parser reads that, and also
# scrubs diagonal-watermark characters that bleed into the numbers
# (e.g. "o5.2%", "-2p.1%", "-10.2%c4.7%", stray single-letter tokens).

_MONTHS_HEADER = re.compile(r'\bjan\b.*\bfeb\b.*\bdec\b', re.I)
_VALUE_RE = re.compile(r'^-?\d+(?:\.\d+)?%?$')
_DASHES = {'-', '\u2013', '\u2014'}


def _scrub_line(line):
    """De-glue two percentages joined by a watermark letter (label-safe)."""
    return re.sub(r'%[A-Za-z@]+(?=-?\d)', '% ', line)


def _text_tokens(line):
    """Tokenise, dropping stray single-letter watermark fragments."""
    return [t for t in line.split() if not (len(t) == 1 and t.isalpha()) and t != '@']


def _token_value(tok):
    """(is_value, value_or_None). Strips a stray watermark letter from a number;
    None value means a dash (missing month)."""
    c = re.sub(r'[A-Za-z@]', '', tok)
    if c in _DASHES:
        return True, None
    if _VALUE_RE.match(c):
        return True, float(c.replace('%', ''))
    return False, None


def _parse_text_grid(text):
    """Parse a page's raw text into {label: [(year, month, ret), ...]}."""
    series, last_year, armed = {}, None, False
    for raw in (text or '').split('\n'):
        if _MONTHS_HEADER.search(raw):
            armed = True
            continue
        if not armed:
            continue
        toks = _text_tokens(_scrub_line(raw))
        if not toks:
            continue
        yr = None
        if _YEAR_RE.fullmatch(toks[0]):
            yr = int(toks[0]); toks = toks[1:]
        # maximal trailing run of value/dash tokens (stops at the label words)
        run = []
        for t in reversed(toks):
            ok, v = _token_value(t)
            if ok:
                run.append(v)
            else:
                break
        run.reverse()
        n = len(run)
        if n not in (12, 13):                 # 12 months, optionally + a YTD column
            continue
        months = run[:12] if n == 13 else run
        label = ' '.join(toks[:len(toks) - n]).strip()
        if not re.search(r'[A-Za-z]', label):
            continue
        y = yr if yr is not None else last_year
        if yr is not None:
            last_year = yr
        if y is None:
            continue
        for mo, v in enumerate(months, start=1):
            if v is not None:
                series.setdefault(label, []).append((y, mo, v))
    return series


# ── Positional grid parser ────────────────────────────────────────────────────
# Some decks label each grid only with a heading above it ("<Name> Monthly
# Returns (Net)") and lay the numbers out under fixed month columns, with
# partial first/last years. Text order alone can't tell which month a value
# belongs to, so we align each value to a month column by its x-coordinate.

def _group_words_into_lines(words, tol=3.0):
    buckets = {}
    for w in words:
        buckets.setdefault(round(w['top'] / tol), []).append(w)
    return [sorted(v, key=lambda x: x['x0']) for _, v in sorted(buckets.items())]


def _parse_positional(page):
    """Parse one page's label-less, column-aligned monthly grid via x-positions.
    Returns {label: [(year, month, ret), ...]}."""
    try:
        lines = _group_words_into_lines(page.extract_words())
    except Exception:
        return {}

    # header row: a line naming all twelve months as separate words
    header_i, centers = None, None
    for i, ln in enumerate(lines):
        names = [w['text'].strip().lower()[:3] for w in ln]
        if sum(1 for n in names if n in _MONTHS) >= 12:
            cs = []
            for w in ln:
                key = w['text'].strip().lower()[:3]
                if key in _MONTHS:
                    cs.append(((w['x0'] + w['x1']) / 2, _MONTHS[key]))
                elif w['text'].strip().upper() == 'YTD':
                    cs.append(((w['x0'] + w['x1']) / 2, 'YTD'))
            if len([c for c in cs if c[1] != 'YTD']) >= 12:
                header_i, centers = i, cs
                break
    if header_i is None:
        return {}

    # label = heading above the grid that says "Monthly Returns"
    label = ''
    for ln in lines[:header_i]:
        txt = ' '.join(w['text'] for w in ln)
        if 'Monthly Returns' in txt:
            label = txt.split('Monthly Returns')[0].strip()

    cxs = [c[0] for c in centers]
    clab = [c[1] for c in centers]
    out = {}
    for ln in lines[header_i + 1:]:
        if not ln or not _YEAR_RE.fullmatch(ln[0]['text']):
            continue
        # only the label-less layout: token right after the year must be a value
        if len(ln) < 2 or _token_value(ln[1]['text'])[0] is False:
            continue
        year = int(ln[0]['text'])
        for w in ln[1:]:
            ok, v = _token_value(w['text'])
            if not ok or v is None:
                continue
            xc = (w['x0'] + w['x1']) / 2
            j = min(range(len(cxs)), key=lambda k: abs(cxs[k] - xc))
            if clab[j] == 'YTD':
                continue
            out.setdefault(label or 'Returns', []).append((year, clab[j], v))
    return out


def _cumulative_return(df):
    """Geometric cumulative return of a (year, month, ret%) frame."""
    return float((1 + df['ret'] / 100.0).prod() - 1)


# Index/benchmark names that should never be mistaken for the fund's own track.
_BENCHMARK_KEYWORDS = (
    'HFRI', 'HFRX', 'HFR ', 'S&P', 'SP500', 'MSCI', 'RUSSELL', 'BLOOMBERG',
    'BARCLAYS', 'BBG', ' AGG', 'ACWI', 'INDEX', 'TREASURY', 'NASDAQ', 'DOW ',
    'EUREKAHEDGE', 'CISDM', 'CREDIT SUISSE', 'BENCHMARK', 'HIGH YIELD', 'US HY',
)


def _is_benchmark(label):
    L = ' ' + label.upper() + ' '
    return any(k in L for k in _BENCHMARK_KEYWORDS)


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
                tables = []
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
            # Also parse the page's raw text: catches decks whose table cells
            # merge all month columns, and scrubs watermark contamination.
            try:
                page_text = page.extract_text() or ''
            except Exception:
                page_text = ''
            for label, recs in _parse_text_grid(page_text).items():
                df = _dedupe(recs)
                if len(df) >= 6:
                    candidates.append({
                        'label': label.strip()[:60] or 'Returns',
                        'df': df,
                        'n': len(df),
                        'page': pno,
                    })
            # And the positional parser for label-less, column-aligned grids.
            try:
                pos = _parse_positional(page)
            except Exception:
                pos = {}
            for label, recs in pos.items():
                df = _dedupe(recs)
                if len(df) >= 6:
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

    # Confusing decks list several share classes (and benchmarks). Ingest ONE
    # record: drop benchmark-looking series, then keep the single lowest-return
    # share class (the most conservative, highest-fee net stream).
    if len(candidates) > 1:
        n_before = len(candidates)
        funds = [c for c in candidates if not _is_benchmark(c['label'])]
        pool = funds if funds else candidates
        chosen = min(pool, key=lambda c: _cumulative_return(c['df']))
        if n_before > 1:
            warnings.append(
                f"{n_before} return series found; ingested the lowest-return one "
                f"(\"{chosen['label']}\") and discarded the rest.")
        candidates = [chosen]

    if not candidates:
        warnings.append('No monthly-return grid was found in this PDF. The table '
                        'may be an image, or in a layout the parser can\'t read.')
    return candidates, warnings
