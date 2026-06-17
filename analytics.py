"""
analytics.py — Meatgrinder Analytics Engine
Pure functions, no Streamlit dependencies. All calculations match the HTML versions exactly.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Optional


# ─── REFERENCE DATA ──────────────────────────────────────────────────────────

# TB3MS monthly rates (% per annum) — Source: FRED
RF = {
    '2005-7': 3.05, '2005-8': 3.33, '2005-9': 3.29, '2005-10': 3.57, '2005-11': 3.77, '2005-12': 3.93,
    '2006-1': 4.18, '2006-2': 4.44, '2006-3': 4.51, '2006-4': 4.60, '2006-5': 4.74, '2006-6': 4.79,
    '2006-7': 4.97, '2006-8': 4.99, '2006-9': 4.87, '2006-10': 4.99, '2006-11': 4.97, '2006-12': 4.89,
    '2007-1': 5.01, '2007-2': 5.03, '2007-3': 4.90, '2007-4': 4.87, '2007-5': 4.82, '2007-6': 4.72,
    '2007-7': 4.73, '2007-8': 4.01, '2007-9': 3.65, '2007-10': 3.80, '2007-11': 3.18, '2007-12': 2.92,
    '2008-1': 2.64, '2008-2': 1.83, '2008-3': 0.64, '2008-4': 0.89, '2008-5': 1.37, '2008-6': 1.47,
    '2008-7': 1.46, '2008-8': 1.53, '2008-9': 0.72, '2008-10': 0.36, '2008-11': 0.02, '2008-12': 0.01,
    '2009-1': 0.11, '2009-2': 0.25, '2009-3': 0.19, '2009-4': 0.15, '2009-5': 0.17, '2009-6': 0.19,
    '2009-7': 0.19, '2009-8': 0.16, '2009-9': 0.13, '2009-10': 0.08, '2009-11': 0.07, '2009-12': 0.06,
    '2010-1': 0.07, '2010-2': 0.12, '2010-3': 0.14, '2010-4': 0.16, '2010-5': 0.14, '2010-6': 0.17,
    '2010-7': 0.15, '2010-8': 0.15, '2010-9': 0.15, '2010-10': 0.13, '2010-11': 0.18, '2010-12': 0.13,
    '2011-1': 0.16, '2011-2': 0.13, '2011-3': 0.08, '2011-4': 0.06, '2011-5': 0.05, '2011-6': 0.02,
    '2011-7': 0.04, '2011-8': 0.01, '2011-9': 0.02, '2011-10': 0.02, '2011-11': 0.01, '2011-12': 0.02,
    '2012-1': 0.04, '2012-2': 0.08, '2012-3': 0.07, '2012-4': 0.10, '2012-5': 0.09, '2012-6': 0.09,
    '2012-7': 0.10, '2012-8': 0.10, '2012-9': 0.10, '2012-10': 0.12, '2012-11': 0.11, '2012-12': 0.07,
    '2013-1': 0.07, '2013-2': 0.12, '2013-3': 0.09, '2013-4': 0.07, '2013-5': 0.04, '2013-6': 0.05,
    '2013-7': 0.04, '2013-8': 0.04, '2013-9': 0.02, '2013-10': 0.04, '2013-11': 0.07, '2013-12': 0.07,
    '2014-1': 0.04, '2014-2': 0.04, '2014-3': 0.04, '2014-4': 0.03, '2014-5': 0.03, '2014-6': 0.04,
    '2014-7': 0.03, '2014-8': 0.03, '2014-9': 0.02, '2014-10': 0.02, '2014-11': 0.03, '2014-12': 0.03,
    '2015-1': 0.03, '2015-2': 0.03, '2015-3': 0.03, '2015-4': 0.01, '2015-5': 0.01, '2015-6': 0.02,
    '2015-7': 0.05, '2015-8': 0.08, '2015-9': 0.02, '2015-10': 0.01, '2015-11': 0.14, '2015-12': 0.24,
    '2016-1': 0.31, '2016-2': 0.32, '2016-3': 0.33, '2016-4': 0.22, '2016-5': 0.28, '2016-6': 0.27,
    '2016-7': 0.31, '2016-8': 0.31, '2016-9': 0.34, '2016-10': 0.34, '2016-11': 0.42, '2016-12': 0.52,
    '2017-1': 0.52, '2017-2': 0.53, '2017-3': 0.64, '2017-4': 0.76, '2017-5': 0.89, '2017-6': 0.98,
    '2017-7': 1.07, '2017-8': 1.01, '2017-9': 1.03, '2017-10': 1.07, '2017-11': 1.23, '2017-12': 1.32,
    '2018-1': 1.41, '2018-2': 1.57, '2018-3': 1.70, '2018-4': 1.76, '2018-5': 1.86, '2018-6': 1.90,
    '2018-7': 1.96, '2018-8': 2.03, '2018-9': 2.13, '2018-10': 2.25, '2018-11': 2.33, '2018-12': 2.37,
    '2019-1': 2.37, '2019-2': 2.39, '2019-3': 2.40, '2019-4': 2.38, '2019-5': 2.35, '2019-6': 2.17,
    '2019-7': 2.10, '2019-8': 1.95, '2019-9': 1.89, '2019-10': 1.65, '2019-11': 1.54, '2019-12': 1.54,
    '2020-1': 1.52, '2020-2': 1.52, '2020-3': 0.29, '2020-4': 0.14, '2020-5': 0.13, '2020-6': 0.16,
    '2020-7': 0.13, '2020-8': 0.10, '2020-9': 0.11, '2020-10': 0.10, '2020-11': 0.09, '2020-12': 0.09,
    '2021-1': 0.08, '2021-2': 0.04, '2021-3': 0.03, '2021-4': 0.02, '2021-5': 0.02, '2021-6': 0.04,
    '2021-7': 0.05, '2021-8': 0.05, '2021-9': 0.04, '2021-10': 0.05, '2021-11': 0.05, '2021-12': 0.06,
    '2022-1': 0.15, '2022-2': 0.33, '2022-3': 0.44, '2022-4': 0.76, '2022-5': 0.98, '2022-6': 1.49,
    '2022-7': 2.23, '2022-8': 2.63, '2022-9': 3.13, '2022-10': 3.72, '2022-11': 4.15, '2022-12': 4.25,
    '2023-1': 4.54, '2023-2': 4.65, '2023-3': 4.69, '2023-4': 4.92, '2023-5': 5.14, '2023-6': 5.16,
    '2023-7': 5.25, '2023-8': 5.30, '2023-9': 5.32, '2023-10': 5.34, '2023-11': 5.27, '2023-12': 5.24,
    '2024-1': 5.22, '2024-2': 5.24, '2024-3': 5.24, '2024-4': 5.24, '2024-5': 5.25, '2024-6': 5.24,
    '2024-7': 5.20, '2024-8': 5.05, '2024-9': 4.72, '2024-10': 4.51, '2024-11': 4.42, '2024-12': 4.27,
    '2025-1': 4.21, '2025-2': 4.22, '2025-3': 4.20, '2025-4': 4.21, '2025-5': 4.25, '2025-6': 4.23,
    '2025-7': 4.25, '2025-8': 4.12, '2025-9': 3.92, '2025-10': 3.82, '2025-11': 3.78, '2025-12': 3.75,
}

# MSCI World Hedged USD monthly returns (%)
RAW_MSCI = [
    (2005,7,3.8),(2005,8,0.1),(2005,9,3.3),(2005,10,-1.9),(2005,11,4.2),(2005,12,2.2),
    (2006,1,3.3),(2006,2,0.4),(2006,3,2.5),(2006,4,1.0),(2006,5,-4.3),(2006,6,0.6),(2006,7,0.7),(2006,8,2.7),(2006,9,1.9),(2006,10,3.2),(2006,11,1.2),(2006,12,2.7),
    (2007,1,1.9),(2007,2,-1.1),(2007,3,1.6),(2007,4,3.6),(2007,5,3.4),(2007,6,-0.9),(2007,7,-3.0),(2007,8,0.0),(2007,9,3.1),(2007,10,2.2),(2007,11,-4.3),(2007,12,-0.8),
    (2008,1,-8.4),(2008,2,-1.7),(2008,3,-2.2),(2008,4,6.1),(2008,5,1.6),(2008,6,-8.3),(2008,7,-1.9),(2008,8,1.1),(2008,9,-10.7),(2008,10,-15.8),(2008,11,-6.1),(2008,12,1.1),
    (2009,1,-7.0),(2009,2,-8.8),(2009,3,6.4),(2009,4,10.4),(2009,5,5.8),(2009,6,-0.1),(2009,7,7.5),(2009,8,3.7),(2009,9,3.1),(2009,10,-2.3),(2009,11,3.1),(2009,12,3.6),
    (2010,1,-3.6),(2010,2,1.9),(2010,3,6.5),(2010,4,0.2),(2010,5,-7.4),(2010,6,-4.2),(2010,7,5.9),(2010,8,-3.4),(2010,9,7.1),(2010,10,2.8),(2010,11,-0.3),(2010,12,5.7),
    (2011,1,1.9),(2011,2,2.9),(2011,3,-1.3),(2011,4,2.3),(2011,5,-1.3),(2011,6,-1.6),(2011,7,-2.8),(2011,8,-6.8),(2011,9,-6.0),(2011,10,8.7),(2011,11,-1.3),(2011,12,0.6),
    (2012,1,4.3),(2012,2,4.7),(2012,3,1.7),(2012,4,-1.6),(2012,5,-6.6),(2012,6,4.3),(2012,7,1.3),(2012,8,2.0),(2012,9,2.2),(2012,10,-0.5),(2012,11,1.4),(2012,12,1.9),
    (2013,1,5.3),(2013,2,1.5),(2013,3,2.7),(2013,4,2.8),(2013,5,1.4),(2013,6,-2.4),(2013,7,4.7),(2013,8,-2.1),(2013,9,3.8),(2013,10,3.9),(2013,11,2.1),(2013,12,2.1),
    (2014,1,-3.2),(2014,2,4.2),(2014,3,0.2),(2014,4,0.7),(2014,5,2.2),(2014,6,1.4),(2014,7,-0.8),(2014,8,2.6),(2014,9,-1.0),(2014,10,1.1),(2014,11,2.8),(2014,12,-0.8),
    (2015,1,-0.6),(2015,2,5.9),(2015,3,-0.4),(2015,4,1.0),(2015,5,1.3),(2015,6,-3.0),(2015,7,2.5),(2015,8,-6.7),(2015,9,-3.5),(2015,10,7.9),(2015,11,0.6),(2015,12,-2.1),
    (2016,1,-5.4),(2016,2,-1.5),(2016,3,5.3),(2016,4,0.9),(2016,5,1.8),(2016,6,-1.3),(2016,7,4.2),(2016,8,0.5),(2016,9,0.2),(2016,10,-0.6),(2016,11,2.6),(2016,12,2.8),
    (2017,1,1.3),(2017,2,3.1),(2017,3,1.0),(2017,4,1.2),(2017,5,1.5),(2017,6,0.1),(2017,7,1.5),(2017,8,0.2),(2017,9,2.4),(2017,10,2.6),(2017,11,1.6),(2017,12,1.2),
    (2018,1,3.8),(2018,2,-3.5),(2018,3,-2.2),(2018,4,2.0),(2018,5,1.3),(2018,6,0.4),(2018,7,3.2),(2018,8,1.4),(2018,9,0.8),(2018,10,-6.7),(2018,11,1.2),(2018,12,-7.8),
    (2019,1,7.4),(2019,2,3.4),(2019,3,1.7),(2019,4,3.8),(2019,5,-5.6),(2019,6,6.0),(2019,7,1.2),(2019,8,-1.9),(2019,9,2.4),(2019,10,2.0),(2019,11,3.2),(2019,12,2.4),
    (2020,1,-0.2),(2020,2,-8.0),(2020,3,-12.6),(2020,4,10.6),(2020,5,4.7),(2020,6,2.4),(2020,7,3.4),(2020,8,6.3),(2020,9,-2.9),(2020,10,-3.0),(2020,11,12.1),(2020,12,3.6),
    (2021,1,-0.8),(2021,2,2.7),(2021,3,4.2),(2021,4,4.1),(2021,5,1.0),(2021,6,2.3),(2021,7,1.7),(2021,8,2.7),(2021,9,-3.7),(2021,10,5.5),(2021,11,-1.4),(2021,12,4.0),
    (2022,1,-4.9),(2022,2,-2.6),(2022,3,3.1),(2022,4,-6.8),(2022,5,-0.2),(2022,6,-7.7),(2022,7,8.0),(2022,8,-3.4),(2022,9,-8.2),(2022,10,7.2),(2022,11,5.8),(2022,12,-5.0),
    (2023,1,6.6),(2023,2,-1.5),(2023,3,2.6),(2023,4,1.7),(2023,5,-0.1),(2023,6,5.8),(2023,7,3.0),(2023,8,-1.7),(2023,9,-3.6),(2023,10,-2.6),(2023,11,8.4),(2023,12,4.2),
    (2024,1,1.8),(2024,2,4.7),(2024,3,3.4),(2024,4,-3.2),(2024,5,4.1),(2024,6,2.4),(2024,7,1.3),(2024,8,1.9),(2024,9,1.6),(2024,10,-0.8),(2024,11,4.9),(2024,12,-1.9),
    (2025,1,3.5),(2025,2,-0.9),(2025,3,-5.0),(2025,4,-0.4),(2025,5,6.0),(2025,6,3.8),(2025,7,2.1),(2025,8,2.1),(2025,9,3.3),(2025,10,2.6),(2025,11,0.3),(2025,12,0.6),
]

# Bloomberg Global Agg monthly returns (%)
RAW_AGG = [
    (2005,7,-0.8),(2005,8,1.6),(2005,9,-1.6),(2005,10,-1.5),(2005,11,-0.7),(2005,12,1.0),
    (2006,1,1.3),(2006,2,-0.4),(2006,3,-1.0),(2006,4,1.9),(2006,5,1.3),(2006,6,-0.8),(2006,7,1.0),(2006,8,1.2),(2006,9,0.0),(2006,10,1.0),(2006,11,2.4),(2006,12,-1.3),
    (2007,1,-1.0),(2007,2,2.1),(2007,3,0.2),(2007,4,1.1),(2007,5,-1.6),(2007,6,-0.4),(2007,7,2.0),(2007,8,1.2),(2007,9,2.2),(2007,10,1.6),(2007,11,1.9),(2007,12,-0.3),
    (2008,1,2.8),(2008,2,1.7),(2008,3,2.0),(2008,4,-1.9),(2008,5,-1.1),(2008,6,0.1),(2008,7,0.1),(2008,8,-1.7),(2008,9,-2.4),(2008,10,-3.7),(2008,11,2.9),(2008,12,6.2),
    (2009,1,-3.3),(2009,2,-2.2),(2009,3,2.3),(2009,4,0.9),(2009,5,3.6),(2009,6,0.4),(2009,7,2.2),(2009,8,1.8),(2009,9,2.1),(2009,10,0.5),(2009,11,2.5),(2009,12,-3.8),
    (2010,1,0.4),(2010,2,0.1),(2010,3,-0.8),(2010,4,0.0),(2010,5,-1.6),(2010,6,1.5),(2010,7,3.4),(2010,8,1.4),(2010,9,2.3),(2010,10,1.3),(2010,11,-3.8),(2010,12,1.3),
    (2011,1,0.2),(2011,2,0.6),(2011,3,0.5),(2011,4,3.1),(2011,5,-0.1),(2011,6,0.1),(2011,7,2.1),(2011,8,1.3),(2011,9,-2.3),(2011,10,1.3),(2011,11,-1.7),(2011,12,0.7),
    (2012,1,1.7),(2012,2,-0.1),(2012,3,-0.7),(2012,4,1.2),(2012,5,-1.0),(2012,6,0.5),(2012,7,1.2),(2012,8,0.9),(2012,9,1.2),(2012,10,-0.1),(2012,11,0.0),(2012,12,-0.3),
    (2013,1,-0.9),(2013,2,-0.9),(2013,3,-0.3),(2013,4,1.4),(2013,5,-3.0),(2013,6,-1.2),(2013,7,1.3),(2013,8,-0.5),(2013,9,2.1),(2013,10,1.0),(2013,11,-0.8),(2013,12,-0.6),
    (2014,1,1.1),(2014,2,1.4),(2014,3,-0.1),(2014,4,1.1),(2014,5,0.6),(2014,6,0.7),(2014,7,-0.9),(2014,8,0.5),(2014,9,-2.8),(2014,10,0.0),(2014,11,-0.4),(2014,12,-0.7),
    (2015,1,-0.2),(2015,2,-0.8),(2015,3,-1.0),(2015,4,1.1),(2015,5,-1.8),(2015,6,-0.4),(2015,7,0.2),(2015,8,0.1),(2015,9,0.5),(2015,10,0.2),(2015,11,-1.7),(2015,12,0.5),
    (2016,1,0.9),(2016,2,2.2),(2016,3,2.7),(2016,4,1.3),(2016,5,-1.3),(2016,6,2.9),(2016,7,0.8),(2016,8,-0.5),(2016,9,0.6),(2016,10,-2.8),(2016,11,-4.0),(2016,12,-0.5),
    (2017,1,1.1),(2017,2,0.5),(2017,3,0.2),(2017,4,1.1),(2017,5,1.5),(2017,6,-0.1),(2017,7,1.7),(2017,8,1.0),(2017,9,-0.9),(2017,10,-0.4),(2017,11,1.1),(2017,12,0.3),
    (2018,1,1.2),(2018,2,-0.9),(2018,3,1.1),(2018,4,-1.6),(2018,5,-0.8),(2018,6,-0.4),(2018,7,-0.2),(2018,8,0.1),(2018,9,-0.9),(2018,10,-1.1),(2018,11,0.3),(2018,12,2.0),
    (2019,1,1.5),(2019,2,-0.6),(2019,3,1.3),(2019,4,-0.3),(2019,5,1.4),(2019,6,2.2),(2019,7,-0.3),(2019,8,2.0),(2019,9,-1.0),(2019,10,0.7),(2019,11,-0.8),(2019,12,0.6),
    (2020,1,1.3),(2020,2,0.7),(2020,3,-2.2),(2020,4,2.0),(2020,5,0.4),(2020,6,0.9),(2020,7,3.2),(2020,8,-0.2),(2020,9,-0.4),(2020,10,0.1),(2020,11,1.8),(2020,12,1.3),
    (2021,1,-0.9),(2021,2,-1.7),(2021,3,-1.9),(2021,4,1.3),(2021,5,0.9),(2021,6,-0.9),(2021,7,1.3),(2021,8,-0.4),(2021,9,-1.8),(2021,10,-0.2),(2021,11,-0.3),(2021,12,-0.1),
    (2022,1,-2.0),(2022,2,-1.2),(2022,3,-3.0),(2022,4,-5.5),(2022,5,0.3),(2022,6,-3.2),(2022,7,2.1),(2022,8,-3.9),(2022,9,-5.1),(2022,10,-0.7),(2022,11,4.7),(2022,12,0.5),
    (2023,1,3.3),(2023,2,-3.3),(2023,3,3.2),(2023,4,0.4),(2023,5,-2.0),(2023,6,0.0),(2023,7,0.7),(2023,8,-1.4),(2023,9,-2.9),(2023,10,-1.2),(2023,11,5.0),(2023,12,4.2),
    (2024,1,-1.4),(2024,2,-1.3),(2024,3,0.6),(2024,4,-2.5),(2024,5,1.3),(2024,6,0.1),(2024,7,2.8),(2024,8,2.4),(2024,9,1.7),(2024,10,-3.4),(2024,11,0.3),(2024,12,-2.1),
    (2025,1,0.6),(2025,2,1.4),(2025,3,0.6),(2025,4,2.9),(2025,5,-0.4),(2025,6,1.9),(2025,7,-1.5),(2025,8,1.5),(2025,9,0.7),(2025,10,-0.3),(2025,11,0.2),(2025,12,0.3),
]

# Macro events for stress-test table
MACRO_EVENTS = [
    {'name': 'Quant Quake',             'period': 'Aug 2007',       'months': [(2007,8)]},
    {'name': 'Lehman / GFC Peak',       'period': 'Sep–Oct 2008',   'months': [(2008,9),(2008,10)]},
    {'name': 'GFC Full Crisis',         'period': 'Jul–Dec 2008',   'months': [(2008,7),(2008,8),(2008,9),(2008,10),(2008,11),(2008,12)]},
    {'name': 'Market Rebound 2009',     'period': 'Mar–Jun 2009',   'months': [(2009,3),(2009,4),(2009,5),(2009,6)]},
    {'name': 'Flash Crash',             'period': 'May 2010',       'months': [(2010,5)]},
    {'name': 'US Debt Ceiling Crisis',  'period': 'Jul–Aug 2011',   'months': [(2011,7),(2011,8)]},
    {'name': 'Taper Tantrum',           'period': 'May–Jun 2013',   'months': [(2013,5),(2013,6)]},
    {'name': 'China Volatility',        'period': 'Aug–Sep 2015',   'months': [(2015,8),(2015,9)]},
    {'name': 'Volmageddon',             'period': 'Feb 2018',       'months': [(2018,2)]},
    {'name': 'COVID Crash',             'period': 'Feb–Mar 2020',   'months': [(2020,2),(2020,3)]},
    {'name': 'COVID Recovery',          'period': 'Apr–Jun 2020',   'months': [(2020,4),(2020,5),(2020,6)]},
    {'name': 'Pfizer Vaccine Rally',    'period': 'Nov 2020',       'months': [(2020,11)]},
    {'name': 'Reddit / Meme Stocks',    'period': 'Jan 2021',       'months': [(2021,1)]},
    {'name': 'Russia–Ukraine Invasion', 'period': 'Feb–Mar 2022',   'months': [(2022,2),(2022,3)]},
    {'name': 'Rate Shock Peak',         'period': 'Apr–Jun 2022',   'months': [(2022,4),(2022,5),(2022,6)]},
    {'name': '2022 Full Bear Market',   'period': 'Jan–Sep 2022',   'months': [(2022,1),(2022,2),(2022,3),(2022,4),(2022,5),(2022,6),(2022,7),(2022,8),(2022,9)]},
    {'name': 'SVB / Banking Crisis',    'period': 'Mar 2023',       'months': [(2023,3)]},
    {'name': 'Liberation Day Tariffs',  'period': 'Apr 2025',       'months': [(2025,4)]},
]

# Build reference DataFrames
MSCI_DF = pd.DataFrame(RAW_MSCI, columns=['year', 'month', 'ret'])
AGG_DF  = pd.DataFrame(RAW_AGG,  columns=['year', 'month', 'ret'])


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def rf_monthly(year: int, month: int) -> float:
    """Return monthly risk-free rate (TB3MS annual % / 12)."""
    key = f'{year}-{month}'
    return RF.get(key, 0.0) / 12.0


def build_excess_returns(df: pd.DataFrame) -> np.ndarray:
    """Return array of monthly excess returns (ret - rf) for a {year,month,ret} DataFrame."""
    return np.array([
        row.ret - rf_monthly(row.year, row.month)
        for row in df.itertuples()
    ])


# ─── CORE STATISTICS ──────────────────────────────────────────────────────────

def geo_return(rets: np.ndarray, freq: int = 12) -> float:
    """Annualised geometric return from monthly percent returns."""
    n = len(rets)
    if n == 0:
        return np.nan
    growth = np.prod(1 + rets / 100)
    return (growth ** (freq / n) - 1) * 100


def ann_vol(rets: np.ndarray, freq: int = 12) -> float:
    """Annualised volatility of monthly percent returns."""
    if len(rets) < 2:
        return np.nan
    return float(np.std(rets, ddof=1) * np.sqrt(freq))


def sharpe_with_rf(df: pd.DataFrame) -> float:
    """Sharpe ratio using actual monthly TB3MS as risk-free rate."""
    ex = build_excess_returns(df)
    if len(ex) < 2:
        return np.nan
    v = np.std(ex, ddof=1)
    if v == 0:
        return 0.0
    return float(np.mean(ex) * np.sqrt(12) / v)


def sortino_with_rf(df: pd.DataFrame) -> float:
    """Sortino ratio using actual monthly TB3MS as MAR."""
    ex = build_excess_returns(df)
    if len(ex) < 2:
        return np.nan
    downside = ex[ex < 0]
    if len(downside) == 0:
        return np.inf
    dv = np.sqrt(np.mean(downside**2)) * np.sqrt(12)
    if dv == 0:
        return np.inf
    return float(np.mean(ex) * 12 / dv)


def max_drawdown(rets: np.ndarray) -> float:
    """Maximum drawdown as negative percent."""
    wealth = np.cumprod(1 + rets / 100)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak * 100
    return float(np.min(dd)) if len(dd) > 0 else 0.0


def drawdown_series(rets: np.ndarray) -> np.ndarray:
    """Full drawdown series as percent from peak."""
    wealth = np.cumprod(1 + rets / 100)
    peak = np.maximum.accumulate(wealth)
    return (wealth - peak) / peak * 100


def calmar(rets: np.ndarray) -> float:
    """Calmar ratio = annualised return / abs(max drawdown)."""
    md = abs(max_drawdown(rets))
    if md == 0:
        return np.nan
    return geo_return(rets) / md


def skewness(rets: np.ndarray) -> float:
    """Sample skewness (Fisher corrected)."""
    n = len(rets)
    if n < 3:
        return np.nan
    m = np.mean(rets)
    s = np.std(rets, ddof=1)
    if s == 0:
        return 0.0
    return float((n / ((n-1) * (n-2))) * np.sum(((rets - m) / s) ** 3))


def excess_kurtosis(rets: np.ndarray) -> float:
    """Sample excess kurtosis (Fisher corrected)."""
    n = len(rets)
    if n < 4:
        return np.nan
    m = np.mean(rets)
    s = np.std(rets, ddof=1)
    if s == 0:
        return 0.0
    s4 = np.sum(((rets - m) / s) ** 4)
    return float((n * (n+1)) / ((n-1)*(n-2)*(n-3)) * s4 - 3*(n-1)**2 / ((n-2)*(n-3)))


def jarque_bera(rets: np.ndarray) -> tuple[float, float]:
    """Jarque-Bera test statistic and p-value."""
    n = len(rets)
    s = skewness(rets)
    k = excess_kurtosis(rets)
    stat = n / 6 * (s**2 + k**2 / 4)
    pval = np.exp(-stat / 2)   # approximate, matches HTML version
    return float(stat), float(pval)


def compute_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where |ret - mean| > 3*std."""
    rets = df['ret'].values
    m = np.mean(rets)
    s = np.std(rets, ddof=1)
    return df[np.abs(rets - m) > 3 * s].copy()


# ─── NORMALITY TESTS ──────────────────────────────────────────────────────────

def normality_tests(rets: np.ndarray) -> dict:
    """Run Jarque-Bera, Shapiro-Wilk, Anderson-Darling, and KS tests."""
    results = {}

    # Jarque-Bera
    jb_stat, jb_p = jarque_bera(rets)
    results['jb'] = {'stat': jb_stat, 'pval': jb_p, 'name': 'Jarque-Bera'}

    # Shapiro-Wilk
    if len(rets) >= 3:
        sw_stat, sw_p = scipy_stats.shapiro(rets)
        results['sw'] = {'stat': float(sw_stat), 'pval': float(sw_p), 'name': 'Shapiro-Wilk'}

    # Anderson-Darling (use significance_level method to avoid future deprecation)
    try:
        ad_result = scipy_stats.anderson(rets, dist='norm')
        ad_stat = float(ad_result.statistic)
        # critical_values[2] is the 5% significance level
        ad_crit = float(ad_result.critical_values[2])
    except Exception:
        ad_stat, ad_crit = np.nan, np.nan
    results['ad'] = {
        'stat': ad_stat,
        'critical': ad_crit,
        'reject': ad_stat > ad_crit if not np.isnan(ad_stat) else False,
        'name': 'Anderson-Darling'
    }

    # Kolmogorov-Smirnov
    ks_stat, ks_p = scipy_stats.kstest(rets, 'norm',
                                        args=(np.mean(rets), np.std(rets, ddof=1)))
    results['ks'] = {'stat': float(ks_stat), 'pval': float(ks_p), 'name': 'Kolmogorov-Smirnov'}

    return results


# ─── DRAWDOWN EPISODES ────────────────────────────────────────────────────────

def top_drawdowns(df: pd.DataFrame, n: int = 7) -> list[dict]:
    """Find top N drawdown episodes with peak/trough/recovery dates."""
    rets = df['ret'].values
    N = len(rets)

    # Build wealth index
    wealth = np.concatenate([[1.0], np.cumprod(1 + rets / 100)])

    episodes = []
    pk = 1.0
    pk_i = 0

    def close_episode(end_i):
        if end_i <= pk_i + 1:
            return
        tr = wealth[pk_i]
        tr_i = pk_i
        for j in range(pk_i + 1, end_i):
            if wealth[j] < tr:
                tr = wealth[j]
                tr_i = j
        if tr < pk:
            episodes.append({'dd': (tr/pk - 1)*100, 'pk_i': pk_i, 'tr_i': tr_i, 'rec_i': end_i - 1})

    for i in range(1, N + 1):
        if wealth[i] >= pk:
            close_episode(i)
            pk = wealth[i]
            pk_i = i

    if pk_i < N:
        tr = wealth[pk_i]
        tr_i = pk_i
        for j in range(pk_i + 1, N + 1):
            if wealth[j] < tr:
                tr = wealth[j]
                tr_i = j
        if tr < pk:
            episodes.append({'dd': (tr/pk - 1)*100, 'pk_i': pk_i, 'tr_i': tr_i, 'rec_i': None})

    episodes.sort(key=lambda e: e['dd'])

    MN = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    def fmt_date(idx):
        if idx is None:
            return 'Ongoing'
        if idx == 0:
            return 'Start'
        row = df.iloc[idx - 1]
        return f"{MN[int(row['month'])-1]} {int(row['year'])}"

    result = []
    for ep in episodes[:n]:
        result.append({
            'drawdown': ep['dd'],
            'peak_date': 'Start' if ep['pk_i'] == 0 else fmt_date(ep['pk_i']),
            'trough_date': fmt_date(ep['tr_i']),
            'recovery_date': fmt_date(ep['rec_i']),
            'peak_to_trough': ep['tr_i'] - ep['pk_i'],
            'trough_to_recovery': None if ep['rec_i'] is None else ep['rec_i'] - ep['tr_i'],
            'total_months': None if ep['rec_i'] is None else ep['rec_i'] - ep['pk_i'],
        })
    return result


# ─── REGRESSION ───────────────────────────────────────────────────────────────

def piecewise_beta_regression(fund_df: pd.DataFrame, bm_df: pd.DataFrame) -> Optional[dict]:
    """
    Full regression including piecewise beta, alpha, t-stats, SEs, confidence intervals.
    fund_df and bm_df must have aligned {year, month, ret} rows.
    """
    # Align on common (year, month)
    merged = fund_df.merge(bm_df, on=['year','month'], suffixes=('_fund','_bm'))
    if len(merged) < 5:
        return None

    y = merged['ret_fund'].values
    x = merged['ret_bm'].values
    n = len(x)

    mx, my = np.mean(x), np.mean(y)
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    if sx == 0 or sy == 0:
        return None

    # OLS beta and alpha
    cov = np.sum((x - mx) * (y - my)) / (n - 1)
    varx = np.sum((x - mx)**2) / (n - 1)
    beta = cov / varx
    alpha = my - beta * mx
    corr = cov / (sx * sy)
    r2 = corr**2

    # Regression standard errors and t-stats
    y_hat = alpha + beta * x
    resid = y - y_hat
    sse = np.sum(resid**2)
    mse = sse / (n - 2) if n > 2 else np.nan
    se_beta = np.sqrt(mse / ((n-1) * varx)) if mse is not np.nan else np.nan
    se_alpha = np.sqrt(mse * (1/n + mx**2 / ((n-1)*varx))) if mse is not np.nan else np.nan
    t_beta  = beta / se_beta if se_beta and se_beta > 0 else np.nan
    t_alpha = alpha / se_alpha if se_alpha and se_alpha > 0 else np.nan

    # 95% confidence intervals
    t_crit = scipy_stats.t.ppf(0.975, df=n-2) if n > 2 else np.nan
    alpha_ci = (alpha - t_crit*se_alpha, alpha + t_crit*se_alpha) if not np.isnan(t_crit) else (np.nan, np.nan)
    beta_ci  = (beta  - t_crit*se_beta,  beta  + t_crit*se_beta)  if not np.isnan(t_crit) else (np.nan, np.nan)

    # Piecewise betas
    up_mask = x > 0
    dn_mask = x <= 0

    def piecewise_beta(xp, yp):
        if len(xp) < 3:
            return None
        pmx, pmy = np.mean(xp), np.mean(yp)
        pc = np.sum((xp-pmx)*(yp-pmy)) / (len(xp)-1)
        pv = np.sum((xp-pmx)**2) / (len(xp)-1)
        return 0.0 if pv == 0 else pc/pv

    beta_up = piecewise_beta(x[up_mask], y[up_mask])
    beta_dn = piecewise_beta(x[dn_mask], y[dn_mask])
    convexity = (beta_up - beta_dn) if (beta_up is not None and beta_dn is not None) else None

    return {
        'n': n,
        'beta': beta, 'alpha': alpha, 'corr': corr, 'r2': r2,
        'se_beta': se_beta, 'se_alpha': se_alpha,
        't_beta': t_beta, 't_alpha': t_alpha,
        'alpha_ci': alpha_ci, 'beta_ci': beta_ci,
        'beta_up': beta_up, 'beta_dn': beta_dn, 'convexity': convexity,
        'x': x, 'y': y,
        'up_x': x[up_mask], 'up_y': y[up_mask],
        'dn_x': x[dn_mask], 'dn_y': y[dn_mask],
        'n_up': int(up_mask.sum()), 'n_dn': int(dn_mask.sum()),
    }


# ─── AUTOCORRELATION ──────────────────────────────────────────────────────────

def acf(rets: np.ndarray, max_lag: int = 12) -> list[float]:
    """Return autocorrelation at lags 1..max_lag."""
    n = len(rets)
    m = np.mean(rets)
    v = np.sum((rets - m)**2) / n
    result = []
    for lag in range(1, max_lag + 1):
        cov = np.sum((rets[lag:] - m) * (rets[:n-lag] - m)) / n
        result.append(0.0 if v == 0 else cov / v)
    return result


def acf_conf_band(n: int) -> float:
    """95% confidence band for autocorrelation ≈ 1.96/sqrt(n)."""
    return 1.96 / np.sqrt(n)


# ─── Q-Q DATA ─────────────────────────────────────────────────────────────────

def qq_data(rets: np.ndarray) -> dict:
    """Return sorted data and theoretical quantiles for Q-Q plot."""
    n = len(rets)
    sorted_r = np.sort(rets)
    probs = (np.arange(1, n+1) - 0.5) / n
    theoretical = scipy_stats.norm.ppf(probs, loc=np.mean(rets), scale=np.std(rets, ddof=1))
    return {'data': sorted_r, 'theoretical': theoretical}


# ─── MULTI-PERIOD STATS ───────────────────────────────────────────────────────

def period_stats(df: pd.DataFrame) -> Optional[dict]:
    """Compute all stats for a slice of data."""
    if df is None or len(df) == 0:
        return None
    rets = df['ret'].values
    return {
        'n': len(rets),
        'total_ret': (np.prod(1 + rets/100) - 1) * 100,
        'ann_ret': geo_return(rets),
        'ann_vol': ann_vol(rets),
        'sharpe': sharpe_with_rf(df),
        'sortino': sortino_with_rf(df),
        'max_dd': max_drawdown(rets),
        'calmar': calmar(rets),
        'hit_rate': float(np.sum(rets > 0) / len(rets) * 100),
        'best': float(np.max(rets)),
        'worst': float(np.min(rets)),
        'skew': skewness(rets),
        'exkurt': excess_kurtosis(rets),
    }


# ─── ROLLING METRICS ──────────────────────────────────────────────────────────

def rolling_metrics(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Compute rolling 12-month return, Sharpe, and volatility."""
    records = []
    for i in range(window - 1, len(df)):
        sub = df.iloc[i - window + 1 : i + 1].copy()
        rets = sub['ret'].values
        records.append({
            'year': int(df.iloc[i]['year']),
            'month': int(df.iloc[i]['month']),
            'roll_ret': geo_return(rets),
            'roll_vol': ann_vol(rets),
            'roll_sharpe': sharpe_with_rf(sub),
        })
    return pd.DataFrame(records)


# ─── SEASONALITY ──────────────────────────────────────────────────────────────

def seasonality(df: pd.DataFrame) -> dict:
    """Monthly and quarterly average returns."""
    monthly = df.groupby('month')['ret'].agg(['mean','std','count']).reset_index()
    monthly.columns = ['month', 'mean', 'std', 'count']

    df2 = df.copy()
    df2['quarter'] = ((df2['month'] - 1) // 3) + 1
    quarterly = df2.groupby('quarter')['ret'].agg(['mean','std','count']).reset_index()
    quarterly.columns = ['quarter', 'mean', 'std', 'count']

    return {'monthly': monthly, 'quarterly': quarterly}


# ─── MACRO EVENTS ─────────────────────────────────────────────────────────────

def compound_return(series_df: pd.DataFrame, months: list[tuple]) -> float:
    """Compound return over specified (year, month) tuples."""
    p = 1.0
    for (y, m) in months:
        row = series_df[(series_df['year'] == y) & (series_df['month'] == m)]
        if len(row) > 0:
            p *= (1 + row.iloc[0]['ret'] / 100)
    return (p - 1) * 100


def macro_events_table(fund_df: pd.DataFrame, bm_df: Optional[pd.DataFrame] = None) -> list[dict]:
    """Build macro events stress-test table."""
    rows = []
    for ev in MACRO_EVENTS:
        months = ev['months']
        # Only include event if fund has data for at least one of those months
        has_data = any(
            len(fund_df[(fund_df['year'] == y) & (fund_df['month'] == m)]) > 0
            for (y, m) in months
        )
        if not has_data:
            continue
        fund_ret = compound_return(fund_df, months)
        bm_ret = compound_return(bm_df, months) if bm_df is not None else None
        spread = (fund_ret - bm_ret) if bm_ret is not None else None
        rows.append({
            'name': ev['name'],
            'period': ev['period'],
            'fund_ret': fund_ret,
            'bm_ret': bm_ret,
            'spread': spread,
        })
    return rows


# ─── CSV/EXCEL INGESTION ──────────────────────────────────────────────────────

def parse_uploaded_file(uploaded_file) -> tuple[pd.DataFrame, str]:
    """
    Parse CSV or Excel file containing monthly returns.
    Attempts to auto-detect date column and return column.
    Returns (df with columns year/month/ret, error_message).
    """
    import io

    try:
        name = uploaded_file.name.lower()
        if name.endswith('.csv'):
            raw = pd.read_csv(uploaded_file)
        elif name.endswith(('.xlsx', '.xls')):
            raw = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file type. Please upload CSV or Excel."
    except Exception as e:
        return None, f"Could not read file: {e}"

    if raw.empty:
        return None, "File appears to be empty."

    # Find date column
    date_col = None
    for col in raw.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ['date', 'month', 'period', 'time', 'year']):
            date_col = col
            break
    if date_col is None:
        date_col = raw.columns[0]  # fall back to first column

    # Find return column
    ret_col = None
    for col in raw.columns:
        if col == date_col:
            continue
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ['return', 'ret', 'r', 'perf', 'pnl', 'nav']):
            ret_col = col
            break
    if ret_col is None:
        # Pick first numeric column that isn't the date
        for col in raw.columns:
            if col == date_col:
                continue
            if pd.api.types.is_numeric_dtype(raw[col]):
                ret_col = col
                break
    if ret_col is None:
        return None, "Could not identify a return column."

    # Parse dates
    dates = pd.to_datetime(raw[date_col], infer_datetime_format=True, errors='coerce')
    if dates.isna().all():
        return None, f"Could not parse dates in column '{date_col}'."

    # Parse returns — handle percent strings like "1.23%" or decimals like 0.0123
    rets_raw = raw[ret_col].copy()
    if rets_raw.dtype == object:
        rets_raw = rets_raw.astype(str).str.replace('%','').str.strip()
        rets_raw = pd.to_numeric(rets_raw, errors='coerce')
    else:
        rets_raw = pd.to_numeric(rets_raw, errors='coerce')

    # Auto-detect percent vs decimal: if all absolute values < 1, assume decimal
    non_nan = rets_raw.dropna()
    if len(non_nan) > 0 and (np.abs(non_nan) < 1).all():
        rets_raw = rets_raw * 100

    df = pd.DataFrame({
        'year':  dates.dt.year,
        'month': dates.dt.month,
        'ret':   rets_raw,
    }).dropna().astype({'year': int, 'month': int, 'ret': float})

    df = df.sort_values(['year','month']).reset_index(drop=True)

    if len(df) < 6:
        return None, "Fewer than 6 valid data points found."

    return df, ""
