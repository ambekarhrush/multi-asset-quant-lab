"""Investable universe: liquid US-listed ETFs as proxies for futures exposures.

Cost assumptions are one-way, in bps of traded notional (spread + impact for
institutional-size clips, conservative for the less liquid funds).
"""

MULTI_ASSET = {
    # ticker: (asset class, description)
    "SPY": ("Equity", "US large cap"),
    "IWM": ("Equity", "US small cap"),
    "EFA": ("Equity", "Developed ex-US"),
    "EEM": ("Equity", "Emerging markets"),
    "SHY": ("Rates", "UST 1-3y"),
    "IEF": ("Rates", "UST 7-10y"),
    "TLT": ("Rates", "UST 20y+"),
    "TIP": ("Rates", "US TIPS"),
    "LQD": ("Credit", "US IG corporate"),
    "HYG": ("Credit", "US high yield"),
    "GLD": ("Commodity", "Gold"),
    "DBC": ("Commodity", "Broad commodities"),
}

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

COST_BPS = {
    "SPY": 1, "IWM": 2, "EFA": 2, "EEM": 3, "SHY": 1, "IEF": 1, "TLT": 1, "TIP": 2,
    "LQD": 2, "HYG": 3, "GLD": 2, "DBC": 5,
    **{s: 2 for s in SECTORS},
}

ASSET_CLASS = {k: v[0] for k, v in MULTI_ASSET.items()}

# FX carry universe: (FRED spot id, quote convention, FRED 3m rate id)
# convention "USDper" means series is USD per 1 unit of foreign ccy; "perUSD" the inverse.
FX = {
    "EUR": ("DEXUSEU", "USDper", "IR3TIB01EZM156N"),
    "GBP": ("DEXUSUK", "USDper", "IR3TIB01GBM156N"),
    "AUD": ("DEXUSAL", "USDper", "IR3TIB01AUM156N"),
    "NZD": ("DEXUSNZ", "USDper", "IR3TIB01NZM156N"),
    "JPY": ("DEXJPUS", "perUSD", "IR3TIB01JPM156N"),
    "CHF": ("DEXSZUS", "perUSD", "IR3TIB01CHM156N"),
    "CAD": ("DEXCAUS", "perUSD", "IR3TIB01CAM156N"),
    "SEK": ("DEXSDUS", "perUSD", "IR3TIB01SEM156N"),
    "NOK": ("DEXNOUS", "perUSD", "IR3TIB01NOM156N"),
}
USD_RATE = "IR3TIB01USM156N"
