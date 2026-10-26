"""Volatility-based equity timing signals (daily).

  term_structure : VIX / VIX3M. Above 1 (backwardation) = near-term stress priced
                   above longer-dated -> de-risk. Below 1 (contango) = normal.
  vrp            : variance risk premium = VIX^2/100^2 - trailing realised variance.
                   High VRP has predicted higher equity returns
                   (Bollerslev, Tauchen & Zhou 2009).
  vol_target     : scale exposure by target / forecast vol (Moreira & Muir 2017).

All signals use data observed at close t and are applied to the t+1 return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def term_structure_signal(vix: pd.Series, vix3m: pd.Series, smooth: int = 3, threshold: float = 1.0) -> pd.Series:
    ratio = (vix / vix3m).rolling(smooth).mean()
    # graded: full risk in contango, linear cut to zero between 1.0 and 1.10
    return (1 - ((ratio - threshold) / 0.10).clip(0, 1)).rename("term_structure")


def vrp_signal(vix: pd.Series, spx_ret: pd.Series, window: int = 21, z_window: int = 504) -> pd.Series:
    implied_var = (vix / 100) ** 2
    realised_var = spx_ret.rolling(window).var() * 252
    vrp = implied_var - realised_var
    z = (vrp - vrp.rolling(z_window, min_periods=252).mean()) / vrp.rolling(z_window, min_periods=252).std()
    # map to exposure in [0.25, 1.5]: lean in when insurance is expensive relative to realised
    return (1 + 0.5 * np.tanh(z.clip(-3, 3))).clip(0.25, 1.5).rename("vrp")


def vol_target_signal(spx_ret: pd.Series, target: float = 0.15, halflife: int = 20, cap: float = 1.5) -> pd.Series:
    fvol = spx_ret.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(252)
    return (target / fvol).clip(upper=cap).rename("vol_target")
