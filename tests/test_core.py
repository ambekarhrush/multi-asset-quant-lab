"""Tests for the pieces where a silent bug would invalidate results."""
import numpy as np
import pandas as pd
import pytest

from quantlab.core import backtest as bt, metrics
from quantlab.overlays import options as op
from quantlab.premia import factors as f


def _idx(n=60):
    return pd.date_range("2010-01-31", periods=n, freq="ME")


def test_no_lookahead_weights_earn_next_period():
    idx = _idx(4)
    r = pd.DataFrame({"A": [0.0, 0.10, -0.05, 0.02]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0]}, index=idx)  # hold only over period 2
    res = bt.run(w, r, cost_bps=0)
    assert res.gross.iloc[1] == pytest.approx(0.10)
    assert res.gross.iloc[2] == pytest.approx(0.0)


def test_costs_charged_on_turnover():
    idx = _idx(3)
    r = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=idx)
    w = pd.DataFrame({"A": [1.0, -1.0, -1.0]}, index=idx)
    res = bt.run(w, r, cost_bps=10)
    assert res.costs.iloc[0] == pytest.approx(1.0 * 10 / 1e4)
    assert res.costs.iloc[1] == pytest.approx(2.0 * 10 / 1e4)


def test_rank_weights_dollar_neutral():
    s = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 6)), index=_idx(10))
    w = f.rank_weights(s)
    assert np.allclose(w.sum(axis=1), 0)
    assert np.allclose(w.abs().sum(axis=1), 2)


def test_risk_budget_hits_budget():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(200, 4))
    cov = np.cov(A.T) + np.diag([0.5, 0.1, 0.2, 0.05])
    b = np.array([0.4, 0.3, 0.2, 0.1])
    w = f.risk_budget_weights(cov, b)
    assert np.allclose(f.risk_contributions(w, cov), b, atol=1e-6)


def test_put_call_parity():
    S, K, T, r, q, s = 100.0, 95.0, 0.25, 0.03, 0.015, 0.2
    c = op.bs_price(S, K, T, r, q, s, "call")
    p = op.bs_price(S, K, T, r, q, s, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), rel=1e-9)


def test_skew_puts_richer_than_calls():
    put_v = op.skew_vol(20, 100, 95, 1 / 12, 0.03)
    call_v = op.skew_vol(20, 100, 105, 1 / 12, 0.03)
    assert put_v > 0.18 > call_v


def test_deflated_sharpe_penalises_trials():
    r = pd.Series(np.random.default_rng(2).normal(0.005, 0.03, 240))
    assert metrics.deflated_sharpe(r, 12, 100) < metrics.deflated_sharpe(r, 12, 2)


def test_max_drawdown():
    r = pd.Series([0.1, -0.5, 0.2])
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)
