import math

import numpy as np
import pandas as pd
import pytest

from rosca import engine, metrics
from rosca.contracts import PRESETS, WEEKLY_COLUMNS, RunResult, ShockEvent
from rosca.params import EconParams, PoolParams, SimParams

KEYS = ("structure seed mttr_weeks recovered payday_loans payday_borrowed payday_fees share_used_payday "
        "weeks_in_debt_per_borrower loans_per_borrower cycle_completed attrition miss_rate frozen_weeks "
        "max_reserve_drawdown avg_payout self_coverage_rate hardship_weeks avg_pool_balance seed_drawn_max "
        "survived mttr_censored payday_fees_per_member").split()


def h(*vals):
    return pd.Series(vals, dtype=float)


# ---------------------------------------------------------------- recovery_time
def test_recovered():
    # baseline 1.0, dips at shock week 4, back at week 7 and holds 4 weeks
    assert metrics.recovery_time(h(1, 1, 1, 1, .5, .6, .8, 1, 1, 1, 1, 1), 4) == (3.0, True)


def test_censored():
    assert metrics.recovery_time(h(1, 1, 1, 1, .5, .6, .7, .8, .9, 1, 1), 4) == (7.0, False)


def test_never_dipped():
    assert metrics.recovery_time(h(1, 1, 1, 1, .97, 1, 1, 1), 4) == (0.0, True)


def test_late_dip_counts_from_shock_start():
    # dips 2 weeks after shock, recovers at week 8 → 4 weeks after shock start
    assert metrics.recovery_time(h(1, 1, 1, 1, 1, 1, .5, .5, 1, 1, 1, 1), 4) == (4.0, True)


def test_collapsed_never_recovers():
    w = synth_weekly([False] * 5 + [True] * 7)
    hs = metrics.health_index(w, PoolParams(), 12)
    assert (hs.iloc[5:] == 0).all() and hs.iloc[0] > 0
    assert metrics.recovery_time(hs, 3) == (9.0, False)


def test_shock_at_week_zero():
    # baseline = week 0 only
    assert metrics.recovery_time(h(1, .5, .5, 1, 1, 1, 1, 1), 0) == (3.0, True)
    assert metrics.recovery_time(h(.5, .5, .5, .5), 0) == (0.0, True)


def test_nan_and_bad_input():
    assert metrics.recovery_time(h(1, 1, np.nan, 1, 1, 1, 1), 2) == (1.0, True)  # NaN week counts as dip
    m, rec = metrics.recovery_time(h(np.nan, np.nan, 1, 1), 2)
    assert math.isnan(m) and not rec
    m, rec = metrics.recovery_time(h(1, 1), 5)
    assert math.isnan(m) and not rec


# ---------------------------------------------------------------- health_index
def synth_weekly(collapsed, reserve=0.0, due=600.0, paid=600.0, debt=0, active=12):
    W = len(collapsed)
    w = pd.DataFrame({c: np.zeros(W) for c in WEEKLY_COLUMNS})
    w["week"] = np.arange(W)
    w["collapsed"], w["frozen"] = collapsed, False
    w["reserve"], w["contributions_due"], w["contributions_paid"] = reserve, due, paid
    w["members_in_payday_debt"], w["active_members"] = debt, active
    return w


def test_health_components():
    pool = PoolParams()  # full pot = 12 × 50 × 2 = 1200
    w = synth_weekly([False] * 4, reserve=600.0, paid=300.0, debt=6, active=9)
    # reserve .5, on-time .5, payday 1−.5, retention .75
    assert np.allclose(metrics.health_index(w, pool, 12), (0.5 + 0.5 + 0.5 + 0.75) / 4)
    # no reserve, nothing due → only payday + retention
    w = synth_weekly([False] * 4, reserve=0.0, due=0.0, paid=0.0, debt=0, active=12)
    assert np.allclose(metrics.health_index(w, pool, 12), 1.0)


# ---------------------------------------------------------------- summarize
def test_summarize_fake_run():
    p = SimParams()
    res = engine.run_many(p, ["no_pool", "rigid", "reserve_staking"], PRESETS["Plant layoff"], 3)
    df = metrics.summarize_many(res, p.pool, p.population.cohort_size)
    assert list(df.columns) == KEYS and len(df) == 9
    assert df.attrs["cohort_size"] == 12 and df.attrs["weeks"] == 52
    assert df.mttr_weeks.notna().all() and (df.mttr_censored == ~df.recovered.astype(bool)).all()
    assert (df[df.structure == "no_pool"].miss_rate == 0).all()


def test_summarize_no_shocks_mttr_nan():
    p = SimParams()
    s = metrics.summarize_run(engine.fake_run(p, "rigid", [], 1), p.pool, 12)
    assert set(KEYS) == set(s) and math.isnan(s["mttr_weeks"])


# ---------------------------------------------------------------- max_absorption
def fake_batch(threshold, loans=0):
    def run_fn(shocks):
        x = shocks[0].magnitude
        out = []
        for seed in range(10):
            w = synth_weekly([x > threshold + seed * 0.001] * 3)
            w["payday_new_loans"] = loans
            out.append(RunResult("rigid", seed, w, pd.DataFrame()))
        return out
    return run_fn


def test_max_absorption_bisection():
    tpl = ShockEvent("hours_cut", 5, magnitude=0.1)
    got = metrics.max_absorption(fake_batch(0.37), tpl, 0.0, 1.0, tol=0.01)
    assert abs(got - 0.37) <= 0.01
    assert tpl.magnitude == 0.1  # template untouched
    assert metrics.max_absorption(fake_batch(-1), tpl, 0.0, 1.0) == 0.0  # lo fails → lo
    assert metrics.max_absorption(fake_batch(5), tpl, 0.0, 1.0) == 1.0  # capped at hi


def test_max_absorption_payday_clause():
    tpl = ShockEvent("hours_cut", 5)
    # pool never collapses but takes 2× the baseline's payday loans → fails at lo
    got = metrics.max_absorption(fake_batch(5, loans=2), tpl, 0.0, 1.0, baseline_fn=fake_batch(5, loans=1))
    assert got == 0.0


# ---------------------------------------------------------------- economics
def econ_summary(structures):
    rows = [dict(structure=s, payday_fees=fee, seed_drawn_max=seed, avg_pool_balance=bal)
            for s, fee, seed, bal in structures]
    df = pd.DataFrame(rows)
    df.attrs.update(cohort_size=10, weeks=26)
    return df


def test_economics_with_no_pool():
    df = econ_summary([("no_pool", 1000, 0, 0), ("rigid", 400, 100, 1000), ("rigid", 600, 300, 3000)])
    e = metrics.economics(df, EconParams(), 1000)
    # no_pool 100/member/26wk → 200/yr; rigid 50 → 100/yr; avoided 100/member
    assert e.loc["rigid", "payday_fees_avoided_per_member"] == pytest.approx(100)
    assert e.loc["rigid", "payday_fees_avoided"] == pytest.approx(100_000)
    assert e.loc["rigid", "float_revenue"] == pytest.approx(2000 * 100 * 0.04)
    assert e.loc["rigid", "fee_revenue"] == pytest.approx(36_000)
    assert e.loc["rigid", "opex"] == pytest.approx(48_000)
    assert e.loc["rigid", "net"] == pytest.approx(8000 + 36_000 - 48_000)
    assert e.loc["rigid", "break_even_members"] == np.inf
    assert e.loc["rigid", "sponsor_capital_at_risk"] == pytest.approx(290 * 100)
    assert e.loc["no_pool", "opex"] == 0 and math.isnan(e.loc["no_pool", "break_even_members"])


def test_economics_without_no_pool():
    df = econ_summary([("reserve_staking", 400, 0, 50_000)])
    e = metrics.economics(df, EconParams(), 1000)
    assert math.isnan(e.loc["reserve_staking", "payday_fees_avoided"])
    assert e.loc["reserve_staking", "break_even_members"] == 10  # float 200k covers opex


# ---------------------------------------------------------------- bands
def test_bands():
    p = SimParams()
    res = engine.run_many(p, ["rigid", "variable_payout"], [], 5)
    b = metrics.mc_bands(res, "reserve")
    assert b.shape == (52, 6) and ("rigid", 0.5) in b.columns
    hb = metrics.health_bands(res, p.pool, 12)
    assert hb.shape == (52, 6) and ((hb >= 0) & (hb <= 1)).all().all()
    assert (hb[("variable_payout", 0.1)] <= hb[("variable_payout", 0.9)]).all()
