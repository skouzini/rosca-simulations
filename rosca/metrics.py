"""Metrics: pool health index, recovery time (MTTR), per-run summaries, max shock absorption,
unit economics, and Monte Carlo bands for fan charts."""
import copy
from typing import Callable

import numpy as np
import pandas as pd

from .contracts import RunResult, ShockEvent
from .params import EconParams, PoolParams, WEEKS_PER_YEAR


def health_index(weekly: pd.DataFrame, pool: PoolParams, cohort_size: int) -> pd.Series:
    """Pool health in [0, 1] per week = mean of the available components:

    - reserve ratio  = clip(reserve / (target_reserve_pots × full pot), 0, 1),
                       full pot = cohort_size × contribution_weekly × round_weeks.
                       Only when the reserve column is ever > 0 (so dropped for no_pool/rigid).
    - on-time rate   = contributions_paid / contributions_due, 1 in weeks with nothing due.
                       Only when contributions are ever due (so dropped for no_pool).
    - payday freedom = 1 − 4-week rolling mean of (members_in_payday_debt / cohort_size).
    - retention      = active_members / cohort_size.

    From the first collapsed week onward, health = 0.
    """
    n = max(cohort_size, 1)
    parts = []
    if (weekly.reserve > 0).any():
        target = pool.target_reserve_pots * n * pool.contribution_weekly * pool.round_weeks
        parts.append((weekly.reserve / target).clip(0, 1) if target > 0 else pd.Series(1.0, weekly.index))
    if (weekly.contributions_due > 0).any():
        due = weekly.contributions_due.where(weekly.contributions_due > 0)
        parts.append((weekly.contributions_paid / due).fillna(1.0).clip(0, 1))
    parts.append(1 - (weekly.members_in_payday_debt / n).clip(0, 1).rolling(4, min_periods=1).mean())
    parts.append((weekly.active_members / n).clip(0, 1))
    h = pd.concat(parts, axis=1).mean(axis=1)
    collapsed = weekly.collapsed.astype(bool).cummax()
    return h.where(~collapsed, 0.0).clip(0, 1).rename("health")


def recovery_time(health: pd.Series, shock_start: int, tol: float = 0.05, hold: int = 4,
                  baseline: pd.Series | None = None) -> tuple[float, bool]:
    """Mean time to baseline recovery for one run.

    Preferred: pass `baseline` = health of the identical run without the user shock (same seed, so same
    workers and draws). Then "within tol of baseline" is checked week by week against that counterfactual,
    which removes the bias from a reserve that is still building up (or starts full from seed money).
    Without `baseline`, falls back to the pre-shock mean below.

    baseline = mean health over weeks [0, shock_start) (if shock_start < 4: the first
               max(shock_start, 1) weeks, so a week-0 shock uses week 0 as baseline).
    dip      = first week ≥ shock_start with health < baseline × (1 − tol).
    Recovery = first week w ≥ dip where health ≥ baseline × (1 − tol) for `hold` consecutive weeks
               (all within the horizon). Returns (w − shock_start, True).
    Never dipped → (0.0, True). Never recovered by horizon → (horizon − shock_start, False),
    i.e. right-censored. NaN health weeks count as below baseline; NaN baseline or a shock
    outside the horizon → (nan, False).
    """
    h = health.to_numpy(dtype=float)
    if shock_start < 0 or shock_start >= len(h):
        return float("nan"), False
    win = h[:shock_start] if shock_start >= 4 else h[:max(shock_start, 1)]
    base = np.nanmean(win) if np.isfinite(win).any() else np.nan
    if np.isnan(base):
        return float("nan"), False
    ref = baseline.to_numpy(dtype=float) if baseline is not None else base
    with np.errstate(invalid="ignore"):
        ok = h >= ref * (1 - tol)  # NaN → False
    below = np.flatnonzero(~ok[shock_start:])
    if len(below) == 0:
        return 0.0, True
    for w in range(shock_start + below[0], len(h) - hold + 1):
        if ok[w:w + hold].all():
            return float(w - shock_start), True
    return float(len(h) - shock_start), False


def member_health(weekly: pd.DataFrame, cohort_size: int) -> pd.Series:
    """Member-side health in [0, 1], comparable across every structure including no_pool:
    1 − 4-week rolling share of the cohort in payday debt or in hardship (missed bills) that week."""
    hurt = (weekly.members_in_payday_debt + weekly.hardship_members).clip(upper=cohort_size) / max(cohort_size, 1)
    return (1 - hurt.rolling(4, min_periods=1).mean()).rename("member_health")


MAX_FROZEN_SHARE = 0.10  # a pool "fails" on liquidity if it can't honor obligations in >10% of weeks


def survived(result: RunResult, baseline_payday_loans: float | None = None, eps: float = 0.10,
             max_frozen_share: float = MAX_FROZEN_SHARE) -> bool:
    """No collapse, frozen in at most max_frozen_share of weeks (an isolated freeze, e.g. a brand-new
    reserve that is still empty, is not failure), and payday loans ≤ (1+eps) × baseline (if given)."""
    w = result.weekly
    ok = not w.collapsed.any() and w.frozen.mean() <= max_frozen_share
    if baseline_payday_loans is not None:
        ok = ok and w.payday_new_loans.sum() <= (1 + eps) * baseline_payday_loans
    return bool(ok)


def summarize_run(result: RunResult, pool: PoolParams, cohort_size: int, calm: RunResult | None = None) -> dict:
    """Flat dict of per-run metrics. Keys (contract): structure, seed, mttr_weeks, recovered,
    payday_loans, payday_borrowed, payday_fees, share_used_payday, weeks_in_debt_per_borrower,
    loans_per_borrower, cycle_completed, attrition, miss_rate, frozen_weeks, max_reserve_drawdown,
    avg_payout, self_coverage_rate, hardship_weeks, avg_pool_balance, seed_drawn_max, survived.
    Added: mttr_censored (= not recovered; MTTR hit the horizon), payday_fees_per_member.

    MTTR is measured from the earliest user ShockEvent start. With no user shocks, mttr_weeks,
    recovered and mttr_censored are NaN so they drop out of averages. survived() is called without
    a payday baseline (collapse/freeze only); max_absorption applies the payday clause.
    self_coverage_rate is NaN when no shocks hit.
    """
    w, m = result.weekly, result.members
    n = max(cohort_size, 1)
    start = min((e.start_week for e in result.shocks), default=None)
    if start is None:
        mttr, rec, cens = np.nan, np.nan, np.nan
        m_mttr, m_rec = np.nan, np.nan
    else:
        m_base = member_health(calm.weekly, cohort_size) if calm is not None else None
        m_mttr, m_rec = recovery_time(member_health(w, cohort_size), start, baseline=m_base)
        base = health_index(calm.weekly, pool, cohort_size) if calm is not None else None
        mttr, rec = recovery_time(health_index(w, pool, cohort_size), start, baseline=base)
        cens = not rec
    borrowers = m[m.loans_taken > 0]
    due, hit = w.contributions_due.sum(), w.shocks_hit.sum()
    fees = float(w.payday_fees.sum())
    return dict(
        structure=result.structure, seed=result.seed, mttr_weeks=mttr, recovered=rec,
        payday_loans=int(w.payday_new_loans.sum()), payday_borrowed=float(w.payday_borrowed.sum()),
        payday_fees=fees, share_used_payday=len(borrowers) / max(len(m), 1),
        weeks_in_debt_per_borrower=float(borrowers.weeks_in_payday_debt.mean()) if len(borrowers) else 0.0,
        loans_per_borrower=float(borrowers.loans_taken.mean()) if len(borrowers) else 0.0,
        cycle_completed=not bool(w.collapsed.iloc[-1]), attrition=float(1 - w.active_members.iloc[-1] / n),
        miss_rate=float(1 - w.contributions_paid.sum() / due) if due > 0 else 0.0,
        frozen_weeks=int(w.frozen.sum()), max_reserve_drawdown=float((w.reserve.cummax() - w.reserve).max()),
        avg_payout=float(w.payouts[w.payouts > 0].mean()) if (w.payouts > 0).any() else 0.0,
        self_coverage_rate=float(w.shocks_self_covered.sum() / hit) if hit > 0 else np.nan,
        hardship_weeks=int(w.hardship_members.sum()), avg_pool_balance=float((w.reserve + w.pot).mean()),
        seed_drawn_max=float(w.seed_drawn.max()), survived=survived(result),
        mttr_censored=cens, payday_fees_per_member=fees / n,
        member_mttr_weeks=m_mttr, member_recovered=m_rec,
    )


def summarize_many(results: list[RunResult], pool: PoolParams, cohort_size: int,
                   calm: list[RunResult] | None = None) -> pd.DataFrame:
    """One row per run (summarize_run dicts). Callers aggregate with groupby('structure').
    calm: the same runs without user shocks (engine.run_many with shocks=[]), matched on
    (structure, seed), used as the per-week recovery baseline.
    attrs: cohort_size, weeks (horizon of the runs, used by economics() to annualize)."""
    pair = {(c.structure, c.seed): c for c in calm or []}
    df = pd.DataFrame([summarize_run(r, pool, cohort_size, pair.get((r.structure, r.seed))) for r in results])
    df.attrs["cohort_size"] = cohort_size
    df.attrs["weeks"] = len(results[0].weekly) if results else WEEKS_PER_YEAR
    return df


def max_absorption(run_fn: Callable[[list[ShockEvent]], list[RunResult]], template: ShockEvent,
                   lo: float, hi: float, survive_p: float = 0.9, tol: float = 0.02,
                   field: str = "magnitude",
                   baseline_fn: Callable[[list[ShockEvent]], list[RunResult]] | None = None) -> float:
    """Largest value of template.<field> (bisection in [lo, hi]) at which ≥ survive_p of runs survive.
    run_fn(shocks) runs the Monte Carlo batch for one structure and returns its results.

    A probe deep-copies template with <field> = x and runs run_fn([shock]). A run survives per
    survived(): no collapse, no frozen week, and — if baseline_fn is given — payday loans
    ≤ 1.1 × the no_pool run with the same seed (baseline_fn returns the no_pool results for the
    same shocks; without it the payday clause is skipped).
    tol is relative to the search range: stops when the bracket is < tol × (hi − lo).
    Returns lo if even lo fails (floor, not a real survival point); returns hi if hi survives
    (capped: true capacity is ≥ hi).
    """
    def ok(x: float) -> bool:
        shocks = [copy.deepcopy(template)]
        setattr(shocks[0], field, x)
        runs = run_fn(shocks)
        base = {r.seed: r.weekly.payday_new_loans.sum() for r in baseline_fn(shocks)} if baseline_fn else {}
        share = np.mean([survived(r, base.get(r.seed)) for r in runs]) if runs else 0.0
        return share >= survive_p

    if not ok(lo):
        return float(lo)
    if ok(hi):
        return float(hi)
    a, b, step = float(lo), float(hi), tol * (hi - lo)
    while b - a > step:
        mid = (a + b) / 2
        a, b = (mid, b) if ok(mid) else (a, mid)
    return a


def economics(summary: pd.DataFrame, econ: EconParams, members: int, weeks: int | None = None) -> pd.DataFrame:
    """Per structure, scaled to `members` total members for one year. Index = structure. Columns:

    payday_fees_avoided_per_member = no_pool mean fees/member − structure's (annualized ×52/weeks);
                                     NaN if no_pool is absent from the summary.
    payday_fees_avoided            = that × members.
    sponsor_capital_at_risk        = P95 of seed_drawn_max per cohort × cohorts (members / cohort_size).
    float_revenue                  = mean avg_pool_balance per cohort × cohorts × tbill_yield.
    fee_revenue                    = (pepm_fee + member_fee_monthly) × 12 × members.
    opex                           = opex_per_member_month × 12 × members.
    net                            = float_revenue + fee_revenue − opex.
    break_even_members             = cohort_size if per-member revenue ≥ per-member opex, else inf
                                     (all costs are per-member, so it is one cohort or never).
    no_pool is the status quo, not a product: its program economics (revenue, opex) are 0 and
    break_even_members is NaN. cohort_size / weeks come from summary.attrs (defaults 12 / 52).
    """
    cohort = max(summary.attrs.get("cohort_size", 12), 1)
    weeks = weeks or summary.attrs.get("weeks", WEEKS_PER_YEAR)
    cohorts = members / cohort
    g = summary.groupby("structure")
    fees_pm = g.payday_fees.mean() / cohort * WEEKS_PER_YEAR / weeks
    out = pd.DataFrame(index=fees_pm.index)
    out["payday_fees_avoided_per_member"] = fees_pm.get("no_pool", np.nan) - fees_pm
    out["payday_fees_avoided"] = out.payday_fees_avoided_per_member * members
    out["sponsor_capital_at_risk"] = g.seed_drawn_max.quantile(0.95) * cohorts
    program = (out.index != "no_pool").astype(float)
    out["float_revenue"] = g.avg_pool_balance.mean() * cohorts * econ.tbill_yield * program
    out["fee_revenue"] = (econ.pepm_fee + econ.member_fee_monthly) * 12 * members * program
    out["opex"] = econ.opex_per_member_month * 12 * members * program
    out["net"] = out.float_revenue + out.fee_revenue - out.opex
    rev_pm = (out.float_revenue + out.fee_revenue) / max(members, 1)
    out["break_even_members"] = np.where(rev_pm >= out.opex / max(members, 1), float(cohort), np.inf)
    out.loc[out.index == "no_pool", "break_even_members"] = np.nan
    return out


def _bands(long: pd.DataFrame, q) -> pd.DataFrame:
    """long has columns structure, week, value → index week, columns (structure, quantile)."""
    return long.groupby(["structure", "week"]).value.quantile(list(q)).unstack([0, 2])


def mc_bands(results: list[RunResult], column: str, q=(0.1, 0.5, 0.9)) -> pd.DataFrame:
    """Per-week quantiles of a WEEKLY_COLUMNS column across runs, for fan charts.
    Index = week; columns = MultiIndex (structure, quantile)."""
    long = pd.concat([pd.DataFrame({"structure": r.structure, "week": r.weekly.week, "value": r.weekly[column]})
                      for r in results])
    return _bands(long, q)


def health_bands(results: list[RunResult], pool: PoolParams, cohort_size: int, q=(0.1, 0.5, 0.9)) -> pd.DataFrame:
    """Same as mc_bands but for health_index."""
    long = pd.concat([pd.DataFrame({"structure": r.structure, "week": r.weekly.week,
                                    "value": health_index(r.weekly, pool, cohort_size).to_numpy()})
                      for r in results])
    return _bands(long, q)
