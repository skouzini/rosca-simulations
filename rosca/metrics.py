"""Metrics. Signatures are the contract; bodies marked PLACEHOLDER are naive stand-ins for UI work
and will be replaced by the metrics implementation (Phase 1 agent 3)."""
from typing import Callable

import numpy as np
import pandas as pd

from .contracts import RunResult, ShockEvent
from .params import EconParams, PoolParams, WEEKS_PER_YEAR


def health_index(weekly: pd.DataFrame, pool: PoolParams, cohort_size: int) -> pd.Series:
    """Pool health in [0, 1] per week: blend of reserve ratio, contribution on-time rate, and
    (1 − 4-week rolling payday reliance). For no_pool/rigid the reserve term is dropped."""
    # PLACEHOLDER
    paid = (weekly.contributions_paid / weekly.contributions_due.replace(0, np.nan)).fillna(1.0)
    payday = 1 - (weekly.members_in_payday_debt / cohort_size).rolling(4, min_periods=1).mean()
    return ((paid + payday) / 2).clip(0, 1)


def recovery_time(health: pd.Series, shock_start: int, tol: float = 0.05, hold: int = 4) -> tuple[float, bool]:
    """Weeks from shock_start until health is within tol (relative) of its pre-shock mean and stays
    there `hold` weeks. Returns (weeks, recovered). Never dipped → (0.0, True).
    Never recovered by horizon → (weeks to horizon, False) i.e. right-censored."""
    # PLACEHOLDER
    base = health.iloc[:shock_start].mean() if shock_start > 0 else health.iloc[0]
    ok = (health >= base * (1 - tol)).to_numpy()
    for w in range(shock_start, len(ok) - hold + 1):
        if ok[w:w + hold].all():
            return float(w - shock_start), True
    return float(len(ok) - shock_start), False


def survived(result: RunResult, baseline_payday_loans: float | None = None, eps: float = 0.10) -> bool:
    """No collapse, no frozen week, and payday loans ≤ (1+eps) × baseline (if baseline given)."""
    w = result.weekly
    ok = not w.collapsed.any() and not w.frozen.any()
    if baseline_payday_loans is not None:
        ok = ok and w.payday_new_loans.sum() <= (1 + eps) * baseline_payday_loans
    return bool(ok)


def summarize_run(result: RunResult, pool: PoolParams, cohort_size: int) -> dict:
    """Flat dict of per-run metrics. Keys (contract): structure, seed, mttr_weeks, recovered,
    payday_loans, payday_borrowed, payday_fees, share_used_payday, weeks_in_debt_per_borrower,
    loans_per_borrower, cycle_completed, attrition, miss_rate, frozen_weeks, max_reserve_drawdown,
    avg_payout, self_coverage_rate, hardship_weeks, avg_pool_balance, seed_drawn_max, survived."""
    # PLACEHOLDER (partial)
    w, m = result.weekly, result.members
    h = health_index(w, pool, cohort_size)
    start = min((e.start_week for e in result.shocks), default=None)
    mttr, rec = recovery_time(h, start) if start is not None else (0.0, True)
    borrowers = m[m.loans_taken > 0]
    return dict(
        structure=result.structure, seed=result.seed, mttr_weeks=mttr, recovered=rec,
        payday_loans=int(w.payday_new_loans.sum()), payday_borrowed=float(w.payday_borrowed.sum()),
        payday_fees=float(w.payday_fees.sum()), share_used_payday=len(borrowers) / max(len(m), 1),
        weeks_in_debt_per_borrower=float(borrowers.weeks_in_payday_debt.mean()) if len(borrowers) else 0.0,
        loans_per_borrower=float(borrowers.loans_taken.mean()) if len(borrowers) else 0.0,
        cycle_completed=not bool(w.collapsed.iloc[-1]), attrition=1 - w.active_members.iloc[-1] / cohort_size,
        miss_rate=float(1 - w.contributions_paid.sum() / max(w.contributions_due.sum(), 1e-9)),
        frozen_weeks=int(w.frozen.sum()), max_reserve_drawdown=float((w.reserve.cummax() - w.reserve).max()),
        avg_payout=float(w.payouts[w.payouts > 0].mean()) if (w.payouts > 0).any() else 0.0,
        self_coverage_rate=float(w.shocks_self_covered.sum() / max(w.shocks_hit.sum(), 1)),
        hardship_weeks=int(w.hardship_members.sum()), avg_pool_balance=float((w.reserve + w.pot).mean()),
        seed_drawn_max=float(w.seed_drawn.max()), survived=survived(result),
    )


def summarize_many(results: list[RunResult], pool: PoolParams, cohort_size: int) -> pd.DataFrame:
    """One row per run (summarize_run dicts). Callers aggregate with groupby('structure')."""
    return pd.DataFrame([summarize_run(r, pool, cohort_size) for r in results])


def max_absorption(run_fn: Callable[[list[ShockEvent]], list[RunResult]], template: ShockEvent,
                   lo: float, hi: float, survive_p: float = 0.9, tol: float = 0.02,
                   field: str = "magnitude") -> float:
    """Largest value of template.<field> (bisection in [lo, hi]) at which ≥ survive_p of runs survive.
    run_fn(shocks) runs the Monte Carlo batch for one structure and returns its results."""
    raise NotImplementedError  # Phase 1 agent 3


def economics(summary: pd.DataFrame, econ: EconParams, members: int) -> pd.DataFrame:
    """Per structure, scaled to `members` total members for one year: payday_fees_avoided (vs no_pool),
    sponsor_capital_at_risk (P95 seed_drawn_max per cohort × cohorts), float_revenue, fee_revenue,
    opex, net, break_even_members. Index = structure."""
    # PLACEHOLDER
    g = summary.groupby("structure")
    per_member_fees = g.payday_fees.mean() / max(summary.attrs.get("cohort_size", 12), 1)
    base = per_member_fees.get("no_pool", per_member_fees.max())
    out = pd.DataFrame({"payday_fees_avoided": (base - per_member_fees) * members})
    out["float_revenue"] = g.avg_pool_balance.mean() / 12 * members * econ.tbill_yield
    out["fee_revenue"] = (econ.pepm_fee + econ.member_fee_monthly) * 12 * members
    out["opex"] = econ.opex_per_member_month * 12 * members
    out["net"] = out.float_revenue + out.fee_revenue - out.opex
    out["sponsor_capital_at_risk"] = g.seed_drawn_max.quantile(0.95) * members / 12
    return out
