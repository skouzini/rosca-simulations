"""Simulation engine: weekly tick loop and shortfall waterfall (see pools.py for call order).

Randomness contract (common random numbers): seq = np.random.SeedSequence(seed);
    pop_rng, shock_rng, event_rng, behavior_rng, pool_rng = [np.random.default_rng(s) for s in seq.spawn(5)]
Everything except pool_rng must be drawn identically regardless of structure, so all four structures
see the same population and the same shocks for a given seed. Draw idiosyncratic shocks for the whole
horizon up front; draw behavior randomness as fixed-size arrays per week (not conditional on pool state).
"""
import numpy as np
import pandas as pd

from .contracts import MEMBER_COLUMNS, WEEKLY_COLUMNS, RunResult, ShockEvent
from .params import SimParams


def run(params: SimParams, structure: str, shocks: list[ShockEvent], seed: int) -> RunResult:
    """Simulate one cohort under one structure for params.weeks weeks."""
    raise NotImplementedError  # Phase 1 agent 1


def run_many(params: SimParams, structures: list[str], shocks: list[ShockEvent],
             n_runs: int, seed: int = 0) -> list[RunResult]:
    """Run n_runs seeds (seed, seed+1, ...) for each structure. Same seeds across structures."""
    runner = run
    try:
        run(params, structures[0], shocks, seed)
    except NotImplementedError:
        runner = fake_run  # shortcut: remove fallback once engine.run lands
    return [runner(params, s, shocks, seed + i) for s in structures for i in range(n_runs)]


# ---------------------------------------------------------------- fake data for dashboard development
_FAKE_PROFILE = {  # (base payday reliance, resilience, has_reserve)
    "no_pool": (1.0, 0.0, False),
    "rigid": (0.75, 0.3, False),
    "variable_payout": (0.45, 0.7, True),
    "reserve_staking": (0.35, 0.8, True),
}


def fake_run(params: SimParams, structure: str, shocks: list[ShockEvent], seed: int) -> RunResult:
    """Plausible-looking RunResult with the exact contract columns. For UI work only."""
    rng = np.random.default_rng(seed)
    W, n = params.weeks, params.population.cohort_size
    rel, resil, has_res = _FAKE_PROFILE[structure]
    stress = np.zeros(W)
    for e in shocks:
        sev = e.share_hit * (1.0 if e.kind in ("job_loss", "member_exit", "reserve_loss") else max(e.magnitude, 0.3))
        for w in range(e.start_week, W):
            stress[w] += sev * np.exp(-(w - e.start_week - e.duration_weeks) * (0.15 + 0.3 * resil)
                                      if w >= e.start_week + e.duration_weeks else 0.0)
    pot_size = n * params.pool.contribution_weekly * params.pool.round_weeks
    target = params.pool.target_reserve_pots * pot_size
    reserve = np.clip(target * (np.minimum(np.arange(W) / 20, 1) - stress * (1.5 - resil))
                      + rng.normal(0, 30, W), 0, None) if has_res else np.zeros(W)
    in_debt = np.clip(np.round(n * (0.3 * rel + stress * rel * 0.8) + rng.normal(0, 0.7, W)), 0, n).astype(int)
    exits = np.cumsum(rng.random(W) < 0.01 + stress * (0.1 - 0.08 * resil)) if structure != "no_pool" else np.zeros(W)
    active = np.clip(n - exits, 0, n).astype(int) if structure != "no_pool" else np.full(W, n)
    collapsed = np.maximum.accumulate(active < params.pool.collapse_min_frac * n)
    due = np.where(structure == "no_pool", 0, active * params.pool.contribution_weekly).astype(float)
    paid = due * np.clip(1 - stress * 0.5, 0, 1)
    new_loans = rng.poisson(0.15 * n * rel * (1 + 2 * stress))
    weekly = pd.DataFrame({
        "week": np.arange(W), "active_members": active, "reserve": reserve,
        "pot": np.where(structure == "no_pool", 0, pot_size * 0.5), "staked": reserve * 0.3 if structure == "reserve_staking" else np.zeros(W),
        "seed_drawn": np.zeros(W), "frozen": (reserve < 50) & has_res & (stress > 0.2), "collapsed": collapsed,
        "contributions_due": due, "contributions_paid": paid,
        "payouts": np.where((np.arange(W) % 2 == 1) & (structure != "no_pool"), pot_size, 0.0),
        "emergency_draws": np.where(has_res, stress * 400 * n * 0.3, 0.0),
        "payday_new_loans": new_loans, "payday_borrowed": new_loans * params.payday.loan_size,
        "payday_fees": in_debt * params.payday.loan_size * params.payday.fee_per_100 / 100 / 2,
        "payday_outstanding": in_debt * params.payday.loan_size, "members_in_payday_debt": in_debt,
        "payday_defaults": rng.poisson(0.02 * n * rel * (1 + stress)),
        "hardship_members": rng.poisson(0.1 * n * (1 - resil) * (1 + 3 * stress)),
        "shocks_hit": rng.poisson(n * 2.5 / 52 * (1 + 3 * stress)),
        "shocks_self_covered": 0, "shock_active": stress > 0.05, "cash_total": 300.0 * n * (1 - stress).clip(0),
    })
    weekly["shocks_self_covered"] = np.floor(weekly["shocks_hit"] * (1 - rel * 0.6)).astype(int)
    members = pd.DataFrame({
        "agent": np.arange(n), "loans_taken": rng.poisson(8 * rel, n), "weeks_in_payday_debt": rng.poisson(20 * rel, n),
        "fees_paid": rng.gamma(2, 260 * rel + 1, n), "hardship_weeks": rng.poisson(3 * (1 - resil), n),
        "payouts_received": np.where(structure == "no_pool", 0, pot_size * 2.0), "exited": np.arange(n) >= active[-1],
    })
    assert list(weekly.columns) == list(WEEKLY_COLUMNS) and list(members.columns) == list(MEMBER_COLUMNS)
    return RunResult(structure, seed, weekly, members, list(shocks))
