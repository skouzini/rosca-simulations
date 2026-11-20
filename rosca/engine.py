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
from .agents import expenses, income, make_population
from .params import SimParams
from .payday import PaydayBook
from .pools import make_pool
from .shocks import apply_events, draw_idiosyncratic


def run(params: SimParams, structure: str, shocks: list[ShockEvent], seed: int) -> RunResult:
    """Simulate one cohort under one structure for params.weeks weeks.

    Hardship (waterfall step 4): a member still short after draw + payday has cash floored at 0; the
    unpaid amount is lost (unpaid bills / arrears, not tracked further) and reported as `forgiven` in
    result.ledger. Extras outside the contract: result.ledger (per-week money flows, see
    tests/test_engine.py for the conservation identity) and result.defaulted (bool per agent: ever
    defaulted on a payday loan).
    """
    W, pp = params.weeks, params.population
    pop_rng, shock_rng, event_rng, behavior_rng, pool_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    state = make_population(pp, pop_rng)
    n = state.n
    idio = draw_idiosyncratic(params.shocks, n, W, shock_rng)
    fx = apply_events(shocks, n, W, event_rng)
    hours_mult = idio.hours_mult * fx.hours_mult
    shock_cost = idio.expense + fx.extra_expense
    pool = make_pool(structure, params.pool, n, pool_rng)
    book = PaydayBook(params.payday, n)
    rows, ledger = [], []

    for w in range(W):
        # fixed-size behavior draws every week (common random numbers across structures)
        z_hours, u_renew, u_default, u_payday, u_walk = (
            behavior_rng.standard_normal(n), *behavior_rng.random((4, n)))
        cash = state.cash
        cash0, hold0 = cash.sum(), pool.holdings()
        # 1. user events
        lost = pool.lose_reserve(fx.reserve_loss_frac[w]) if fx.reserve_loss_frac[w] > 0 else 0.0
        leave = fx.exits_by_week.get(w, np.array([], int))
        leave = leave[state.active[leave]]
        if leave.size:
            cash += pool.on_exit(leave, state)
        # 2. income, expenses, shocks, payday settlement
        inc = income(state, w, hours_mult[max(w - 1, 0):w + 1].mean(axis=0), z_hours, pp.hours_sd_frac)
        exp = expenses(state, w)
        cash += inc - exp - shock_cost[w]
        flows = book.settle(w, cash, u_renew, u_default)
        # 3. waterfall
        granted = pool.emergency_draw(w, state, np.maximum(-cash, 0.0))
        cash += granted
        borrowed = book.borrow(w, np.maximum(-cash, 0.0), u_payday < state.payday_propensity, cash)
        hardship = cash < 0
        forgiven = -cash[hardship].sum()
        cash[hardship] = 0.0
        state.hardship_weeks += hardship
        # 4. contributions
        due = pool.due(w, state)
        paid = np.minimum(due, cash)
        cash -= paid
        pool.collect(w, state, paid)
        # 5. payout
        out = pool.payout(w, state)
        cash += out
        state.payouts_received += out
        # 6. strategic walk-away after receiving a payout
        walk = np.nonzero((out > 0) & state.active & (u_walk < pp.strategic_default_prob))[0]
        if walk.size:
            cash += pool.on_exit(walk, state)
        if w == W - 1:  # horizon ends: return member-owned stakes (C only; zeros elsewhere)
            cash += pool.release_stakes()
        book.end_week()
        # 7. record
        hit = shock_cost[w] > 0
        snap = pool.snapshot()
        rows.append(dict(
            week=w, active_members=int(state.active.sum()), reserve=snap["reserve"], pot=snap["pot"],
            staked=snap["staked"], seed_drawn=snap["seed_drawn"], frozen=bool(snap["frozen"]),
            collapsed=bool(snap["collapsed"]), contributions_due=due.sum(), contributions_paid=paid.sum(),
            payouts=out.sum(), emergency_draws=granted.sum(),
            payday_new_loans=int((borrowed > 0).sum()) + flows["new_loans"], payday_borrowed=borrowed.sum(),
            payday_fees=flows["fees"], payday_outstanding=book.principal.sum(),
            members_in_payday_debt=int((book.principal > 0).sum()), payday_defaults=flows["defaults"],
            hardship_members=int(hardship.sum()), shocks_hit=int(hit.sum()),
            shocks_self_covered=int((hit & (borrowed == 0) & ~hardship).sum()),
            shock_active=bool(fx.shock_active[w]), cash_total=cash.sum()))
        ledger.append(dict(
            d_cash=cash.sum() - cash0, d_holdings=pool.holdings() - hold0, income=inc.sum(),
            expenses=exp.sum(), shocks=shock_cost[w].sum(), fees=flows["fees"], borrowed=borrowed.sum(),
            repaid=flows["repaid"], forgiven=forgiven, reserve_lost=lost))

    state.loans_taken[:] = book.loans_taken
    state.weeks_in_payday_debt[:] = book.weeks_in_debt
    state.fees_paid[:] = book.fees_paid
    weekly = pd.DataFrame(rows, columns=list(WEEKLY_COLUMNS))
    members = pd.DataFrame(dict(
        agent=np.arange(n), loans_taken=state.loans_taken.astype(int),
        weeks_in_payday_debt=state.weeks_in_payday_debt.astype(int), fees_paid=state.fees_paid,
        hardship_weeks=state.hardship_weeks.astype(int), payouts_received=state.payouts_received,
        exited=~state.active), columns=list(MEMBER_COLUMNS))
    result = RunResult(structure, seed, weekly, members, list(shocks))
    # extras outside the contract (plain attributes; DataFrame.attrs would break pandas concat)
    result.ledger = pd.DataFrame(ledger)
    result.defaulted = book.defaulted.copy()
    return result


def run_many(params: SimParams, structures: list[str], shocks: list[ShockEvent],
             n_runs: int, seed: int = 0) -> list[RunResult]:
    """Run n_runs seeds (seed, seed+1, ...) for each structure. Same seeds across structures."""
    return [run(params, s, shocks, seed + i) for s in structures for i in range(n_runs)]


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
