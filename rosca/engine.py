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
            staked=snap["staked"], personal=snap["personal"], seed_drawn=snap["seed_drawn"], frozen=bool(snap["frozen"]),
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

