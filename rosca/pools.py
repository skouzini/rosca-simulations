"""Pool structures. All share the Pool interface; the engine calls them in this order each week:

    1. user events:   pool.on_exit(idx, state) for member exits, pool.lose_reserve(frac)
    2. engine credits income, debits expenses/shocks, settles payday loans due
    3. waterfall if cash < 0:  granted = pool.emergency_draw(week, state, need)   (need ≥ 0 per agent)
                               then payday, then hardship (cash floored at 0)
    4. due = pool.due(week, state); engine pays paid = min(due, cash); pool.collect(week, state, paid)
    5. out = pool.payout(week, state); engine adds out to cash
    6. engine may call pool.on_exit for strategic walk-aways after payout
    7. engine records pool.snapshot()

Rules every pool must follow:
  - Money only moves via the return values / arguments above; pool never touches state.cash.
  - Pool may set state.active[i] = False (expulsion) and must do so in on_exit.
  - holdings() = all dollars physically held by the pool (reserve + pot + staked + unspent seed).
    Conservation: holdings changes only by collect(paid) − granted draws − payouts − lost reserve.
  - Inactive members are never due anything and never receive payouts (except returning their own
    staked balance on exit, which on_exit returns as an array the engine credits).
"""
import numpy as np

from .contracts import AgentState
from .params import WEEKS_PER_YEAR, PoolParams


class Pool:
    name = "base"

    def __init__(self, params: PoolParams, n: int, rng: np.random.Generator):
        self.p, self.n, self.rng = params, n, rng
        self.start_members = n
        self.frozen = False      # reset by engine-facing methods each week; True if pool failed an obligation
        self.collapsed = False   # sticky

    def due(self, week: int, state: AgentState) -> np.ndarray:
        """$ each agent owes this week (contribution + emergency-draw repayment). 0 for inactive."""
        raise NotImplementedError

    def collect(self, week: int, state: AgentState, paid: np.ndarray) -> None:
        """Receive payments. paid ≤ due; any shortfall is a miss → apply miss_policy."""
        raise NotImplementedError

    def emergency_draw(self, week: int, state: AgentState, need: np.ndarray) -> np.ndarray:
        """Grant up to `need` per agent from pool funds (own stake first in C). Returns granted $."""
        raise NotImplementedError

    def payout(self, week: int, state: AgentState) -> np.ndarray:
        """$ paid to each agent this week (rotation payout, staking withdrawals/yield)."""
        raise NotImplementedError

    def on_exit(self, idx: np.ndarray, state: AgentState) -> np.ndarray:
        """Members idx leave. Set state.active False. Return $ refunded to each agent (len n)."""
        raise NotImplementedError

    def lose_reserve(self, frac: float) -> float:
        """Lose frac of reserve (fraud/loss event). Return $ lost."""
        raise NotImplementedError

    def holdings(self) -> float:
        raise NotImplementedError

    def release_stakes(self) -> np.ndarray:
        """Return all member-owned balances (stakes, personal shares); engine calls once on the final week."""
        return np.zeros(self.n)

    def snapshot(self) -> dict:
        """Keys: reserve, pot, staked, personal, seed_drawn, frozen, collapsed (see contracts.WEEKLY_COLUMNS)."""
        raise NotImplementedError


class NoPool(Pool):
    """Status quo: no savings circle. Agents rely on their own cash and payday loans."""
    name = "no_pool"

    def due(self, week, state):
        return np.zeros(self.n)

    def collect(self, week, state, paid):
        pass

    def emergency_draw(self, week, state, need):
        return np.zeros(self.n)

    def payout(self, week, state):
        return np.zeros(self.n)

    def on_exit(self, idx, state):
        state.active[idx] = False
        return np.zeros(self.n)

    def lose_reserve(self, frac):
        return 0.0

    def holdings(self):
        return 0.0

    def snapshot(self):
        return dict(reserve=0.0, pot=0.0, staked=0.0, personal=0.0, seed_drawn=0.0, frozen=False, collapsed=False)


EPS = 1e-9


class _Rotating(Pool):
    """Shared rotation, miss policy, collapse and liquidation logic for A, B and C.

    Rotation: `order` is a random permutation drawn from self.rng at init. `queue` holds the members
    still waiting for their payout this cycle; the payout week is the last week of each round
    ((week + 1) % round_weeks == 0) and pays queue[0]. When the queue empties the cycle ends and a new
    one starts with the still-active members, in the original order.

    Each week's payment is allocated contribution first, then emergency-draw repayment. Any shortfall
    versus due() counts as one miss (total misses, not consecutive):
      'expel'  — after expel_after_misses misses the member is removed. Whatever they paid in stays
                 (forfeited); if they were already paid out, the loss falls on the others. Outstanding
                 draw debt and owed withholdings are written off. Expelled stakes (C) are returned in
                 that week's payout().
      'demote' — a member still in the queue swaps places with the member behind them.
    Collapse (sticky): active < collapse_min_frac × start members, or a payout can't reach 50% of the
    full pot (frozen=True that week). The next payout() then liquidates everything the pool holds:
    stakes to owners first, the rest pro rata to this round's contributions among active members
    (equally if none). Rigid keeps its unspent seed. Afterwards nothing is due and nothing is paid.

    `frozen` is reset at the first week-aware call of a new week (emergency_draw/due/collect/payout),
    not in due(), because emergency_draw runs before due() and its freeze must reach the snapshot.
    A freeze raised by on_exit/release_stakes carries into the next week's first call.
    """
    alpha = 0.0  # share of each contribution (after the personal share) routed to reserve

    def __init__(self, params, n, rng):
        super().__init__(params, n, rng)
        self.order = self.rng.permutation(n)
        self.queue = list(self.order)
        self.pot = 0.0
        self.reserve = 0.0
        self.seed_left = 0.0
        self.misses = np.zeros(n, dtype=int)
        self.round_paid = np.zeros(n)
        self.debt = np.zeros(n)       # emergency-draw principal outstanding (B, C)
        self.inst = np.zeros(n)       # weekly installment on that debt
        self.owed = np.zeros(n)       # withheld payout owed back (B)
        self.stake = np.zeros(n)      # member-owned staked balance (C)
        self.pending = np.zeros(n)    # refunds paid in the next payout() (stakes of expelled members)
        self.beta = params.personal_share_frac
        self.personal = np.zeros(n)   # member-owned emergency share (dual-track design), always withdrawable
        self._week = None
        self._carry_frozen = False

    # ---- helpers
    def _tick(self, week):
        if week != self._week:
            self._week, self.frozen, self._carry_frozen = week, self._carry_frozen, False

    def full_pot(self, state):
        return (state.active.sum() * self.p.contribution_weekly * self.p.round_weeks
                * (1 - self.beta) * (1 - self.alpha))

    def _check_size(self, state):
        if state.active.sum() < self.p.collapse_min_frac * self.start_members:
            self.collapsed = True

    def _refund_stakes(self, idx):
        return np.zeros(self.n)

    def _refund_own(self, idx):
        """Return member-owned balances: staked payouts (C) and personal emergency shares."""
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        out = self._refund_stakes(idx)
        out[idx] += self.personal[idx]
        self.personal[idx] = 0.0
        return out

    def _own_personal(self, i, want, granted):
        x = min(want, self.personal[i])
        self.personal[i] -= x
        granted[i] += x
        return x

    def _remove(self, idx, state):
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        refund = self._refund_own(idx)
        state.active[idx] = False
        self.queue = [m for m in self.queue if state.active[m]]
        self.debt[idx] = self.inst[idx] = self.owed[idx] = 0.0
        self._check_size(state)
        return refund

    def _liquidate(self, state):
        out = self._refund_own(np.arange(self.n))
        rest = self.pot + self.reserve
        w = self.round_paid * state.active
        if w.sum() <= 0:
            w = state.active.astype(float)
        if rest > 0 and w.sum() > 0:
            out += rest * w / w.sum()
            self.pot = self.reserve = 0.0
        return out

    # ---- engine interface
    def due(self, week, state):
        self._tick(week)
        if self.collapsed:
            return np.zeros(self.n)
        return np.where(state.active, self.p.contribution_weekly + np.minimum(self.debt, self.inst), 0.0)

    def collect(self, week, state, paid):
        due = self.due(week, state)
        paid = np.asarray(paid, dtype=float)
        contrib = np.minimum(paid, np.where(due > 0, self.p.contribution_weekly, 0.0))
        repay = paid - contrib
        self.personal += contrib * self.beta
        shared = contrib.sum() * (1 - self.beta)
        self.pot += shared * (1 - self.alpha)
        self.reserve += shared * self.alpha + repay.sum()
        self.debt -= np.minimum(repay, self.debt)
        self.round_paid += contrib
        missed = np.flatnonzero(state.active & (paid < due - EPS))
        self.misses[missed] += 1
        if self.p.miss_policy == "expel":
            gone = missed[self.misses[missed] >= self.p.expel_after_misses]
            if len(gone):
                self.pending += self._remove(gone, state)
        else:
            for m in missed:
                if m in self.queue:
                    j = self.queue.index(m)
                    if j + 1 < len(self.queue):
                        self.queue[j], self.queue[j + 1] = self.queue[j + 1], self.queue[j]

    def emergency_draw(self, week, state, need):
        self._tick(week)
        granted = np.zeros(self.n)
        for i in np.flatnonzero(state.active & (np.asarray(need) > EPS)):
            self._own_personal(i, need[i], granted)
        return granted

    def payout(self, week, state):
        self._tick(week)
        out, self.pending = self.pending, np.zeros(self.n)
        if self.collapsed:
            return out + self._liquidate(state)
        out += self._weekly(state)
        if (week + 1) % self.p.round_weeks or not self.queue:
            return out
        i = self.queue.pop(0)
        out[i] += self._pay_round(i, state)
        if self.collapsed:
            return out + self._liquidate(state)
        self.round_paid[:] = 0.0
        if not self.queue:
            out += self._cycle_end(state)
            self.queue = [m for m in self.order if state.active[m]]
        return out

    def on_exit(self, idx, state):
        refund = self._remove(idx, state)
        self._carry_frozen |= self.frozen
        return refund

    def lose_reserve(self, frac):
        lost = frac * self.reserve
        self.reserve -= lost
        return lost

    def holdings(self):
        return self.pot + self.reserve + self.seed_left + self.pending.sum() + self.personal.sum()

    def release_stakes(self):
        refund = self._refund_own(np.arange(self.n))
        self._carry_frozen |= self.frozen
        return refund

    def seed_drawn(self):
        return self.p.seed_liquidity - self.seed_left

    def snapshot(self):
        return dict(reserve=float(self.reserve), pot=float(self.pot), staked=float(self.stake.sum()),
                    personal=float(self.personal.sum()),
                    seed_drawn=float(self.seed_drawn()), frozen=bool(self.frozen), collapsed=bool(self.collapsed))

    # ---- structure hooks
    def _weekly(self, state):
        return np.zeros(self.n)

    def _cycle_end(self, state):
        return np.zeros(self.n)

    def _pay_round(self, i, state):
        raise NotImplementedError


class RigidROSCA(_Rotating):
    """A. Fixed contribution, fixed rotation, payout = whatever the round's pot collected (short if misses).
    miss_policy 'expel' removes after expel_after_misses misses; 'demote' moves unpaid member back one slot.
    Seed liquidity (if any) tops up short pots (tracked as seed_drawn). No reserve, no emergency draws.
    on_exit refunds nothing: an exiting member's contributions are forfeited to the pot."""
    name = "rigid"

    def __init__(self, params, n, rng):
        super().__init__(params, n, rng)
        self.seed_left = float(params.seed_liquidity)

    def _pay_round(self, i, state):
        full = self.full_pot(state)
        top = min(self.seed_left, max(0.0, full - self.pot))
        if self.pot + top < 0.5 * full - EPS:
            self.collapsed = self.frozen = True
            return 0.0
        amt, self.pot = self.pot + top, 0.0
        self.seed_left -= top
        return amt


class _ReservePool(_Rotating):
    """Reserve + emergency draws shared by B and C.

    Seed liquidity starts in the reserve; seed_drawn = clip(seed − reserve, 0, seed).
    Reserve health h = reserve / (target_reserve_pots × full net pot), full net pot = active ×
    contribution × round_weeks × (1 − reserve_frac).
    Emergency draw (active members only): grant = min(need, draw_cap − outstanding debt) × g(h), capped
    by the reserve, with g(h) = clip(h, 0.25, 1) — a linear utilization curve: as the reserve is used up
    grants are rationed proportionally, with a 25% floor so a non-empty reserve always lends something.
    Grants become debt repaid in installments of debt / draw_repay_weeks (recomputed on each new grant),
    added to due().

    Draw order (the waterfall inside the pool): personal emergency share → own stake (C) → pooled
    reserve → payout pot (only if pot_advance) → engine falls back to payday. Pot tier: a member still
    waiting for their payout this cycle takes an *advance* on it (up to full net pot − advances already
    taken), deducted from that payout; a member already paid this cycle borrows from the pot as ordinary
    draw debt (within draw_cap), repaid in installments. Advances are forfeited if the member exits.
    frozen=True if the design has a pooled reserve (reserve_frac > 0 or seed) and a draw still went
    unmet with the reserve empty. Designs without a pooled reserve never freeze on draws.
    The round payout tops a short pot up from the reserve; leftover pot goes to the reserve. A payout
    (cash + the recipient's advance) below 50% of the full net pot collapses the pool; any other
    shortfall vs target sets frozen.
    """

    def __init__(self, params, n, rng):
        super().__init__(params, n, rng)
        self.alpha = params.reserve_frac
        self.reserve = float(params.seed_liquidity)
        self.adv = np.zeros(n)  # pot advances against this cycle's payout (pot_advance)
        self.has_reserve = params.reserve_frac > 0 or params.seed_liquidity > 0

    def health(self, state):
        target = self.p.target_reserve_pots * self.full_pot(state)
        return self.reserve / target if target > 0 else 1.0

    def seed_drawn(self):
        return float(np.clip(self.p.seed_liquidity - self.reserve, 0, self.p.seed_liquidity))

    def emergency_draw(self, week, state, need):
        self._tick(week)
        granted = np.zeros(self.n)
        if self.collapsed:
            return granted
        need = np.asarray(need, dtype=float)
        for i in np.flatnonzero(state.active & (need > EPS)):
            rem = need[i] - self._own_personal(i, need[i], granted)
            rem -= self._own_stake(i, rem, granted)
            room = min(rem, self.p.draw_cap - self.debt[i])
            if room > EPS and self.reserve > EPS:
                x = min(room * np.clip(self.health(state), 0.25, 1.0), self.reserve)
                self.reserve -= x
                self._borrow(i, x)
                granted[i] += x
                rem -= x
            if rem > EPS and self.p.pot_advance:
                rem -= self._pot_draw(i, rem, granted, state)
            if rem > EPS and self.has_reserve and self.reserve <= EPS:
                self.frozen = True
        return granted

    def _borrow(self, i, x):
        self.debt[i] += x
        self.inst[i] = self.debt[i] / self.p.draw_repay_weeks

    def _pot_draw(self, i, want, granted, state):
        if i in self.queue:  # advance on own upcoming payout
            x = min(want, max(self.full_pot(state) - self.adv[i], 0.0), self.pot)
            self.adv[i] += x
        else:                # already paid this cycle: ordinary draw debt from the pot
            x = min(want, max(self.p.draw_cap - self.debt[i], 0.0), self.pot)
            self._borrow(i, x)
        self.pot -= x
        granted[i] += x
        return x

    def _remove(self, idx, state):
        self.adv[np.atleast_1d(np.asarray(idx, dtype=int))] = 0.0
        return super()._remove(idx, state)

    def _own_stake(self, i, want, granted):
        return 0.0

    def _target(self, full, state):
        return full

    def _pay_round(self, i, state):
        full = self.full_pot(state)
        target = self._target(full, state)
        a, self.adv[i] = self.adv[i], 0.0
        if a > target:  # advance exceeds what this payout turned out to be: rest becomes draw debt
            self._borrow(i, a - target)
            a = target
        amt = min(target - a, self.pot + self.reserve)
        if amt + a < 0.5 * full - EPS:
            self.collapsed = self.frozen = True
            if a:
                self._borrow(i, a)  # payout never happened: advance stays owed
            return 0.0
        self.frozen |= amt < target - a - EPS
        self.reserve += self.pot - amt
        self.pot = 0.0
        return self._after_payout(i, amt, amt + a, full, state)

    def _after_payout(self, i, cash, total, full, state):
        """cash: paid now; total: cash + advance already taken against this payout."""
        return cash


class VariablePayout(_ReservePool):
    """B. reserve_frac of each contribution → reserve. Payout = full net pot × f(h), h = reserve/target,
    f(h) = payout_floor + (1 − payout_floor)·min(h, 1); h is measured before the payout. The withheld
    amount (full net pot − amount paid) stays in the reserve and is recorded as owed to that member.
    At cycle end owed amounts are repaid from the reserve, member by member in rotation order, only
    down to h = 0.5; anything unpaid carries to later cycles. Owed amounts are forfeited on exit and
    are not paid at the end of the horizon. Emergency draws as in _ReservePool."""
    name = "variable_payout"

    def _target(self, full, state):
        return full * (self.p.payout_floor + (1 - self.p.payout_floor) * min(self.health(state), 1.0))

    def _after_payout(self, i, cash, total, full, state):
        self.owed[i] += max(0.0, full - total)
        return cash

    def _cycle_end(self, state):
        out = np.zeros(self.n)
        floor = 0.5 * self.p.target_reserve_pots * self.full_pot(state)
        for m in self.order:
            x = min(self.owed[m], self.reserve - floor)
            if x <= EPS:
                continue
            self.owed[m] -= x
            self.reserve -= x
            out[m] += x
        return out


class ReserveStaking(_ReservePool):
    """C. Same reserve + draws as B, but the payout targets the full net pot (no multiplier). At payout,
    a recipient whose cash (before the payout) is ≥ stake_cash_threshold stakes the whole payout with
    prob stake_prob (self.rng): it stays in the pool as member-owned `stake`.
    Bookkeeping: staked cash sits physically inside `reserve` (snapshot reserve includes it, so it
    raises h and funds other members' draws); snapshot `staked` is the members' claims. Do not add them.
    Yield: each week stake × staking_yield_apr / 52 is paid in cash from the free reserve
    (reserve − stakes) if it covers the total, otherwise it is added to the stake balances (accrues).
    emergency_draw uses the member's own stake first (no debt), then a normal reserve draw.
    Stakes come back on exit (on_exit refund), on expulsion (that week's payout), on collapse
    (liquidation), and at the end of the horizon via release_stakes(), which the engine must call on
    the final week. If the reserve can't honor a stake withdrawal in cash, frozen=True and the
    unhonored part stays staked (draws) or is written off (exit/release refunds are paid pro rata)."""
    name = "reserve_staking"

    def _own_stake(self, i, want, granted):
        want = min(want, self.stake[i])
        x = min(want, max(self.reserve, 0.0))
        self.frozen |= x < want - EPS
        self.stake[i] -= x
        self.reserve -= x
        granted[i] += x
        return x

    def _after_payout(self, i, cash, total, full, state):
        if cash > 0 and state.cash[i] >= self.p.stake_cash_threshold and self.rng.random() < self.p.stake_prob:
            self.stake[i] += cash
            self.reserve += cash
            return 0.0
        return cash

    def _weekly(self, state):
        y = self.stake * self.p.staking_yield_apr / WEEKS_PER_YEAR
        if self.reserve - self.stake.sum() >= y.sum():
            self.reserve -= y.sum()
            return y
        self.stake += y
        return np.zeros(self.n)

    def _refund_stakes(self, idx):
        out = np.zeros(self.n)
        want = self.stake[idx].sum()
        if want <= 0:
            return out
        scale = min(1.0, max(self.reserve, 0.0) / want)
        self.frozen |= scale < 1.0
        out[idx] = self.stake[idx] * scale
        self.reserve -= out.sum()
        self.stake[idx] = 0.0
        return out

POOLS: dict[str, type[Pool]] = {
    "no_pool": NoPool,
    "rigid": RigidROSCA,
    "variable_payout": VariablePayout,
    "reserve_staking": ReserveStaking,
}


def make_pool(structure: str, params: PoolParams, n: int, rng: np.random.Generator) -> Pool:
    return POOLS[structure](params, n, rng)
