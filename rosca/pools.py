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
from .params import PoolParams


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

    def snapshot(self) -> dict:
        """Keys: reserve, pot, staked, seed_drawn, frozen, collapsed (see contracts.WEEKLY_COLUMNS)."""
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
        return dict(reserve=0.0, pot=0.0, staked=0.0, seed_drawn=0.0, frozen=False, collapsed=False)


class RigidROSCA(Pool):
    """A. Fixed contribution, fixed rotation, payout = whatever the round's pot collected (short if misses).
    miss_policy 'expel' removes after expel_after_misses misses; 'demote' moves unpaid member back one slot.
    Seed liquidity (if any) tops up short pots. No reserve, no emergency draws."""
    name = "rigid"


class VariablePayout(Pool):
    """B. reserve_frac of each contribution → reserve. Payout = full pot × f(h), h = reserve/target,
    f(0)=payout_floor, f(h≥1)=1, linear between; withheld amount goes to reserve and is owed back to
    that member at cycle end if the reserve allows. Emergency draws from reserve, capped at draw_cap,
    with grant fraction shrinking as h falls (utilization-curve friction); repaid over draw_repay_weeks."""
    name = "variable_payout"


class ReserveStaking(Pool):
    """C. Same reserve + draws as B, but payout is fixed (full pot). At payout, a member with
    cash ≥ stake_cash_threshold stakes the payout with prob stake_prob: it stays in the reserve,
    member-owned, earning staking_yield_apr; withdrawn on demand via emergency_draw (own stake first)
    or returned at the end of the horizon/on exit. If the reserve can't honor a withdrawal → frozen."""
    name = "reserve_staking"


POOLS: dict[str, type[Pool]] = {
    "no_pool": NoPool,
    "rigid": RigidROSCA,
    "variable_payout": VariablePayout,
    "reserve_staking": ReserveStaking,
}


def make_pool(structure: str, params: PoolParams, n: int, rng: np.random.Generator) -> Pool:
    return POOLS[structure](params, n, rng)
