import numpy as np
import pytest

from rosca.contracts import AgentState
from rosca.params import PoolParams
from rosca.pools import ReserveStaking, RigidROSCA, VariablePayout

N = 6
C, RW = 50.0, 2
FULL = N * C * RW            # 600 gross pot
FULL_NET = FULL * 0.8        # 480 with reserve_frac 0.2


def make_state(n=N, cash=1000.0):
    z = np.zeros(n)
    return AgentState(wage=z + 17, hours_mean=z + 32, rent_monthly=z + 1350, essentials_weekly=z + 200,
                      cash=z + cash, payday_propensity=z + 0.7, active=np.ones(n, dtype=bool))


def make(cls, seed=0, **kw):
    return cls(PoolParams(contribution_weekly=C, round_weeks=RW, **kw), N, np.random.default_rng(seed))


def week(pool, state, w, pay=1.0, need=None):
    """One week in the documented order. pay: fraction of due paid (scalar or per agent)."""
    granted = pool.emergency_draw(w, state, np.zeros(N) if need is None else need)
    paid = pool.due(w, state) * pay
    pool.collect(w, state, paid)
    out = pool.payout(w, state)
    state.cash += out
    return out


# ---------------------------------------------------------------- A. rigid
def test_rigid_full_cycle_everyone_paid_once():
    pool, s = make(RigidROSCA), make_state()
    outs = np.array([week(pool, s, w) for w in range(N * RW)])
    assert ((outs > 0).sum(axis=0) == 1).all()
    np.testing.assert_allclose(outs.sum(axis=0), FULL)
    assert pool.pot == 0 and pool.holdings() == 0
    assert len(pool.queue) == N  # new cycle started


def test_rigid_expel_removes_and_never_pays():
    pool, s = make(RigidROSCA, miss_policy="expel", expel_after_misses=2), make_state()
    m = pool.order[-1]
    pay = np.ones(N)
    pay[m] = 0
    outs = np.array([week(pool, s, w, pay) for w in range(N * RW)])
    assert not s.active[m] and outs[:, m].sum() == 0
    assert m not in pool.queue
    # first round: m paid nothing, pot = 5 members × 100; m's absence is not topped up
    assert outs[1].sum() == pytest.approx(5 * C * RW)


def test_rigid_demote_moves_back_one_slot():
    pool, s = make(RigidROSCA, miss_policy="demote"), make_state()
    first, second = pool.order[0], pool.order[1]
    pay = np.ones(N)
    pay[first] = 0
    pool.collect(0, s, pool.due(0, s) * pay)
    assert pool.queue[:2] == [second, first] and s.active[first]
    pool.payout(0, s)
    out = week(pool, s, 1)
    assert out[second] > 0 and out[first] == 0


def test_rigid_collapse_on_exits_refunds_pot():
    pool, s = make(RigidROSCA), make_state()
    week(pool, s, 0)                     # pot = 300
    pool.on_exit(np.array([0, 1, 2, 3]), s)
    assert pool.collapsed
    out = pool.payout(1, s)
    assert out.sum() == pytest.approx(300) and out[:4].sum() == 0
    assert pool.pot == 0 and (pool.due(2, s) == 0).all()


def test_rigid_failed_payout_freezes_and_collapses():
    pool, s = make(RigidROSCA), make_state()
    pay = np.array([1, 1, 0, 0, 0, 0.0])
    week(pool, s, 0, pay)
    out = week(pool, s, 1, pay)          # pot 200 < 50% of 600
    assert pool.frozen and pool.collapsed
    np.testing.assert_allclose(out, [100, 100, 0, 0, 0, 0])
    week(pool, s, 2)
    assert not pool.frozen and pool.collapsed  # frozen resets, collapse is sticky


def test_rigid_seed_tops_up_short_pot():
    pool, s = make(RigidROSCA, seed_liquidity=150), make_state()
    pay = np.ones(N)
    pay[pool.order[3]] = 0
    week(pool, s, 0, pay)
    out = week(pool, s, 1, pay)
    assert out.sum() == pytest.approx(FULL) and pool.snapshot()["seed_drawn"] == pytest.approx(100)


# ---------------------------------------------------------------- B. variable payout
def test_variable_payout_shrinks_when_reserve_low():
    pool, s = make(VariablePayout), make_state()
    week(pool, s, 0)
    out = week(pool, s, 1)
    h = (N * C * RW * 0.2) / FULL_NET    # reserve 120 / target 480
    assert out.sum() == pytest.approx(FULL_NET * (0.7 + 0.3 * h))
    assert pool.owed.sum() == pytest.approx(FULL_NET - out.sum())
    rich, s2 = make(VariablePayout, seed_liquidity=FULL_NET), make_state()
    week(rich, s2, 0)
    assert week(rich, s2, 1).sum() == pytest.approx(FULL_NET)


def test_variable_payout_repays_owed_at_cycle_end():
    pool, s = make(VariablePayout), make_state()
    outs = np.array([week(pool, s, w) for w in range(N * RW)])
    assert pool.owed.sum() == pytest.approx(0)
    np.testing.assert_allclose(outs.sum(axis=0), FULL_NET)


def test_variable_payout_draws_friction_and_freeze():
    empty, s = make(VariablePayout), make_state()
    need = np.zeros(N)
    need[0] = 200
    assert empty.emergency_draw(0, s, need).sum() == 0 and empty.frozen
    full, s = make(VariablePayout, seed_liquidity=FULL_NET), make_state()
    assert full.emergency_draw(0, s, need)[0] == pytest.approx(200)         # h = 1
    assert full.due(0, s)[0] == pytest.approx(C + 200 / 8)
    half, s = make(VariablePayout, seed_liquidity=FULL_NET / 2), make_state()
    assert half.emergency_draw(0, s, need)[0] == pytest.approx(100)         # g(0.5)
    need[0] = 900
    assert full.emergency_draw(1, s, need)[0] <= 500 - 200 + 1e-9            # draw_cap


# ---------------------------------------------------------------- C. reserve + staking
def test_staking_keeps_reserve_higher():
    def run(prob):
        pool, s = make(ReserveStaking, stake_prob=prob, stake_cash_threshold=0), make_state()
        for w in range(N * RW):
            week(pool, s, w)
        return pool
    stakers, keepers = run(1.0), run(0.0)
    assert stakers.reserve > keepers.reserve + 1000
    assert stakers.stake.sum() > 0 and keepers.stake.sum() == 0


def test_own_stake_withdrawal_and_release():
    pool, s = make(ReserveStaking, stake_prob=1.0, stake_cash_threshold=0), make_state()
    week(pool, s, 0)
    out = week(pool, s, 1)
    i = pool.order[0]
    assert out[i] == 0 and pool.stake[i] == pytest.approx(FULL_NET)
    need = np.zeros(N)
    need[i] = 100
    assert pool.emergency_draw(2, s, need)[i] == pytest.approx(100)
    assert pool.stake[i] == pytest.approx(FULL_NET - 100) and pool.debt[i] == 0
    pool.lose_reserve(1.0)
    assert pool.emergency_draw(3, s, need)[i] == 0 and pool.frozen
    pool2, s2 = make(ReserveStaking, stake_prob=1.0, stake_cash_threshold=0), make_state()
    week(pool2, s2, 0)
    week(pool2, s2, 1)
    refund = pool2.release_stakes()
    assert refund.sum() == pytest.approx(FULL_NET) and pool2.stake.sum() == 0


def test_snapshot_keys_and_types():
    for cls in (RigidROSCA, VariablePayout, ReserveStaking):
        snap = make(cls).snapshot()
        assert set(snap) == {"reserve", "pot", "staked", "personal", "seed_drawn", "frozen", "collapsed"}
        assert all(type(v) in (float, bool) for v in snap.values())


# ---------------------------------------------------------------- conservation
@pytest.mark.parametrize("cls", [RigidROSCA, VariablePayout, ReserveStaking])
@pytest.mark.parametrize("policy", ["expel", "demote"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("beta", [0.0, 0.25])
def test_conservation_every_week(cls, policy, seed, beta):
    """Δholdings == paid_in − granted − payouts − refunds − lost reserve, every week."""
    rng = np.random.default_rng(seed)
    pool = make(cls, seed=seed, miss_policy=policy, seed_liquidity=200.0 * (seed % 2),
                collapse_min_frac=0.3 if seed < 3 else 0.5, stake_cash_threshold=300,
                personal_share_frac=beta)
    s = make_state(cash=rng.uniform(0, 1500))
    for w in range(52):
        h0, flows = pool.holdings(), 0.0
        if rng.random() < 0.04:
            flows -= pool.on_exit(rng.choice(np.flatnonzero(s.active) if s.active.any() else [0], 1), s).sum()
        if rng.random() < 0.05:
            flows -= pool.lose_reserve(0.3)
        need = np.where(rng.random(N) < 0.2, rng.uniform(0, 600, N), 0.0)
        granted = pool.emergency_draw(w, s, need)
        assert (granted <= need + 1e-9).all() and (granted[~s.active] == 0).all()
        due = pool.due(w, s)
        assert (due[~s.active] == 0).all()
        paid = due * np.where(rng.random(N) < 0.85, 1.0, rng.choice([0.0, 0.5], N))
        pool.collect(w, s, paid)
        out = pool.payout(w, s)
        s.cash += out
        flows += paid.sum() - granted.sum() - out.sum()
        if w == 51:
            flows -= pool.release_stakes().sum()
        assert pool.holdings() - h0 == pytest.approx(flows, abs=1e-6)
        assert pool.pot >= -1e-9 and pool.reserve >= -1e-9 and (out >= -1e-9).all()
        snap = pool.snapshot()
        assert pool.holdings() >= -1e-9 and 0 <= snap["seed_drawn"] <= pool.p.seed_liquidity + 1e-9


# ---------------------------------------------------------------- personal emergency share (all structures)
@pytest.mark.parametrize("cls", [RigidROSCA, VariablePayout, ReserveStaking])
def test_personal_share_split_draw_and_refund(cls):
    pool = make(cls, personal_share_frac=0.25)
    s = make_state()
    out = week(pool, s, 0) + week(pool, s, 1)                    # one full round
    assert pool.personal == pytest.approx(np.full(N, 2 * C * 0.25))
    assert pool.full_pot(s) == pytest.approx(FULL * 0.75 * (1 - pool.alpha))
    need = np.zeros(N); need[0] = 10.0
    g = pool.emergency_draw(2, s, need)
    assert g[0] == pytest.approx(10.0) and pool.debt[0] == 0.0      # own money first: no debt
    assert pool.personal[0] == pytest.approx(15.0)
    refund = pool.on_exit(np.array([1]), s)
    assert refund[1] == pytest.approx(25.0) and pool.personal[1] == 0.0
    rel = pool.release_stakes()
    assert rel[0] == pytest.approx(15.0) and pool.personal.sum() == 0.0
