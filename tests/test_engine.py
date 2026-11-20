import numpy as np
import pandas as pd
import pytest

from rosca import pools
from rosca.contracts import MEMBER_COLUMNS, PRESETS, WEEKLY_COLUMNS, ShockEvent
from rosca.engine import run, run_many
from rosca.params import PaydayParams, SimParams
from rosca.payday import PaydayBook
from rosca.pools import NoPool, Pool


class FakePool(Pool):
    """Tiny pool: $20/wk contributions, draws from the pot, whole pot paid out every 4 weeks in rotation,
    a 'reserve' that lose_reserve can hit. Exercises every engine->pool money path."""
    name = "fake"

    def __init__(self, params, n, rng):
        super().__init__(params, n, rng)
        self.pot, self.reserve, self.turn = 0.0, 100.0, 0

    def due(self, week, state):
        return np.where(state.active, 20.0, 0.0)

    def collect(self, week, state, paid):
        self.pot += paid.sum() * 0.8
        self.reserve += paid.sum() * 0.2

    def emergency_draw(self, week, state, need):
        granted = np.where(state.active, np.minimum(need, 50.0), 0.0)
        granted *= min(1.0, self.reserve / max(granted.sum(), 1e-9))
        self.reserve -= granted.sum()
        return granted

    def payout(self, week, state):
        out = np.zeros(self.n)
        if week % 4 == 3:
            out[self.turn % self.n] = self.pot if state.active[self.turn % self.n] else 0.0
            self.pot -= out.sum()
            self.turn += 1
        return out

    def on_exit(self, idx, state):
        state.active[idx] = False
        refund = np.zeros(self.n)
        refund[idx] = 1.0  # token refund
        self.reserve -= refund.sum()
        return refund

    def lose_reserve(self, frac):
        lost = self.reserve * frac
        self.reserve -= lost
        return lost

    def holdings(self):
        return self.pot + self.reserve

    def snapshot(self):
        return dict(reserve=self.reserve, pot=self.pot, staked=0.0, personal=0.0, seed_drawn=0.0, frozen=False, collapsed=False)


@pytest.fixture
def fake_pool(monkeypatch):
    monkeypatch.setitem(pools.POOLS, "fake", FakePool)
    return "fake"


STORM = PRESETS["Perfect storm"] + [ShockEvent("reserve_loss", 30, 1, 0, 0.5), ShockEvent("job_loss", 5, 6, 0.25)]


@pytest.mark.parametrize("structure", ["no_pool", "fake"])
@pytest.mark.parametrize("seed", range(5))
def test_money_conservation(structure, seed, fake_pool):
    """Every week: Δ(Σ member cash) + Δ pool.holdings() ==
       income − expenses − shocks − payday fees + payday borrowed − payday repaid (principal + EPP
       installments) + hardship forgiven (cash floored at 0) − reserve lost.
    Defaulted principal moves no cash; pool refunds/draws/payouts/contributions net to zero."""
    r = run(SimParams(), structure, STORM, seed)
    l = r.ledger
    rhs = (l.income - l.expenses - l.shocks - l.fees + l.borrowed - l.repaid + l.forgiven - l.reserve_lost)
    np.testing.assert_allclose(l.d_cash + l.d_holdings, rhs, atol=1e-6)
    assert (r.weekly.cash_total >= 0).all()


def test_contract_columns(fake_pool):
    for s in ("no_pool", "fake"):
        r = run(SimParams(), s, STORM, 0)
        assert list(r.weekly.columns) == list(WEEKLY_COLUMNS)
        assert list(r.members.columns) == list(MEMBER_COLUMNS)
        assert len(r.weekly) == SimParams().weeks


def test_fake_pool_paths_exercised(fake_pool):
    r = run(SimParams(), "fake", STORM, 1)
    w = r.weekly
    assert w.payouts.sum() > 0 and w.contributions_paid.sum() > 0 and w.emergency_draws.sum() > 0
    assert w.active_members.iloc[-1] < SimParams().population.cohort_size  # member_exit event
    assert r.ledger.reserve_lost.sum() > 0


def test_common_random_numbers_across_structures(fake_pool):
    """Same seed -> same population, shocks and income draws regardless of pool."""
    a, b = run(SimParams(), "no_pool", STORM, 7), run(SimParams(), "fake", STORM, 7)
    pd.testing.assert_series_equal(a.ledger.income, b.ledger.income)
    pd.testing.assert_series_equal(a.ledger.expenses, b.ledger.expenses)
    pd.testing.assert_series_equal(a.ledger.shocks, b.ledger.shocks)
    pd.testing.assert_series_equal(a.weekly.shocks_hit, b.weekly.shocks_hit)
    pd.testing.assert_series_equal(a.weekly.shock_active, b.weekly.shock_active)


def test_deterministic():
    a, b = run(SimParams(), "no_pool", STORM, 3), run(SimParams(), "no_pool", STORM, 3)
    pd.testing.assert_frame_equal(a.weekly, b.weekly)
    pd.testing.assert_frame_equal(a.members, b.members)
    assert not a.weekly.equals(run(SimParams(), "no_pool", STORM, 4).weekly)


def test_no_shocks_large_buffers_no_payday():
    p = SimParams()
    p.shocks.expense_rate_per_year = 0.0
    p.shocks.job_loss_rate_per_year = 0.0
    p.population.buffer_median = 1e5
    p.population.buffer_sigma = 0.0
    for seed in range(5):
        r = run(p, "no_pool", [], seed)
        assert r.weekly.payday_new_loans.sum() == 0 and r.members.loans_taken.sum() == 0
        assert r.weekly.hardship_members.sum() == 0


# ---------------------------------------------------------------- payday book
def test_payday_repay_renew_default_epp():
    p = PaydayParams(loan_size=375.0, fee_per_100=21.0, renew_prob=0.5, default_prob_when_unable=0.5)
    book = PaydayBook(p, 4)
    cash = np.zeros(4)
    lent = book.borrow(0, np.array([100.0, 400.0, 100.0, 100.0]), np.ones(4, bool), cash)
    assert list(lent) == [375, 750, 375, 375] and (book.due_week == 3).all()
    cash[:] = [1000.0, 0.0, 0.0, 0.0]
    #            repay   renew  default  forced repay
    f = book.settle(3, cash, np.array([0.9, 0.1, 0.9, 0.9]), np.array([0.9, 0.9, 0.1, 0.9]))
    fee = 375 * 0.21
    assert f["new_loans"] == 1 and f["defaults"] == 1 and f["written_off"] == 375
    np.testing.assert_allclose(cash, [1000 - 375 - fee, -750 * 0.21, 0, -375 - fee])
    assert list(book.principal) == [0, 750, 0, 0] and book.defaulted[2] and book.due_week[1] == 5
    assert list(book.loans_taken) == [1, 2, 1, 1]
    # agent 1 keeps renewing until the 10-week limit, then converts to an 8-week EPP
    for w in (5, 7, 9):
        book.settle(w, cash, np.zeros(4), np.ones(4))
    assert book.epp_left[1] == 0
    book.settle(11, cash, np.zeros(4), np.ones(4))
    assert book.epp_left[1] == p.epp_weeks and book.principal[1] == 750
    assert not book.can_borrow()[1]
    for w in range(12, 12 + p.epp_weeks):
        book.settle(w, cash, np.zeros(4), np.ones(4))
        book.end_week()
    assert book.principal[1] == 0 and book.can_borrow()[1]


def test_apply_events():
    from rosca.shocks import apply_events
    ev = [ShockEvent("hours_cut", 2, 3, 0.5, 0.3), ShockEvent("expense_spike", 4, 1, 0.25, 100.0),
          ShockEvent("member_exit", 6, 1, 0.25), ShockEvent("reserve_loss", 8, 1, 0, 0.5)]
    fx = apply_events(ev, 12, 10, np.random.default_rng(0))
    assert ((fx.hours_mult[2:5] == 0.7).sum(axis=1) == 6).all() and (fx.hours_mult[[0, 1, 5]] == 1).all()
    assert (fx.extra_expense[4] == 100).sum() == 3 and fx.extra_expense.sum() == 300
    assert len(fx.exits_by_week[6]) == 3 and fx.reserve_loss_frac[8] == 0.5
    assert list(np.nonzero(fx.shock_active)[0]) == [2, 3, 4, 6, 8]
