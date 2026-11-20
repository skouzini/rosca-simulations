"""Payday loans, Utah rules (Utah Code 7-23). One loan sequence per agent at a time.

Loan model:
  - Principal = need rounded UP to a multiple of loan_size (a $500 shortfall borrows $750); the excess
    stays in the borrower's cash. Borrowing again while a loan is outstanding adds to its principal.
  - Fee = fee_per_100/100 × principal per term. Due on the first payday (odd week) ≥ term_weeks after
    origination/renewal.
  - At due: repay principal+fee if cash allows. Else renew (pay fee only) with renew_prob. Else default
    with default_prob_when_unable (principal written off, agent flagged). Else the borrower repays in
    full anyway, going negative; the engine's waterfall then covers it (usually a fresh loan:
    a back-to-back re-borrow, which restarts the 10-week clock, as it does in practice).
  - Utah: once a sequence is ≥ max_weeks old and the borrower can't repay at due, it converts to an
    interest-free extended payment plan of epp_weeks equal weekly installments (no renew/default).
    Installments are debited every week; a shortfall goes through the waterfall (no new payday loan
    while on a plan: Utah bars the lender from new loans during an EPP; we apply it to all lenders).
Counting (CFPB): every origination, top-up and renewal counts as one loan in loans_taken.
weeks_in_payday_debt counts weeks ending with any principal outstanding (EPP included).
"""
import numpy as np

from .agents import is_payday
from .params import PaydayParams


class PaydayBook:
    def __init__(self, p: PaydayParams, n: int):
        self.p, self.n = p, n
        self.principal = np.zeros(n)
        self.due_week = np.full(n, -1)
        self.seq_start = np.full(n, -1)
        self.epp_left = np.zeros(n, dtype=int)
        self.epp_amt = np.zeros(n)
        self.loans_taken = np.zeros(n, dtype=int)
        self.weeks_in_debt = np.zeros(n, dtype=int)
        self.fees_paid = np.zeros(n)
        self.defaulted = np.zeros(n, dtype=bool)  # ever defaulted

    def _next_due(self, week: int) -> int:
        d = week + self.p.term_weeks
        return d if is_payday(d) else d + 1

    def can_borrow(self) -> np.ndarray:
        return self.epp_left == 0

    def settle(self, week: int, cash: np.ndarray, u_renew: np.ndarray, u_default: np.ndarray) -> dict:
        """Settle loans due this week; mutates cash. Returns flows: fees, repaid, new_loans (renewals),
        defaults, written_off."""
        p = self.p
        fee = self.principal * p.fee_per_100 / 100
        out = dict(fees=0.0, repaid=0.0, new_loans=0, defaults=0, written_off=0.0)
        # EPP installments, every week
        epp = self.epp_left > 0
        inst = np.where(epp, np.minimum(self.epp_amt, self.principal), 0.0)
        cash -= inst
        self.principal -= inst
        self.epp_left[epp] -= 1
        out["repaid"] += inst.sum()

        due = (self.due_week == week) & (self.principal > 0) & ~epp
        can = due & (cash >= self.principal + fee)
        unable = due & ~can
        to_epp = unable & (week - self.seq_start >= p.max_weeks)
        unable &= ~to_epp
        renew = unable & (u_renew < p.renew_prob)
        default = unable & ~renew & (u_default < p.default_prob_when_unable)
        forced = unable & ~renew & ~default  # repays anyway; waterfall covers the hole
        repay = can | forced

        out["repaid"] += self.principal[repay].sum()
        cash[repay] -= self.principal[repay] + fee[repay]
        cash[renew] -= fee[renew]
        paid_fee = np.where(repay | renew, fee, 0.0)
        self.fees_paid += paid_fee
        out["fees"] = paid_fee.sum()
        self.loans_taken[renew] += 1
        out["new_loans"] = int(renew.sum())
        self.due_week[renew] = self._next_due(week)

        out["defaults"] = int(default.sum())
        out["written_off"] = self.principal[default].sum()
        self.defaulted |= default
        self.principal[repay | default] = 0.0

        self.epp_left[to_epp] = p.epp_weeks
        self.epp_amt[to_epp] = self.principal[to_epp] / p.epp_weeks
        self.due_week[~(self.principal > 0)] = -1
        return out

    def borrow(self, week: int, need: np.ndarray, take: np.ndarray, cash: np.ndarray) -> np.ndarray:
        """Agents in `take` borrow need rounded up to loan_size multiples; mutates cash. Returns $ lent each."""
        take = take & self.can_borrow() & (need > 0)
        amt = np.where(take, np.ceil(need / self.p.loan_size) * self.p.loan_size, 0.0)
        fresh = take & (self.principal == 0)
        self.seq_start[fresh] = week
        self.due_week[fresh] = self._next_due(week)
        self.principal += amt
        cash += amt
        self.loans_taken[take] += 1
        return amt

    def end_week(self) -> None:
        self.principal[self.principal < 1e-9] = 0.0
        self.epp_left[self.principal == 0] = 0
        self.weeks_in_debt += self.principal > 0
