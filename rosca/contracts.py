"""Shared data contracts. Every module builds against these; change only with the integrator.

Money units are dollars. Time unit is the week (0-based week index).
"""
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class AgentState:
    """Per-agent arrays, all length n (cohort size). Owned by engine, mutated in place."""
    wage: np.ndarray               # $/hr
    hours_mean: np.ndarray         # baseline hrs/wk
    rent_monthly: np.ndarray       # $/mo
    essentials_weekly: np.ndarray  # $/wk
    cash: np.ndarray               # liquid checking balance, may not go below 0 after waterfall
    payday_propensity: np.ndarray  # P(take payday when short)
    active: np.ndarray             # bool, still a pool member
    # running per-agent tallies (engine writes these; metrics reads them via RunResult.members)
    loans_taken: np.ndarray = None
    weeks_in_payday_debt: np.ndarray = None
    fees_paid: np.ndarray = None
    hardship_weeks: np.ndarray = None
    payouts_received: np.ndarray = None

    @property
    def n(self) -> int:
        return len(self.cash)

    def __post_init__(self):
        n = len(self.cash)
        for name in ("loans_taken", "weeks_in_payday_debt", "fees_paid", "hardship_weeks", "payouts_received"):
            if getattr(self, name) is None:
                setattr(self, name, np.zeros(n))


ShockKind = Literal["hours_cut", "expense_spike", "job_loss", "member_exit", "reserve_loss"]
SHOCK_KINDS: tuple[str, ...] = ("hours_cut", "expense_spike", "job_loss", "member_exit", "reserve_loss")


@dataclass
class ShockEvent:
    """A user-defined shock. Members hit are drawn with the run's rng (same draw for every structure).

    magnitude meaning by kind:
      hours_cut     — fraction of hours lost (0–1) each week for duration_weeks
      expense_spike — $ one-time cost per hit member at start_week
      job_loss      — ignored; hit members earn 0 for duration_weeks
      member_exit   — ignored; share_hit of active members leave the pool at start_week
      reserve_loss  — fraction (0–1) of pool reserve lost at start_week; share_hit ignored
    """
    kind: ShockKind
    start_week: int
    duration_weeks: int = 1
    share_hit: float = 0.25
    magnitude: float = 0.3
    label: str = ""


PRESETS: dict[str, list[ShockEvent]] = {
    "None": [],
    "Winter heating spike": [ShockEvent("expense_spike", 20, 1, 0.75, 350.0, "Winter heating spike")],
    "Plant layoff": [ShockEvent("job_loss", 16, 10, 0.25, 0.0, "Plant layoff")],
    "Seasonal hours cut": [ShockEvent("hours_cut", 12, 8, 0.5, 0.3, "Seasonal hours cut")],
    "3 members quit": [ShockEvent("member_exit", 18, 1, 0.25, 0.0, "3 members quit")],
    "Perfect storm": [
        ShockEvent("hours_cut", 14, 10, 0.5, 0.3, "Hours cut"),
        ShockEvent("expense_spike", 16, 1, 0.5, 600.0, "Expense wave"),
        ShockEvent("member_exit", 20, 1, 0.15, 0.0, "Exits"),
    ],
}


# One row per week per run. Engine MUST produce exactly these columns.
WEEKLY_COLUMNS: dict[str, str] = {
    "week": "int, 0-based",
    "active_members": "int, members still in pool",
    "reserve": "$ pool reserve balance (0 for rigid/no_pool)",
    "pot": "$ undistributed rotating pot balance",
    "staked": "$ member-owned staked payouts held in reserve (C only)",
    "personal": "$ member-owned personal emergency shares held by the pool",
    "seed_drawn": "$ institutional seed liquidity currently drawn/lost",
    "frozen": "bool, pool could not honor a payout or emergency draw this week",
    "collapsed": "bool, pool has collapsed (sticky once True)",
    "contributions_due": "$",
    "contributions_paid": "$",
    "payouts": "$ paid to members this week",
    "emergency_draws": "$ granted from pool this week",
    "payday_new_loans": "int, new payday loans originated this week",
    "payday_borrowed": "$ new payday principal this week",
    "payday_fees": "$ payday fees paid this week",
    "payday_outstanding": "$ payday principal outstanding at end of week",
    "members_in_payday_debt": "int",
    "payday_defaults": "int, loans defaulted this week",
    "hardship_members": "int, members who missed bills this week (step 4 of waterfall)",
    "shocks_hit": "int, idiosyncratic + event expense shocks this week",
    "shocks_self_covered": "int, of those covered without payday or hardship",
    "shock_active": "bool, any user ShockEvent active this week",
    "cash_total": "$ sum of member cash",
}

MEMBER_COLUMNS: dict[str, str] = {
    "agent": "int",
    "loans_taken": "int",
    "weeks_in_payday_debt": "int",
    "fees_paid": "$",
    "hardship_weeks": "int",
    "payouts_received": "$",
    "exited": "bool",
}


@dataclass
class RunResult:
    structure: str
    seed: int
    weekly: pd.DataFrame          # columns = WEEKLY_COLUMNS
    members: pd.DataFrame         # columns = MEMBER_COLUMNS
    shocks: list[ShockEvent] = field(default_factory=list)
