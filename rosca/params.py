"""All simulation parameters with defaults. Every value notes its source or says ASSUMPTION.

Sources: see docs/parameters.md for full links.
"""
from dataclasses import dataclass, field
from typing import Literal

WEEKS_PER_YEAR = 52


@dataclass
class PopulationParams:
    cohort_size: int = 12                 # business draft: 10–20 per cohort
    wage_median: float = 17.0             # $/hr. BLS OEWS May 2025 Salt Lake MSA hourly jobs: $13.98–21.83, weighted $17.91
    wage_sigma: float = 0.20              # lognormal sigma. ASSUMPTION
    hours_mean: float = 30.0              # hrs/wk. BLS CES Oct 2026: retail 30.7, food svc 23.8, private 33.8
    hours_sd_frac: float = 0.15           # weekly hours noise; JPMC: median monthly income CV 0.38
    rent_median: float = 1456.0           # $/mo. HUD FY2026 FMR Salt Lake County 1BR (2BR: $1,747)
    rent_sigma: float = 0.15
    essentials_monthly: float = 900.0     # $/mo non-rent essentials. ASSUMPTION, low vs BLS CE (~$2,300): read as net of benefits/2nd earner
    buffer_median: float = 600.0          # starting liquid cash. SHED 2024: 37% can't cover $400 (calibrated; lognormal σ=1 → 34% < $400)
    buffer_sigma: float = 1.0
    payday_propensity: float = 0.5        # P(take payday | short). ASSUMPTION (present bias) (calibrated)
    strategic_default_prob: float = 0.01  # per-cycle walk-away after payout. MAF ~0.7–1%
    # budget_mode 'dollars': rent/essentials from the dollar medians above (payday-borrower calibration).
    # budget_mode 'margin': costs = income × (1 − margin), margin ~ N(margin_mean, margin_sd) clipped to
    # [0.01, 0.25]; rent = rent_share of costs. Models the target group: break-even-or-better workers whose
    # shortfalls come from shocks and pay timing, not a chronic deficit (Solution Direction).
    budget_mode: Literal["dollars", "margin"] = "dollars"
    margin_mean: float = 0.08             # ASSUMPTION: thin monthly surplus as share of gross pay
    margin_sd: float = 0.04
    rent_share: float = 0.5               # rent share of costs; HUD cost-burdened renters pay 30–50%+ of income


@dataclass
class ShockParams:
    expense_rate_per_year: float = 1.2    # Poisson rate. Fed SHED 2025: 59% had ≥1 major unexpected expense; types sum to 110%
    expense_median: float = 1000.0        # SHED 2025: median vehicle/house/medical expense $1,000–1,999
    expense_sigma: float = 0.8            # ASSUMPTION (SHED publishes binned medians only)
    job_loss_rate_per_year: float = 0.12  # BLS JOLTS Sep 2026: layoffs 1.0%/mo overall, 1.4%/mo leisure & hospitality
    job_loss_weeks_mean: float = 10.0     # BLS CPS Oct 2026: median unemployment duration 11.9 wks


@dataclass
class PaydayParams:
    loan_size: float = 465.0              # Utah DFI annual report Oct 2026 (CY2025 avg; prior yrs $504, $487)
    fee_per_100: float = 18.0             # per 2-wk term; Utah DFI CY2025 avg APR 469% / 26 ≈ $18
    term_weeks: int = 2
    max_weeks: int = 10                   # Utah Code 7-23: 10-week limit, then extended payment plan
    epp_weeks: int = 9                    # Utah Code 7-23-403: ≥60 days, ≥4 equal payments, no interest
    renew_prob: float = 0.4               # rollover of the same loan; Utah DFI: paid-in-full loans avg 23 days.
                                          # CFPB's >80% "renewed within 14 days" also counts re-borrowing, which the
                                          # model produces separately via new loans
    default_prob_when_unable: float = 0.08  # P(default | can't pay or renew). calibrated to CFPB ~20% of borrowers default


POPULATIONS = {  # name: (label, description, PopulationParams overrides)
    "timing_mismatch": ("Timing mismatch (target group)",
                        "Workers who break even or better on average; shortfalls come from shocks and pay timing.",
                        dict(budget_mode="margin")),
    "payday_calibrated": ("Payday-borrower calibrated (stress test)",
                          "Tuned to reproduce national payday-borrower behavior; about a third run a chronic deficit.",
                          dict(budget_mode="dollars")),
}


StructureName = Literal["no_pool", "rigid", "variable_payout", "reserve_staking"]
STRUCTURES: tuple[str, ...] = ("no_pool", "rigid", "variable_payout", "reserve_staking")
STRUCTURE_LABELS = {
    "no_pool": "Payday only (status quo)",
    "rigid": "A. Rigid ROSCA",
    "variable_payout": "B. Variable payout",
    "reserve_staking": "C. Reserve + staking",
}


@dataclass
class PoolParams:
    contribution_weekly: float = 25.0     # ASSUMPTION; ≈ target group's surplus. pot = n × 2 × this per round ($600 > $465 payday loan)
    round_weeks: int = 2                  # one payout every 2 weeks
    miss_policy: Literal["expel", "demote"] = "demote"
    expel_after_misses: int = 2
    collapse_min_frac: float = 0.5        # pool collapses if active members < this share of start
    seed_liquidity: float = 0.0           # institutional seed, $ (Solution Direction assumption #3)
    # all rotating structures (A, B, C)
    personal_share_frac: float = 0.0      # β: share of each payment to the member's own emergency share,
                                          # withdrawable any time (dual-track design; business draft uses 0.25)
    # B and C
    reserve_frac: float = 0.20            # share of each contribution routed to reserve (α)
    target_reserve_pots: float = 1.0      # target reserve = this × one pot
    draw_cap: float = 500.0               # max outstanding emergency draw per member. ASSUMPTION
    draw_repay_weeks: int = 8             # emergency draw repaid evenly over this many weeks
    pot_advance: bool = False             # allow emergency advances from the rotating payout pot (last tier
                                          # before payday), deducted from the member's own payout. B and C only
    # B only
    payout_floor: float = 0.7             # payout multiplier when reserve health h = 0
    # C only
    staking_yield_apr: float = 0.04       # T-bill sweep. ASSUMPTION (business draft 3–5%)
    stake_prob: float = 0.5               # P(stake | no deficit at payout). ASSUMPTION
    stake_cash_threshold: float = 500.0   # agent stakes only if cash after payout-free week ≥ this


@dataclass
class EconParams:
    tbill_yield: float = 0.04             # float yield on pooled balances
    pepm_fee: float = 3.0                 # employer per-employee-per-month ($2–4)
    member_fee_monthly: float = 0.0       # direct membership fee ($3–5) for non-employer cohorts
    opex_per_member_month: float = 4.0    # ASSUMPTION


@dataclass
class SimParams:
    weeks: int = 52
    population: PopulationParams = field(default_factory=PopulationParams)
    shocks: ShockParams = field(default_factory=ShockParams)
    payday: PaydayParams = field(default_factory=PaydayParams)
    pool: PoolParams = field(default_factory=PoolParams)
    econ: EconParams = field(default_factory=EconParams)
