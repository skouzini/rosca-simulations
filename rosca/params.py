"""All simulation parameters with defaults. Every value notes its source or says ASSUMPTION.

Sources: see docs/parameters.md for full links.
"""
from dataclasses import dataclass, field
from typing import Literal

WEEKS_PER_YEAR = 52


@dataclass
class PopulationParams:
    cohort_size: int = 12                 # business draft: 10–20 per cohort
    wage_median: float = 17.0             # $/hr. ASSUMPTION — verify BLS OEWS Salt Lake MSA
    wage_sigma: float = 0.20              # lognormal sigma. ASSUMPTION
    hours_mean: float = 32.0              # hrs/wk. ASSUMPTION
    hours_sd_frac: float = 0.15           # weekly hours noise; JPMC: median monthly income CV 0.38
    rent_median: float = 1350.0           # $/mo. ASSUMPTION — verify HUD FMR / ACS
    rent_sigma: float = 0.15
    essentials_monthly: float = 900.0     # $/mo non-rent essentials. ASSUMPTION
    buffer_median: float = 600.0          # starting liquid cash. SHED 2024: 37% can't cover $400 (calibrated; lognormal σ=1 → 34% < $400)
    buffer_sigma: float = 1.0
    payday_propensity: float = 0.5        # P(take payday | short). ASSUMPTION (present bias) (calibrated)
    strategic_default_prob: float = 0.01  # per-cycle walk-away after payout. MAF ~0.7–1%


@dataclass
class ShockParams:
    expense_rate_per_year: float = 2.5    # Poisson rate. ASSUMPTION, calibrate
    expense_median: float = 400.0         # SHED $400 benchmark
    expense_sigma: float = 1.256          # gives p90 ≈ $2,000
    job_loss_rate_per_year: float = 0.05  # ASSUMPTION
    job_loss_weeks_mean: float = 8.0      # ASSUMPTION


@dataclass
class PaydayParams:
    loan_size: float = 375.0              # Pew average
    fee_per_100: float = 21.0             # per 2-wk term ≈ 554% APR, Utah DFI 2020
    term_weeks: int = 2
    max_weeks: int = 10                   # Utah Code 7-23: 10-week limit, then extended payment plan
    epp_weeks: int = 8                    # extended payment plan length (interest-free installments)
    renew_prob: float = 0.8               # CFPB 2014: >80% renewed within 14 days
    default_prob_when_unable: float = 0.3  # P(default | can't pay or renew). calibrate to ~20%/yr borrowers (calibrated)


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
    contribution_weekly: float = 50.0     # ASSUMPTION; pot = n * 2 * this per round
    round_weeks: int = 2                  # one payout every 2 weeks
    miss_policy: Literal["expel", "demote"] = "demote"
    expel_after_misses: int = 2
    collapse_min_frac: float = 0.5        # pool collapses if active members < this share of start
    seed_liquidity: float = 0.0           # institutional seed, $ (Solution Direction assumption #3)
    # B and C
    reserve_frac: float = 0.20            # share of each contribution routed to reserve (α)
    target_reserve_pots: float = 1.0      # target reserve = this × one pot
    draw_cap: float = 500.0               # max outstanding emergency draw per member. ASSUMPTION
    draw_repay_weeks: int = 8             # emergency draw repaid evenly over this many weeks
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
