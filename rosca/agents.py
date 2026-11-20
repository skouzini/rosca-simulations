"""Agent population: hourly-wage renters. Income is biweekly; rent hits on the first week of each month."""
import numpy as np

from .contracts import AgentState
from .params import WEEKS_PER_YEAR, PopulationParams


def make_population(p: PopulationParams, rng: np.random.Generator) -> AgentState:
    """Lognormal wage/rent/starting buffer around their medians. budget_mode 'margin' instead sets each
    agent's rent + essentials from their own pay so they run a thin surplus (see PopulationParams)."""
    n = p.cohort_size
    lognorm = lambda median, sigma: median * np.exp(rng.normal(0.0, sigma, n))
    wage = lognorm(p.wage_median, p.wage_sigma)
    rent = lognorm(p.rent_median, p.rent_sigma)
    essentials = np.full(n, p.essentials_monthly)
    margin = np.clip(rng.normal(p.margin_mean, p.margin_sd, n), 0.01, 0.25)  # drawn in both modes (same rng stream)
    if p.budget_mode == "margin":
        costs = wage * p.hours_mean * WEEKS_PER_YEAR / 12 * (1 - margin)
        rent, essentials = costs * p.rent_share, costs * (1 - p.rent_share)
    return AgentState(
        wage=wage,
        hours_mean=np.full(n, p.hours_mean),
        rent_monthly=rent,
        essentials_weekly=essentials * 12 / WEEKS_PER_YEAR,
        cash=lognorm(p.buffer_median, p.buffer_sigma),
        payday_propensity=np.full(n, p.payday_propensity),
        active=np.ones(n, dtype=bool),
    )


def is_payday(week: int) -> bool:
    return week % 2 == 1


def income(state: AgentState, week: int, hours_mult: np.ndarray, rng_draws: np.ndarray,
           hours_sd_frac: float = PopulationParams.hours_sd_frac) -> np.ndarray:
    """Biweekly pay on odd weeks covering 2 weeks of hours. rng_draws: standard normals (len n), drawn by
    the engine every week whether or not it is payday. hours_mult (len n) carries hours_cut/job_loss for
    the pay period (engine passes the mean over the two weeks)."""
    if not is_payday(week):
        return np.zeros(state.n)
    hours = np.clip(state.hours_mean * (1 + hours_sd_frac * rng_draws), 0, None)
    return state.wage * 2 * hours * hours_mult


def month_of(week: int) -> int:
    return week * 12 // WEEKS_PER_YEAR


def expenses(state: AgentState, week: int) -> np.ndarray:
    """Essentials every week, plus rent on the first week of each new month. Month 0's rent is assumed
    paid before the simulation starts (the starting buffer is cash left after it)."""
    rent_due = week > 0 and month_of(week) != month_of(week - 1)
    return state.essentials_weekly + (state.rent_monthly if rent_due else 0.0)
