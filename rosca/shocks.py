"""Shocks: idiosyncratic expense shocks and job losses (drawn up front), plus user ShockEvents."""
from dataclasses import dataclass

import numpy as np

from .contracts import ShockEvent
from .params import WEEKS_PER_YEAR, ShockParams


@dataclass
class Idiosyncratic:
    expense: np.ndarray     # [weeks, n] $ of unexpected expense
    hours_mult: np.ndarray  # [weeks, n] 0 while job lost, else 1


def draw_idiosyncratic(p: ShockParams, n: int, weeks: int, rng: np.random.Generator) -> Idiosyncratic:
    counts = rng.poisson(p.expense_rate_per_year / WEEKS_PER_YEAR, (weeks, n))
    sizes = rng.lognormal(np.log(p.expense_median), p.expense_sigma, counts.sum())
    expense = np.zeros((weeks, n))
    np.add.at(expense, _cells(counts), sizes)
    # job loss: Bernoulli start per agent-week, geometric duration with the given mean (≥1 week)
    starts = rng.random((weeks, n)) < p.job_loss_rate_per_year / WEEKS_PER_YEAR
    durations = rng.geometric(1 / max(p.job_loss_weeks_mean, 1.0), (weeks, n))
    hours_mult = np.ones((weeks, n))
    for w, i in zip(*np.nonzero(starts)):
        hours_mult[w:w + durations[w, i], i] = 0.0
    return Idiosyncratic(expense, hours_mult)


def _cells(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(week, agent) index for each individual shock, so sizes can be summed into their cell."""
    flat = np.repeat(np.arange(counts.size), counts.ravel())
    return np.unravel_index(flat, counts.shape)


@dataclass
class EventEffects:
    hours_mult: np.ndarray            # [weeks, n]
    extra_expense: np.ndarray         # [weeks, n]
    exits_by_week: dict               # week -> agent idx array (engine skips already-inactive ones)
    reserve_loss_frac: np.ndarray     # [weeks]
    shock_active: np.ndarray          # [weeks] bool


def apply_events(events: list[ShockEvent], n: int, weeks: int, rng: np.random.Generator) -> EventEffects:
    """Translate user events into arrays. Members hit = round(share_hit * n) drawn from all n members with
    rng (independent of pool state, so identical across structures). For member_exit the engine drops
    those already inactive, so 'share of active members' is approximated by share of the starting cohort."""
    fx = EventEffects(np.ones((weeks, n)), np.zeros((weeks, n)), {}, np.zeros(weeks), np.zeros(weeks, bool))
    for e in events:
        hit = np.sort(rng.choice(n, int(round(e.share_hit * n)), replace=False))
        s, end = e.start_week, min(e.start_week + max(e.duration_weeks, 1), weeks)
        if s >= weeks:
            continue
        fx.shock_active[s:end] = True
        if e.kind == "hours_cut":
            fx.hours_mult[s:end, hit] *= 1 - e.magnitude
        elif e.kind == "job_loss":
            fx.hours_mult[s:end, hit] = 0.0
        elif e.kind == "expense_spike":
            fx.extra_expense[s, hit] += e.magnitude
        elif e.kind == "member_exit":
            fx.exits_by_week[s] = np.union1d(fx.exits_by_week.get(s, np.array([], int)), hit)
        elif e.kind == "reserve_loss":
            fx.reserve_loss_frac[s] = 1 - (1 - fx.reserve_loss_frac[s]) * (1 - e.magnitude)
    return fx
