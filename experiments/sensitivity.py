"""Which factors drive shock resistance and recovery? Latin-hypercube sensitivity analysis.

For each sampled scenario, run no_pool, A, B, C on the same seeds (common random numbers) with the shock
preset and without it, then record per structure:
  fee_reduction   1 − payday fees / payday-only fees (same scenario, same seeds)   ← shock resistance
  survival        share of runs with no collapse and ≤10% frozen weeks             ← shock resistance
  member_mttr     median weeks until members' payday debt/hardship is back to the no-shock run ← recovery
  member_recovered share of runs that recovered within the horizon
  pool_mttr       median weeks until the pool health index is back to the no-shock run
Factor effects = Spearman rank correlation of each factor with each outcome, per structure.

    python3 -m experiments.sensitivity [n_samples=300] [n_runs=20] [preset="Plant layoff"]

Writes results/sensitivity.parquet (one row per sample × structure) and results/sensitivity_effects.parquet.
"""
import dataclasses as dc
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

from rosca.contracts import PRESETS
from rosca.engine import run_many
from rosca.metrics import summarize_many
from rosca.params import SimParams

OUT = Path(__file__).resolve().parent.parent / "results"
STRUCTS = ["no_pool", "rigid", "variable_payout", "reserve_staking"]

# name: (params section, field, low, high, kind, group)
FACTORS = {
    "contribution_weekly": ("pool", "contribution_weekly", 10, 75, float, "design"),
    "cohort_size": ("population", "cohort_size", 6, 20, int, "design"),
    "personal_share_frac": ("pool", "personal_share_frac", 0.0, 0.4, float, "design"),
    "reserve_frac": ("pool", "reserve_frac", 0.0, 0.4, float, "design"),
    "pot_advance": ("pool", "pot_advance", 0, 1, bool, "design"),
    "draw_cap": ("pool", "draw_cap", 250, 2500, float, "design"),
    "seed_liquidity": ("pool", "seed_liquidity", 0, 5000, float, "design"),
    "payout_floor": ("pool", "payout_floor", 0.5, 0.9, float, "design"),
    "stake_prob": ("pool", "stake_prob", 0.0, 1.0, float, "design"),
    "expense_rate_per_year": ("shocks", "expense_rate_per_year", 0.6, 2.4, float, "environment"),
    "job_loss_rate_per_year": ("shocks", "job_loss_rate_per_year", 0.04, 0.2, float, "environment"),
    "buffer_median": ("population", "buffer_median", 200, 1500, float, "environment"),
    "margin_mean": ("population", "margin_mean", 0.02, 0.15, float, "environment"),
    "payday_propensity": ("population", "payday_propensity", 0.3, 0.9, float, "environment"),
}
OUTCOMES = ["fee_reduction", "survival", "member_mttr", "member_recovered", "pool_mttr"]


def sample_factors(n: int, seed: int = 0) -> pd.DataFrame:
    u = qmc.LatinHypercube(d=len(FACTORS), seed=seed).random(n)
    cols = {}
    for j, (name, (_, _, lo, hi, kind, _)) in enumerate(FACTORS.items()):
        x = lo + u[:, j] * (hi - lo)
        cols[name] = (u[:, j] >= 0.5) if kind is bool else np.round(x).astype(int) if kind is int else x
    return pd.DataFrame(cols)


def params_for(row: dict, base: SimParams) -> SimParams:
    """Timing-mismatch population (the target group) with the sampled factors applied."""
    secs = {"pool": {}, "population": {"budget_mode": "margin"}, "shocks": {}}
    for name, (sec, field, *_rest) in FACTORS.items():
        v = row[name]
        secs[sec][field] = bool(v) if isinstance(v, (bool, np.bool_)) else v.item() if hasattr(v, "item") else v
    return dc.replace(base, **{s: dc.replace(getattr(base, s), **kw) for s, kw in secs.items()})


def evaluate(args) -> list[dict]:
    i, row, n_runs, preset = args
    p = params_for(row, SimParams())
    shocks = PRESETS[preset]
    s = summarize_many(run_many(p, STRUCTS, shocks, n_runs, seed=1000 * i), p.pool, p.population.cohort_size,
                       calm=run_many(p, STRUCTS, [], n_runs, seed=1000 * i))
    g = s.groupby("structure")
    base_fees = g.payday_fees.mean()["no_pool"]
    out = []
    for st in STRUCTS[1:]:
        out.append(dict(sample=i, structure=st, **row,
                        fee_reduction=1 - g.payday_fees.mean()[st] / base_fees if base_fees > 0 else 0.0,
                        survival=g.survived.mean()[st], member_mttr=g.member_mttr_weeks.median()[st],
                        member_recovered=g.member_recovered.mean()[st], pool_mttr=g.mttr_weeks.median()[st]))
    return out


def run_sensitivity(n_samples: int = 300, n_runs: int = 20, preset: str = "Plant layoff",
                    workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = sample_factors(n_samples)
    jobs = [(i, X.iloc[i].to_dict(), n_runs, preset) for i in range(n_samples)]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for chunk in ex.map(evaluate, jobs, chunksize=4) for r in chunk]
    df = pd.DataFrame(rows)
    df.attrs.update(preset=preset, n_runs=n_runs)
    return df, effects(df)


def effects(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho of every factor with every outcome, per structure (NaN if the factor is constant)."""
    rows = []
    for st, d in df.groupby("structure"):
        for f, (*_, group) in FACTORS.items():
            for o in OUTCOMES:
                x, y = d[f].astype(float), d[o].astype(float)
                ok = x.notna() & y.notna()
                rho = spearmanr(x[ok], y[ok]).statistic if ok.sum() > 5 and x[ok].nunique() > 1 and y[ok].nunique() > 1 else np.nan
                rows.append(dict(structure=st, factor=f, group=group, outcome=o, rho=rho))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    preset = sys.argv[3] if len(sys.argv) > 3 else "Plant layoff"
    df, eff = run_sensitivity(n, runs, preset)
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / "sensitivity.parquet")
    eff.to_parquet(OUT / "sensitivity_effects.parquet")
    pd.set_option("display.width", 200)
    for o in OUTCOMES:
        t = eff[eff.outcome == o].pivot(index="factor", columns="structure", values="rho")
        print(f"\n== Spearman rho with {o} ({preset}, {n} samples × {runs} runs)")
        print(t.reindex(t.abs().max(axis=1).sort_values(ascending=False).index).round(2).to_string())
