"""Compare structures × emergency-fund designs × populations under a shock preset.

    python3 -m experiments.compare [preset] [n_runs]
"""
import dataclasses as dc
import sys

import pandas as pd

from rosca.contracts import PRESETS
from rosca.engine import run_many
from rosca.metrics import summarize_many
from rosca.params import POPULATIONS, STRUCTURES, SimParams

FUNDS = {  # emergency-fund design: (personal_share_frac, reserve_frac). A ignores reserve_frac.
    "none": (0.0, 0.0),
    "personal 25%": (0.25, 0.0),
    "pooled 20%": (0.0, 0.20),
    "both 15%+15%": (0.15, 0.15),
}
COLS = ["payday_fees_per_member", "share_used_payday", "hardship_weeks", "survived", "member_recovered",
        "member_mttr_weeks", "recovered", "mttr_weeks",
        "miss_rate", "frozen_weeks"]


def compare(preset: str = "Plant layoff", n_runs: int = 200) -> pd.DataFrame:
    base, out = SimParams(), []
    for pop, (_, _, overrides) in POPULATIONS.items():
        for fund, (beta, alpha) in FUNDS.items():
            p = dc.replace(base, population=dc.replace(base.population, **overrides),
                           pool=dc.replace(base.pool, personal_share_frac=beta, reserve_frac=alpha))
            structs = [s for s in STRUCTURES if s != "no_pool" or fund == "none"]
            s = summarize_many(run_many(p, structs, PRESETS[preset], n_runs), p.pool, p.population.cohort_size,
                               calm=run_many(p, structs, [], n_runs))
            out.append(s[["structure"] + COLS].groupby("structure").mean().assign(population=pop, fund=fund))
    return pd.concat(out).reset_index().set_index(["population", "fund", "structure"])


if __name__ == "__main__":
    preset = sys.argv[1] if len(sys.argv) > 1 else "Plant layoff"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    pd.set_option("display.width", 200)
    print(f"Preset: {preset}, {n} runs per cell\n")
    print(compare(preset, n).round(2).to_string())
