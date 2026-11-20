"""Calibrate the status quo (no_pool, no user shocks) against payday-borrower benchmarks.

    python3 -m experiments.calibrate [n_seeds]
"""
import sys

import numpy as np
import pandas as pd

from rosca.engine import run
from rosca.params import SimParams

TARGETS = {  # metric: (target, source); "among borrowers" unless noted
    "loans_per_borrower": (8.0, "Pew 2012"),
    "weeks_in_debt_per_borrower": (22.0, "Pew 2012 (~5 months)"),
    "share_7plus_loans": (2 / 3, "CFPB 2013"),
    "share_defaulted": (0.20, "CFPB 2014"),
    "share_borrowing": (np.nan, "report only (all members)"),
    "hardship_weeks_per_member": (np.nan, "report only (all members)"),
}


def calibration_table(params: SimParams, n_seeds: int = 500) -> pd.DataFrame:
    members = []
    for seed in range(n_seeds):
        r = run(params, "no_pool", [], seed)
        members.append(r.members.assign(defaulted=r.defaulted))
    m = pd.concat(members, ignore_index=True)
    b = m[m.loans_taken > 0]
    achieved = {
        "loans_per_borrower": b.loans_taken.mean(),
        "weeks_in_debt_per_borrower": b.weeks_in_payday_debt.mean(),
        "share_7plus_loans": (b.loans_taken >= 7).mean(),
        "share_defaulted": b.defaulted.mean(),
        "share_borrowing": len(b) / len(m),
        "hardship_weeks_per_member": m.hardship_weeks.mean(),
    }
    t = pd.DataFrame({"achieved": achieved, "target": {k: v[0] for k, v in TARGETS.items()},
                      "source": {k: v[1] for k, v in TARGETS.items()}})
    t["rel_err"] = t.achieved / t.target - 1
    return t


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print(calibration_table(SimParams(), n).round(3).to_string())
