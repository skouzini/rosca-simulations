# ROSCA Shared-Lending-Pool Simulations

Agent-based Monte Carlo simulations comparing savings-circle (ROSCA) structures against the payday-loan
status quo for hourly-wage workers in Utah. Each structure is scored on **max shock absorption** and
**mean time to baseline recovery**, plus payday, pool and sponsor-economics metrics.

Structures: payday only (baseline) · A. Rigid ROSCA · B. Variable payout by reserve health · C. Emergency reserve + deferred-payout staking.

## Run
```bash
pip install -r requirements.txt
pytest
streamlit run app.py
```

## Layout
- `rosca/params.py` — all parameters with sources (details in `docs/parameters.md`)
- `rosca/contracts.py` — shared data types and the weekly output schema
- `rosca/engine.py` — weekly tick loop and shortfall waterfall
- `rosca/pools.py` — pool structures
- `rosca/metrics.py` — recovery, absorption, economics
- `app.py` — interactive shock dashboard
