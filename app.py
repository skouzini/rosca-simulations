"""Interactive shock dashboard: apply a shock to a savings circle and watch each structure recover."""
import dataclasses
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rosca import engine, metrics, params as params_mod
from rosca.contracts import PRESETS, SHOCK_KINDS, ShockEvent
from rosca.params import STRUCTURE_LABELS, STRUCTURES, SimParams

st.set_page_config(page_title="Savings circle shock lab", page_icon="🛟", layout="wide")

# Status quo is a neutral baseline; the three structures take validated categorical slots in fixed order.
COLORS = {"no_pool": "#8a8984", "rigid": "#eb6834", "variable_payout": "#2a78d6", "reserve_staking": "#1baf7a"}
SHOCK_COLS = ["kind", "start_week", "duration_weeks", "share_hit", "magnitude", "label"]
MEMBER_STEPS = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000]


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


@st.cache_resource
def engine_is_fake() -> bool:
    try:
        engine.run(SimParams(weeks=4), "no_pool", [], 0)
        return False
    except NotImplementedError:
        return True


@st.cache_data(show_spinner="Running Monte Carlo…")
def simulate(params_json: str, structures: tuple, shocks: tuple, n_runs: int, seed: int):
    """Cache key = serialized params + structures + shocks + seed. Returns (long weekly df, summary df)."""
    d = json.loads(params_json)
    p = SimParams(weeks=d["weeks"], **{k: getattr(params_mod, type(getattr(SimParams(), k)).__name__)(**v)
                                       for k, v in d.items() if k != "weeks"})
    events = [ShockEvent(*s) for s in shocks]
    n = p.population.cohort_size
    results = engine.run_many(p, list(structures), events, n_runs, seed)
    frames = []
    for r in results:
        w = r.weekly[["week", "members_in_payday_debt", "reserve", "active_members"]].copy()
        w["health"] = metrics.health_index(r.weekly, p.pool, n).to_numpy()
        w["structure"], w["seed"] = r.structure, r.seed
        frames.append(w)
    summary = metrics.summarize_many(results, p.pool, n)
    return pd.concat(frames, ignore_index=True), summary


def bands(long: pd.DataFrame, col: str) -> pd.DataFrame:
    """p10 / median / p90 per structure × week."""
    q = long.groupby(["structure", "week"])[col].quantile([0.1, 0.5, 0.9]).unstack()
    q.columns = ["p10", "p50", "p90"]
    return q.reset_index()


def shade_shocks(fig, events, weeks):
    for i, e in enumerate(events):
        x1 = min(e.start_week + max(e.duration_weeks, 1), weeks - 1)
        fig.add_vrect(x0=e.start_week, x1=x1, fillcolor="#9aa0a6", opacity=0.13, line_width=0,
                      annotation_text=e.label or e.kind, annotation_position="top left" if i % 2 == 0 else "bottom left",
                      annotation_font_size=11, annotation_font_color="#5f6368")


def fan_chart(long, col, title, structures, events, weeks, recovery, yfmt=None):
    b = bands(long, col)
    fig = go.Figure()
    for s in structures:
        d = b[b.structure == s]
        if d.empty:
            continue
        c = COLORS[s]
        fig.add_trace(go.Scatter(x=np.r_[d.week, d.week[::-1]], y=np.r_[d.p90, d.p10[::-1]], fill="toself",
                                 fillcolor=rgba(c, 0.15), line=dict(width=0), hoverinfo="skip",
                                 legendgroup=s, showlegend=False))
        fig.add_trace(go.Scatter(x=d.week, y=d.p50, name=STRUCTURE_LABELS[s], legendgroup=s,
                                 line=dict(color=c, width=2),
                                 customdata=np.c_[d.p10, d.p90],
                                 hovertemplate="%{y:.2f} (p10 %{customdata[0]:.2f} – p90 %{customdata[1]:.2f})"))
        if s in recovery:
            fig.add_vline(x=recovery[s], line=dict(color=c, width=1.5, dash="dot"))
    shade_shocks(fig, events, weeks)
    fig.update_layout(title=dict(text=title, font_size=15), height=330, margin=dict(l=10, r=10, t=50, b=10),
                      hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", y=-0.2, font_size=11), xaxis_title="Week")
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    return fig


def replay_chart(long, structures, events, weeks):
    """Median health path animated week by week (Play button + slider)."""
    b = bands(long, "health")
    paths = {s: b[b.structure == s] for s in structures}
    step = max(1, weeks // 52)  # keep ≤ ~52 frames

    def traces(k):
        return [go.Scatter(x=paths[s].week[: k + 1], y=paths[s].p50[: k + 1], name=STRUCTURE_LABELS[s],
                           line=dict(color=COLORS[s], width=2.5)) for s in structures]

    ks = list(range(0, weeks, step)) + ([weeks - 1] if (weeks - 1) % step else [])
    fig = go.Figure(data=traces(weeks - 1), frames=[go.Frame(data=traces(k), name=str(k)) for k in ks])
    shade_shocks(fig, events, weeks)
    fig.update_layout(
        title=dict(text="Replay: median health index, week by week", font_size=15), height=380,
        template="plotly_white", margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(range=[0, weeks - 1], title="Week"), yaxis=dict(range=[0, 1.02], title="Health (0–1)"),
        legend=dict(orientation="h", y=1.12, font_size=11),
        updatemenus=[dict(type="buttons", x=0, y=-0.18, xanchor="left", showactive=False, buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=120, redraw=False), fromcurrent=True, transition=dict(duration=0))]),
            dict(label="❚❚ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(x=0.18, y=-0.12, len=0.82, currentvalue=dict(prefix="Week ", font=dict(color="#1f2a2e")),
                     font=dict(color="rgba(0,0,0,0)"), steps=[
            dict(label=str(k), method="animate",
                 args=[[str(k)], dict(frame=dict(duration=0, redraw=False), mode="immediate")]) for k in ks])],
    )
    return fig


def param_table(p: SimParams) -> pd.DataFrame:
    """Every field in rosca/params.py with its current value and the source comment from the file."""
    src = inspect.getsource(params_mod)
    group_of = {type(getattr(p, f.name)).__name__: f.name for f in dataclasses.fields(p) if f.name != "weeks"}
    rows, cls = [], None
    for line in src.splitlines():
        if m := re.match(r"class (\w+)", line):
            cls = m.group(1)
        elif cls and (m := re.match(r"\s+(\w+): [^=]+=[^#]*(?:#\s*(.*))?$", line)):
            name, note = m.group(1), (m.group(2) or "").strip()
            if cls == "SimParams":
                if name != "weeks":
                    continue
                val, group = p.weeks, "simulation"
            elif cls in group_of:
                group = group_of[cls]
                val = getattr(getattr(p, group), name)
            else:
                continue
            rows.append(dict(group=group, parameter=name, value=str(val), source=note or "—"))
    return pd.DataFrame(rows)


def events_from_editor(df: pd.DataFrame, weeks: int) -> list[ShockEvent]:
    out = []
    for r in df.to_dict("records"):
        if r.get("kind") not in SHOCK_KINDS or pd.isna(r.get("start_week")):
            continue
        num = lambda k, d: d if pd.isna(r.get(k)) else r[k]  # noqa: E731
        out.append(ShockEvent(r["kind"], int(np.clip(r["start_week"], 0, weeks - 1)),
                              int(max(num("duration_weeks", 1), 1)), float(np.clip(num("share_hit", 0.25), 0, 1)),
                              float(max(num("magnitude", 0.3), 0)), str(r.get("label") or "")))
    return out


def money(x: float) -> str:
    sign = "-" if x < 0 else ""  # leading "-" so st.metric deltas point the right way
    return f"{sign}${abs(x):,.0f}" if abs(x) < 1e6 else f"{sign}${abs(x) / 1e6:,.2f}M"

# ------------------------------------------------------------------ header
st.title("Savings circle shock lab")
st.markdown("Can a savings circle absorb a shock that would push members into **554% APR payday loans**? "
            "Apply a shock and watch each structure recover — or fail.")
if engine_is_fake():
    st.warning("Preview data: the simulation engine is still being built, so these results come from a "
               "placeholder generator (`engine.fake_run`). Shapes are illustrative, not findings.", icon="🧪")

# ------------------------------------------------------------------ sidebar
base = SimParams()
with st.sidebar:
    st.header("Scenario")
    structures = st.multiselect("Structures to compare", STRUCTURES, default=list(STRUCTURES),
                                format_func=STRUCTURE_LABELS.get)
    with st.expander("Cohort", expanded=True):
        cohort = st.slider("Members per circle", 6, 30, base.population.cohort_size)
        contrib = st.slider("Weekly contribution ($)", 10, 150, int(base.pool.contribution_weekly), step=5)
        miss = st.radio("When a member misses payments", ["demote", "expel"], horizontal=True,
                        index=["demote", "expel"].index(base.pool.miss_policy),
                        help="demote: moved to the back of the payout order · expel: removed after repeated misses")
    with st.expander("Structure B/C"):
        alpha = st.slider("Reserve fraction α", 0.0, 0.5, base.pool.reserve_frac, 0.05,
                          help="Share of each contribution routed to the emergency reserve")
        yld = st.slider("Staking yield (APR)", 0.0, 0.08, base.pool.staking_yield_apr, 0.005, format="%.3f")
        stake_p = st.slider("Stake probability", 0.0, 1.0, base.pool.stake_prob, 0.05,
                            help="Chance a member with no shortfall leaves their payout staked (C only)")
    with st.expander("Sponsor"):
        seed_liq = st.number_input("Seed liquidity per circle ($)", 0, 20_000, int(base.pool.seed_liquidity), step=250,
                                   help="Credit-union backstop available to each circle")
    with st.expander("Simulation"):
        weeks = st.select_slider("Horizon (weeks)", [52, 78, 104], value=52)
        n_runs = st.slider("Monte Carlo runs", 10, 500, 50, step=10)
        seed = st.number_input("Random seed", 0, 1_000_000, 0)

params = dataclasses.replace(
    base, weeks=weeks,
    population=dataclasses.replace(base.population, cohort_size=cohort),
    pool=dataclasses.replace(base.pool, contribution_weekly=float(contrib), reserve_frac=alpha,
                             staking_yield_apr=yld, stake_prob=stake_p, seed_liquidity=float(seed_liq),
                             miss_policy=miss))

# ------------------------------------------------------------------ shock builder
st.subheader("1 · Choose a shock")
left, right = st.columns([3, 2], gap="large")
with left:
    preset = st.selectbox("Start from a preset", list(PRESETS), index=list(PRESETS).index("Perfect storm"), key="preset")
    preset_df = pd.DataFrame([dataclasses.asdict(e) for e in PRESETS[preset]], columns=SHOCK_COLS)
    edited = st.data_editor(
        preset_df, key=f"shocks_{preset}", num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "kind": st.column_config.SelectboxColumn("Kind", options=list(SHOCK_KINDS), required=True),
            "start_week": st.column_config.NumberColumn("Start week", min_value=0, max_value=103, step=1),
            "duration_weeks": st.column_config.NumberColumn("Weeks", min_value=1, max_value=52, step=1),
            "share_hit": st.column_config.NumberColumn("Share hit", min_value=0.0, max_value=1.0, step=0.05),
            "magnitude": st.column_config.NumberColumn("Magnitude", min_value=0.0, step=0.05),
            "label": st.column_config.TextColumn("Label"),
        })
with right:
    st.markdown("**What “magnitude” means**")
    doc = ShockEvent.__doc__.split("by kind:", 1)[1]
    st.markdown("\n".join(f"- **{k}** — {v.strip()}" for k, v in re.findall(r"(\w+)\s+—\s+(.+)", doc)))
    st.caption("Share hit = fraction of members affected. The same members and shocks hit every structure.")

events = events_from_editor(edited, weeks)
cfg = (json.dumps(dataclasses.asdict(params), sort_keys=True), tuple(structures),
       tuple(tuple(dataclasses.astuple(e)) for e in events), int(n_runs), int(seed))
if st.button("Run simulation", type="primary", key="run") or "cfg" not in st.session_state:
    st.session_state.cfg = cfg
elif st.session_state.cfg != cfg:
    st.caption("Settings changed — press **Run simulation** to update the results.")

run_cfg = st.session_state.cfg
run_params_json, run_structs, run_shocks, run_n, run_seed = run_cfg
if not run_structs:
    st.info("Select at least one structure in the sidebar.")
    st.stop()
long, summary = simulate(*run_cfg)
run_weeks = json.loads(run_params_json)["weeks"]
run_cohort = json.loads(run_params_json)["population"]["cohort_size"]
run_events = [ShockEvent(*s) for s in run_shocks]
summary.attrs["cohort_size"] = run_cohort  # economics() placeholder reads this
structs = [s for s in STRUCTURES if s in run_structs]

# ------------------------------------------------------------------ tabs
st.subheader("2 · See what happens")
tab_shock, tab_econ, tab_assump, tab_sens = st.tabs(["Shock & recovery", "Impact & economics", "Assumptions", "Sensitivity"])

g = summary.groupby("structure")
shock_start = min((e.start_week for e in run_events), default=None)
mttr_med = g.mttr_weeks.median()
recovery = {s: shock_start + mttr_med[s] for s in structs} if shock_start is not None else {}
fees_pm = g.payday_fees.mean() / run_cohort

with tab_shock:
    st.caption(f"{run_n} Monte Carlo runs per structure · bands show the middle 80% of runs (p10–p90), line is the median · "
               "grey bands mark shock windows · dotted lines mark each structure's median recovery week.")
    for s in structs:
        rec_rate = g.recovered.mean()[s]
        never = 1 - rec_rate
        st.markdown(f"<span style='color:{COLORS[s]};font-size:1.2em'>●</span> **{STRUCTURE_LABELS[s]}**",
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Time to recover (median)", "—" if shock_start is None else f"{mttr_med[s]:.0f} wks",
                  help="Weeks from first shock until the health index is back within 5% of its pre-shock level "
                       "for 4 straight weeks.")
        c1.caption("no shock applied" if shock_start is None else f"{never:.0%} never recovered in horizon")
        c2.metric("Survival rate", "n/a" if s == "no_pool" else f"{g.survived.mean()[s]:.0%}",
                  help="Share of runs with no collapse and no frozen week.")
        base_fee = fees_pm.get("no_pool")
        c3.metric("Payday fees per member", money(fees_pm[s]),
                  delta=None if base_fee is None or s == "no_pool" else f"{money(fees_pm[s] - base_fee)} vs status quo",
                  delta_color="inverse", help=f"Mean payday-loan fees per member over {run_weeks} weeks.")
        c4.metric("Self-coverage rate", f"{g.self_coverage_rate.mean()[s]:.0%}",
                  help="Share of expense shocks covered without a payday loan or missed bill.")

    pool_structs = [s for s in structs if s != "no_pool"]
    charts = [("health", "Pool health index (0–1)", structs, ".0%"),
              ("members_in_payday_debt", "Members in payday debt", structs, None),
              ("reserve", "Pool reserve ($)", [s for s in pool_structs if s != "rigid"], "$,.0f"),
              ("active_members", "Active members", pool_structs, None)]
    for row in (charts[:2], charts[2:]):
        cols = st.columns(2)
        for col, (c, title, ss, fmt) in zip(cols, row):
            with col:
                if ss:
                    st.plotly_chart(fan_chart(long, c, title, ss, run_events, run_weeks, recovery, fmt),
                                    width="stretch")
    st.plotly_chart(replay_chart(long, structs, run_events, run_weeks), width="stretch")

with tab_econ:
    st.markdown("#### What this looks like at scale")
    members = st.select_slider("Members enrolled", MEMBER_STEPS, value=10_000, format_func=lambda v: f"{v:,}")
    econ = metrics.economics(summary, params.econ, members)
    choices = [s for s in structs if s != "no_pool" and s in econ.index]
    if not choices:
        st.info("Select at least one savings-circle structure to compare against the status quo.")
    else:
        pick = st.radio("Structure", choices, index=len(choices) - 1, format_func=STRUCTURE_LABELS.get, horizontal=True)
        e = econ.loc[pick]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Payday fees avoided / yr", money(e.payday_fees_avoided))
        k1.caption("Status-quo payday fees per member minus this structure's, × members.")
        k2.metric("Sponsor capital at risk (P95)", money(e.sponsor_capital_at_risk))
        k2.caption("95th-percentile peak seed liquidity drawn per circle × number of circles.")
        k3.metric("Float revenue / yr", money(e.float_revenue))
        k3.caption(f"Average pooled balance × {params.econ.tbill_yield:.0%} T-bill yield.")
        k4.metric("Net / yr", money(e.net))
        k4.caption(f"Float + fees (\\${params.econ.pepm_fee:.0f} PEPM) − opex (\\${params.econ.opex_per_member_month:.0f}/member/mo).")

        cols = ["payday_fees_avoided", "float_revenue", "net", "sponsor_capital_at_risk"]
        nice = {"payday_fees_avoided": "Fees avoided", "float_revenue": "Float revenue", "net": "Net",
                "sponsor_capital_at_risk": "Capital at risk (P95)"}
        fig = go.Figure()
        for s in choices:
            fig.add_trace(go.Bar(x=[nice[c] for c in cols], y=[econ.loc[s, c] for c in cols], name=STRUCTURE_LABELS[s],
                                 marker_color=COLORS[s], marker_line=dict(color="#ffffff", width=2),
                                 hovertemplate="%{x}: $%{y:,.0f}<extra>" + STRUCTURE_LABELS[s] + "</extra>"))
        fig.update_layout(barmode="group", template="plotly_white", height=380, margin=dict(l=10, r=10, t=40, b=10),
                          title=dict(text=f"Annual economics at {members:,} members", font_size=15),
                          yaxis_tickformat="$,.2s", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        with st.expander("Full table"):
            st.dataframe(econ.rename(index=STRUCTURE_LABELS).style.format("${:,.0f}"), width="stretch")

with tab_assump:
    st.markdown("Every parameter in use, with the source note from `rosca/params.py`. "
                "Full citations are in `docs/parameters.md`. **ASSUMPTION** marks values still to be calibrated.")
    st.dataframe(param_table(params), hide_index=True, width="stretch",
                 column_config={"source": st.column_config.TextColumn("source", width="large")})

with tab_sens:
    path = Path(__file__).parent / "results" / "sensitivity.parquet"
    if path.exists():
        st.markdown("Sensitivity sweep results from `results/sensitivity.parquet`.")
        st.dataframe(pd.read_parquet(path), width="stretch")
    else:
        st.info("Sensitivity results load from `results/sensitivity.parquet` when present. "
                "Run the sensitivity experiment to generate them.")
