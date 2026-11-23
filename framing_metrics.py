"""Context-dependent value of a called strike relative to a called ball."""
import numpy as np
import pandas as pd

RUN_FIELDS = ["outs_when_up", "delta_run_exp", "events"]
STATE = ["bases", "outs_when_up", "balls", "strikes"]


def strike_values(takes):
    bases = sum(
        2**i * takes[base].notna().astype("int8")
        for i, base in enumerate(("on_1b", "on_2b", "on_3b"))
    )
    data = takes.assign(bases=bases)
    # Keep ordinary takes and count-ending walks/strikeouts. Exclude separate
    # baserunning events and blocked balls from the run-value reference sample.
    reference = data.loc[
        data["description"].isin(["ball", "called_strike"])
        & (data["events"].isna() | data["events"].isin(["walk", "strikeout"]))
        & data["delta_run_exp"].notna()
        & data["outs_when_up"].between(0, 2)
    ]
    by_state = reference.groupby([*STATE, "description"])["delta_run_exp"].agg(["median", "count"])
    medians = by_state["median"].unstack("description")
    counts = by_state["count"].unstack("description").fillna(0)
    by_count = reference.groupby(["balls", "strikes", "description"])["delta_run_exp"].median().unstack()
    states = pd.MultiIndex.from_product([range(8), range(3), range(4), range(3)], names=STATE)
    medians = medians.reindex(states)
    counts = counts.reindex(states, fill_value=0)
    for call in ("ball", "called_strike"):
        prior = by_count[call].reindex(states.droplevel(["bases", "outs_when_up"])).to_numpy()
        weight = counts[call].to_numpy() / (counts[call].to_numpy() + 20)
        medians[call] = weight * medians[call].fillna(0).to_numpy() + (1 - weight) * prior
    values = medians["ball"] - medians["called_strike"]
    if not np.isfinite(values).all() or values.le(0).any():
        raise ValueError("Unexpected nonpositive base-out-count strike value")
    weights = values.reindex(pd.MultiIndex.from_frame(data[STATE])).to_numpy()
    if not np.isfinite(weights).all():
        raise ValueError("Some takes have no base-out-count strike value")
    return weights
