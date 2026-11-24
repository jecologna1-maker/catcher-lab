from pathlib import Path

import pandas as pd
import streamlit as st

# 2026 cutoffs chosen to put one third of the expected misses in each bucket.
ROUTINE_CUTOFF = 0.06880060988531445
DIFFICULT_CUTOFF = 0.17454921870228157
BLOCKING_BUCKETS = ("Routine+", "Extended+", "Difficult+")


@st.cache_data(show_spinner=False, max_entries=4)
def load_blocking_pitches(path, version):
    pitches = pd.read_parquet(path)
    pitches["difficulty"] = pd.cut(
        pitches["expected_misplays"],
        [0, ROUTINE_CUTOFF, DIFFICULT_CUTOFF, 1],
        labels=BLOCKING_BUCKETS, right=False,
    )
    return pitches


def blocking_pitches(season):
    path = Path(__file__).with_name(f"blocking_pitches_{season}.parquet")
    file = path.stat()
    return load_blocking_pitches(path, (file.st_mtime_ns, file.st_size))


def blocking_rate_totals(pitches):
    grouped = pitches.assign(
        held=~pitches["misplay"], expected_blocks=1 - pitches["expected_misplays"],
    )
    rates = {
        "Block Rate": grouped.groupby("fielder_2").agg(
            actual=("held", "sum"), denominator=("held", "size"),
        )
    }
    for metric in BLOCKING_BUCKETS:
        rates[metric] = grouped.loc[grouped["difficulty"].eq(metric)].groupby("fielder_2").agg(
            actual=("held", "sum"), denominator=("expected_blocks", "sum"),
        )
    return rates
