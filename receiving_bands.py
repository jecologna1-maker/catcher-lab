from pathlib import Path

import pandas as pd
import streamlit as st

RECEIVING_BANDS = {
    "50/50%": (0.25, 0.75, "left", "25% to under 75%", "takes"),
    "Win+": (0.0, 0.249, "both", "24.9% and below", "expected"),
    "Loss+": (0.751, 1.0, "both", "75.1% and above", "expected"),
}


def receiving_band_mask(probability, metric):
    lower, upper, inclusive, _, _ = RECEIVING_BANDS[metric]
    return probability.between(lower, upper, inclusive=inclusive)


@st.cache_data(max_entries=7)
def receiving_band_totals(season, data_version):
    pitches = pd.read_parquet(
        Path(__file__).with_name(f"catcher_pitches_{season}.parquet"),
        columns=["fielder_2", "expected", "called"],
    )
    totals = {}
    for metric, (_, _, _, _, denominator) in RECEIVING_BANDS.items():
        band = pitches.loc[receiving_band_mask(pitches["expected"], metric)]
        grouped = band.groupby("fielder_2")
        totals[metric] = pd.DataFrame({
            "actual": grouped["called"].sum(),
            "denominator": grouped.size() if denominator == "takes" else grouped["expected"].sum(),
        })
    return totals
