
import json

import streamlit as st
import pandas as pd
import requests
from datetime import date
from io import StringIO
from pathlib import Path
from streamlit.components.v1 import declare_component

from catcher_profiles import catcher_profile_url, has_receiving_data, show_catcher_profile
from receiving_bands import RECEIVING_BANDS, receiving_band_totals
from sl_plus_leaderboard import show_sl_plus_leaderboard
from framing_video_review import show_blocking_review, show_framing_review

st.set_page_config(
    page_title="Catcher Lab",
    page_icon="⚾",
    layout="wide"
)

st.html(Path(__file__).with_name("catcher_lab.css"))
st.html(
    '<header class="catcher-brand" aria-label="Catcher Lab">'
    '<span class="catcher-brand__mark" aria-hidden="true">JC</span>'
    '<div class="catcher-brand__identity">'
    '<span class="catcher-brand__name">Cologna Catching Development</span>'
    '<h1>Catcher Lab</h1></div>'
    '<span class="catcher-brand__product">MLB Catching Analytics</span>'
    '</header>'
)

receiving_table = declare_component(
    "receiving_table", path=str(Path(__file__).with_name("receiving_table")),
)
SEASONS = list(range(2026, 2019, -1))


def open_catcher_profile(catcher, season):
    st.session_state["catcher_nav"] = "Catcher Profiles"
    st.session_state[f"catcher_profile_{season}"] = catcher


DATASETS = {
    "Throwing": "catcher-throwing",
    "Pop Time": "poptime"
}

@st.cache_data(ttl=21600)
def load_data(category, season):
    endpoint = DATASETS[category]
    url = (
        "https://baseballsavant.mlb.com/"
        f"leaderboard/{endpoint}"
    )

    response = requests.get(
        url,
        params={
            "year": season,
            "csv": "true"
        },
        timeout=30
    )

    response.raise_for_status()

    return pd.read_csv(
        StringIO(response.text)
    )

@st.cache_data(ttl=21600, max_entries=32)
def load_pitches(start, end):
    response = requests.get(
        "https://baseballsavant.mlb.com/statcast_search/csv",
        params={
            "all": "true",
            "type": "details",
            "hfGT": "R|",
            "game_date_gt": start.isoformat(),
            "game_date_lt": end.isoformat(),
        },
        timeout=30,
    )
    response.raise_for_status()
    return pd.read_csv(StringIO(response.content.decode("utf-8-sig")))


def show_results(df, identity_columns, filename, widget_key):
    statistics = [column for column in df.columns if column not in identity_columns]
    visible = st.sidebar.multiselect(
        "Visible columns", statistics, default=statistics, key=widget_key
    )
    results = df[identity_columns + visible]
    st.metric("Records", len(results))
    st.dataframe(results, width="stretch", hide_index=True)
    st.download_button(
        "Download CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def show_blocking(season, search):
    if season != 2026:
        st.info("Blocking is available for the 2026 regular season.")
        return

    st.caption(
        "Tracked pitches projected to touch dirt by the back edge of the batter’s box. "
        "A pitch qualifies with any runner on base, or with the bases empty on a two-strike swing-and-miss."
    )
    with st.expander("How blocking is calculated"):
        st.write(
            "The pitch trajectory estimates where the ball would touch the ground. "
            "The model estimates the chance of a scored wild pitch or passed ball from "
            "location, projected depth below ground at the cutoff, speed, movement, handedness, "
            "and the runner and out situation. Catcher identity is excluded, and "
            "each game is scored using a model trained on other games."
        )
        st.write(
            "Block+ is 100 × actual blocks ÷ expected blocks. A score of 100 means the catcher "
            "blocked as many pitches as expected. Missed Blocks counts scored wild pitches and "
            "passed balls, and xMissed Blocks adds up their predicted probabilities. "
            "Blocks Above Average is xMissed Blocks minus Missed Blocks. "
            "A score of +5 means five fewer scored misplays than expected given the pitches received. "
            "A pitch without a scored misplay can still have bounced away without anyone advancing."
        )
    minimum = st.sidebar.number_input("Minimum Block Opportunities", min_value=0, value=50, step=10)
    frame = pd.read_csv(Path(__file__).with_name(f"dirt_blocking_{season}.csv"))
    frame = frame.loc[frame["opportunities"].ge(minimum)].copy()
    frame["blocks"] = frame["opportunities"] - frame["actual_misplays"]
    frame["expected_blocks"] = frame["opportunities"] - frame["expected_misplays"]
    frame["block_plus"] = 100 * frame["blocks"] / frame["expected_blocks"]
    qualified = frame
    if search:
        frame = frame.loc[frame["player"].str.contains(search, case=False, regex=False, na=False)]
    columns = {
        "player": "Player",
        "opportunities": "Block Opportunities",
        "block_plus": "Block+",
        "blocks_above_average": "Blocks Above Average",
        "actual_misplays": "Missed Blocks",
        "expected_misplays": "xMissed Blocks",
    }
    frame = frame.sort_values("block_plus", ascending=False)
    st.caption(
        "Select a player for their profile or a number for pitch video and location, lowest WP/PB "
        "probability first. Missed Blocks opens only scored wild pitches and passed balls."
    )
    show_catcher_table(
        frame, qualified, columns, season, minimum, search, category="Blocking",
        integer_columns=["Block Opportunities", "Missed Blocks"], review=show_blocking_review,
    )


def show_leaderboard(season):
    category = st.sidebar.radio("Category", ["Receiving", "Blocking", *DATASETS])
    search = st.sidebar.text_input("Search player").strip()
    st.header(f"{season} MLB {category}")
    if category == "Blocking":
        show_blocking(season, search)
        return
    if category == "Receiving":
        if has_receiving_data(season):
            show_sl_plus(search, season)
        else:
            st.info(f"Catcher Lab receiving scores are unavailable for the {season} season.")
        return

    with st.spinner("Loading Statcast data..."):
        df = load_data(category, season)

    name_column = next(
        (column for column in ("player_name", "name", "entity_name") if column in df),
        None,
    )
    team_column = next(
        (column for column in ("team_name", "team", "team_id") if column in df),
        None,
    )
    identity_columns = [
        column for column in (name_column, team_column) if column is not None
    ]

    if team_column is not None:
        teams = sorted(df[team_column].dropna().unique())
        selected = st.sidebar.multiselect(
            "Teams", teams, key=f"teams_{category}_{season}"
        )
        if selected:
            df = df[df[team_column].isin(selected)]

    if search and name_column is not None:
        names = df[name_column].astype("string")
        parts = names.str.partition(", ")
        first_last = (parts[2] + " " + parts[0]).str.strip()
        df = df[
            names.str.contains(search, case=False, regex=False, na=False)
            | first_last.str.contains(search, case=False, regex=False, na=False)
        ]

    show_results(df, identity_columns, f"{category}_{season}.csv", f"columns_{category}")


def show_sl_plus(search, season):
    st.subheader("Catcher jcSL+")
    st.caption(
        f"{season} regular season. 100 × actual called strikes ÷ expected called strikes. "
        "100 means the catcher received as many called strikes as expected."
    )
    with st.expander("How the framing metrics are calculated"):
        st.write(
            "Catcher Lab estimates called-strike probabilities from tracked takes "
            "using pitch location, batter zone height, count, batter and pitcher "
            "handedness, velocity, horizontal and vertical approach angles, "
            "horizontal and vertical movement, spin rate and axis, and the home-plate "
            "umpire. Each game is scored by a model trained on other games in the same season. "
            "Catcher identity is excluded from the model."
        )
        if season == 2026:
            st.write("For 2026, scoring uses the initial umpire call before any ABS overturn.")
        st.write(
            "jcSLAA (Strikes Looking Above Average) adds actual minus expected called strikes across pitches. "
            "jcFRAA (Framing Runs Above Average) weights that difference by the value of changing "
            "a ball to a strike for the pre-pitch runners, outs, and count. "
            "Run weights use the difference between Savant’s median run-expectancy changes "
            "for balls and called strikes, with sparse states pulled toward their count average."
        )
        review = json.loads(Path(__file__).with_name(f"called_strike_model_{season}.json").read_text())
        baseline = review["baseline"]["edge"]
        challenger = review["challenger"]["edge"]
        st.caption(
            f"On {challenger['pitches']:,} borderline takes from games starting {review['test_start']}, "
            f"Brier score improved from {baseline['brier']:.4f} to {challenger['brier']:.4f}. "
            "Lower scores indicate more accurate probabilities."
        )
    minimum = st.sidebar.number_input(
        "Minimum eligible takes", min_value=0, value=200, step=50
    )
    df = pd.read_csv(Path(__file__).with_name(f"sl_plus_{season}.csv"))
    df = df.loc[df["eligible_takes"].ge(minimum)]
    data_file = Path(__file__).with_name(f"catcher_pitches_{season}.parquet")
    bands = receiving_band_totals(season, data_file.stat().st_mtime_ns)
    for metric, totals in bands.items():
        totals = totals.reindex(df["catcher_id"])
        df[metric] = (100 * totals["actual"] / totals["denominator"]).to_numpy()
    qualified = df
    if search:
        df = df.loc[df["player"].str.contains(search, case=False, regex=False, na=False)]
    df = df.sort_values("SL+", ascending=False)
    columns = {
        "player": "Player",
        "eligible_takes": "Total Pitches",
        "SL+": "jcSL+",
        **{metric: metric for metric in RECEIVING_BANDS},
        "SLAA": "jcSLAA",
        "Framing Runs Above Average": "jcFRAA",
    }
    st.caption(
        "Select a player for their profile or a number for pitch video and location "
        "50/50% is the actual strike rate from 25% to under 75% predicted probability. "
        "Win+ (75.1% and above) and Loss+ (24.9% and below) compare actual to expected strikes."
    )
    show_catcher_table(
        df, qualified, columns, season, minimum, search, category="Receiving",
        integer_columns=["Total Pitches"], review=show_framing_review, bands=bands,
    )


def show_catcher_table(df, qualified, columns, season, minimum, search, *, category, integer_columns, review, bands=None):
    results = df[list(columns)].rename(columns=columns)
    rate_columns = (
        ["called_strikes", "expected_strikes"] if category == "Receiving"
        else ["blocks", "expected_blocks"]
    )
    rate_totals = df[rate_columns].to_numpy().tolist()
    band_totals = {
        metric: table.reindex(df["catcher_id"]).fillna(0).to_numpy().tolist()
        for metric, table in (bands or {}).items()
    }
    rows = [
        {
            "id": int(catcher),
            "profile": catcher_profile_url(season, catcher),
            "values": [value if pd.notna(value) else None for value in values],
            "bandTotalsByMetric": {metric: totals[position] for metric, totals in band_totals.items()},
            "strikeTotals": totals if category == "Receiving" else None,
            "blockTotals": totals if category == "Blocking" else None,
        }
        for position, (catcher, values, totals) in enumerate(zip(
            df["catcher_id"], results.itertuples(index=False, name=None), rate_totals,
        ))
    ]
    summary = qualified[list(columns)[1:]]
    benchmarks = None if summary.empty else {
        "mean": [None, *[float(v) if pd.notna(v) else None for v in summary.mean()]],
        "percentile": [None, *[float(v) if pd.notna(v) else None for v in summary.quantile(0.9)]],
    }
    event = receiving_table(
        rows=rows, headers=list(results.columns), benchmarks=benchmarks, default=None,
        table_label=f"{category} Leaderboard", integer_columns=integer_columns,
        key=f"{category}_{season}_{minimum}_{search}",
    )
    event_key = f"last_{category}_click"
    if event and event != st.session_state.get(event_key):
        st.session_state[event_key] = event
        matching = df.loc[df["catcher_id"].eq(event["catcher"])]
        if not matching.empty:
            row = matching.iloc[0]
            column = event["column"]
            if column == "Player":
                st.session_state["pending_profile"] = (int(row["catcher_id"]), season)
                st.rerun()
            elif column in results.columns:
                review(int(row["catcher_id"]), row["player"], column, season)
    report = "framing" if category == "Receiving" else "blocking"
    st.download_button(
        "Download CSV", results.round(2).to_csv(index=False),
        file_name=f"catcher_{report}_{season}.csv", mime="text/csv",
    )


PITCH_COLUMNS = {
    "game_date": "Game date",
    "fielder_2": "Catcher MLB ID",
    "fielding_team": "Fielding team",
    "pitch_name": "Pitch type",
    "description": "Pitch result",
    "release_speed": "Velocity (mph)",
    "plate_x": "Plate X (ft)",
    "plate_z": "Plate Z (ft)",
    "balls": "Balls",
    "strikes": "Strikes",
    "inning": "Inning",
    "pitcher": "Pitcher MLB ID",
    "batter": "Batter MLB ID",
    "game_pk": "Game ID",
    "at_bat_number": "Plate appearance number",
    "pitch_number": "Pitch number",
}


def show_pitch_search(season):
    st.header(f"{season} Statcast Pitch Search")
    st.caption("Regular-season pitches. Choose a date range of up to 7 days.")

    default_pitch_day = date(season, 7, 23) if season == 2020 else date(season, 6, 1)
    with st.sidebar.form(f"pitch_dates_{season}"):
        dates = st.date_input(
            "Game dates",
            value=(default_pitch_day, default_pitch_day),
            min_value=date(season, 1, 1),
            max_value=date(season, 12, 31),
        )
        submitted = st.form_submit_button("Load pitches")

    if submitted:
        if len(dates) != 2 or (dates[1] - dates[0]).days >= 7:
            st.warning("Select a start and end date within a 7-day range.")
            return
        st.session_state[f"pitch_range_{season}"] = dates

    active_dates = st.session_state.get(f"pitch_range_{season}")
    if active_dates is None:
        return

    start, end = active_dates
    st.write(f"Game dates: {start:%b %d, %Y} to {end:%b %d, %Y}")
    with st.spinner("Loading pitch-level Statcast data..."):
        pitches = load_pitches(start, end)

    if pitches.empty:
        st.info("No regular-season pitches found for those dates.")
        return

    pitches = pitches.dropna(subset=["fielder_2"]).copy()
    pitches["fielder_2"] = pitches["fielder_2"].astype(int)
    pitches["fielding_team"] = pitches["home_team"].where(
        pitches["inning_topbot"].eq("Top"), pitches["away_team"]
    )
    pitches = pitches[list(PITCH_COLUMNS)].rename(columns=PITCH_COLUMNS)
    for column in ("Fielding team", "Catcher MLB ID", "Pitch type", "Pitch result"):
        choices = sorted(pitches[column].dropna().unique())
        selected = st.sidebar.multiselect(
            column, choices, key=f"{start}_{end}_{column}"
        )
        if selected:
            pitches = pitches[pitches[column].isin(selected)]

    show_results(
        pitches,
        ["Game date", "Catcher MLB ID", "Fielding team"],
        f"statcast_pitches_{start}_{end}.csv",
        "pitch_columns",
    )


def query_number(name):
    value = st.query_params.get(name, "")
    return int(value) if value.isascii() and value.isdecimal() and len(value) <= 10 else None


st.sidebar.header("Filters")
requested_season = query_number("season")
season = st.sidebar.selectbox(
    "Season", SEASONS, index=SEASONS.index(requested_season) if requested_season in SEASONS else 0,
)
views = ["Leaderboards", "Pitch Search", "Catcher Profiles", "jcSL+ Leaderboard"]
requested_view = st.query_params.get("view")
default_view = {"sl-plus": "jcSL+ Leaderboard", "catcher": "Catcher Profiles"}.get(
    requested_view, "Leaderboards"
)
if pending := st.session_state.pop("pending_profile", None):
    open_catcher_profile(*pending)
view = st.radio(
    "View", views, index=views.index(default_view), horizontal=True,
    label_visibility="collapsed", key="catcher_nav",
)

try:
    if view == "Leaderboards":
        show_leaderboard(season)
    elif view == "Pitch Search":
        show_pitch_search(season)
    elif view == "Catcher Profiles":
        show_catcher_profile(season, query_number("catcher") if requested_season == season else None)
    else:
        show_sl_plus_leaderboard(season)
except pd.errors.EmptyDataError:
    st.info("Baseball Savant returned no data for this selection.")
except (requests.RequestException, ValueError, KeyError):
    st.error("Unable to load this Baseball Savant dataset. Try again or change the selection.")

st.caption("Data source: MLB Baseball Savant")
