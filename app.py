
import json

import streamlit as st
import pandas as pd
import requests
from datetime import date
from io import StringIO
from pathlib import Path
from streamlit.components.v1 import declare_component

from catcher_profiles import catcher_profile_url, has_receiving_data, show_catcher_profile
from sl_plus_leaderboard import show_sl_plus_leaderboard
from framing_video_review import show_framing_review

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
    "Blocking": "catcher-blocking",
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


def show_dirt_blocking(season, search):
    if season != 2026:
        st.info("Dirt Blocking is available for the 2026 regular season.")
        return

    st.caption(
        "Tracked pitches projected to touch dirt by the front of home plate, with a runner "
        "on base or a chance for the batter to reach on an uncaught third strike."
    )
    with st.expander("How Dirt Blocking is calculated"):
        st.write(
            "The pitch trajectory estimates where the ball would touch the ground. "
            "The model estimates the chance of a scored wild pitch or passed ball from "
            "location, depth below ground at the plate, speed, movement, handedness, "
            "and the runner and out situation. Catcher identity is excluded, and "
            "each game is scored using a model trained on other games."
        )
        st.write(
            "Blocks Above Average equals expected minus actual wild pitches and passed balls. "
            "A score of +5 means five fewer scored misplays than expected given the pitches received. "
            "A pitch without a scored misplay can still have bounced away without anyone advancing."
        )
    minimum = st.sidebar.number_input("Minimum Dirt Opportunities", min_value=0, value=50, step=10)
    frame = pd.read_csv(Path(__file__).with_name(f"dirt_blocking_{season}.csv"))
    frame = frame.loc[frame["opportunities"].ge(minimum)]
    if search:
        frame = frame.loc[frame["player"].str.contains(search, case=False, regex=False, na=False)]
    columns = {
        "player": "Player",
        "opportunities": "Dirt Opportunities",
        "blocks_above_average": "Blocks Above Average",
        "actual_misplays": "Actual WP + PB",
        "expected_misplays": "Expected WP + PB",
    }
    frame = frame.sort_values("blocks_above_average", ascending=False)[list(columns)].rename(columns=columns)
    show_results(frame.round(2), ["Player"], f"dirt_blocking_{season}.csv", "dirt_blocking_columns")


def show_leaderboard(season):
    category = st.sidebar.radio("Category", ["Receiving", "Dirt Blocking", *DATASETS])
    search = st.sidebar.text_input("Search player").strip()
    st.header(f"{season} MLB {category}")
    if category == "Dirt Blocking":
        show_dirt_blocking(season, search)
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
    if search:
        df = df.loc[df["player"].str.contains(search, case=False, regex=False, na=False)]
    df = df.sort_values("SL+", ascending=False)
    columns = {
        "player": "Player",
        "eligible_takes": "Total Pitches",
        "SL+": "jcSL+",
        "SLAA": "jcSLAA",
        "Framing Runs Above Average": "jcFRAA",
    }
    results = df[list(columns)].rename(columns=columns)
    st.caption(
        "Select a player for their profile or a number to review pitches "
        "with 5%–95% called-strike probability."
    )
    rows = [
        {
            "id": int(catcher),
            "profile": catcher_profile_url(season, catcher),
            "values": list(values),
        }
        for catcher, values in zip(df["catcher_id"], results.itertuples(index=False, name=None))
    ]
    event = receiving_table(
        rows=rows, headers=list(results.columns), default=None,
        key=f"receiving_{season}_{minimum}_{search}",
    )
    if event and event != st.session_state.get("last_receiving_click"):
        st.session_state["last_receiving_click"] = event
        matching = df.loc[df["catcher_id"].eq(event["catcher"])]
        if not matching.empty:
            row = matching.iloc[0]
            column = event["column"]
            if column == "Player":
                st.session_state["pending_profile"] = (int(row["catcher_id"]), season)
                st.rerun()
            elif column in results.columns:
                show_framing_review(int(row["catcher_id"]), row["player"], column, season)
    st.download_button(
        "Download CSV", results.round(2).to_csv(index=False),
        file_name=f"catcher_framing_{season}.csv", mime="text/csv",
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
