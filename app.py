
import streamlit as st
import pandas as pd
import requests
from datetime import date
from io import StringIO
from pathlib import Path

from catcher_profiles import show_catcher_profile
from sl_plus_leaderboard import show_sl_plus_leaderboard

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

DATASETS = {
    "Receiving": "catcher-framing",
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


def show_leaderboard(season):
    category = st.sidebar.radio("Category", list(DATASETS))
    search = st.sidebar.text_input("Search player").strip()
    st.header(f"{season} MLB {category}")
    if category == "Receiving" and season == 2026:
        metric = st.sidebar.radio("Receiving metric", ["Savant framing", "SL+"])
        if metric == "SL+":
            show_sl_plus(search)
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


def show_sl_plus(search):
    st.subheader("Catcher SL+")
    st.caption(
        "2026 regular season. 100 × actual called strikes ÷ expected called strikes. "
        "100 means the catcher received as many called strikes as expected."
    )
    with st.expander("How SL+ is calculated"):
        st.write(
            "Catcher Lab estimates called-strike probabilities from tracked takes "
            "using pitch location, batter zone height, count, and batter and pitcher "
            "handedness. Games are scored using a model fitted on other games, "
            "without catcher identity. The scoring uses the initial umpire call "
            "before any ABS overturn."
        )
    minimum = st.sidebar.number_input(
        "Minimum eligible takes", min_value=0, value=200, step=50
    )
    df = pd.read_csv(Path(__file__).with_name("sl_plus_2026.csv"))
    df = df.loc[df["eligible_takes"].ge(minimum)]
    if search:
        df = df.loc[df["player"].str.contains(search, case=False, regex=False, na=False)]
    df = df.sort_values("SL+", ascending=False).rename(
        columns={
            "player": "Catcher",
            "catcher_id": "MLB player ID",
            "called_strikes": "Called strikes",
            "expected_strikes": "Expected strikes",
            "eligible_takes": "Eligible takes",
        }
    )
    df = df.round({"SL+": 2, "Expected strikes": 2})
    show_results(df, ["Catcher"], "catcher_sl_plus_2026.csv", "sl_plus_columns")


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

    with st.sidebar.form(f"pitch_dates_{season}"):
        dates = st.date_input(
            "Game dates",
            value=(date(season, 6, 1), date(season, 6, 1)),
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


st.sidebar.header("Filters")
season = st.sidebar.selectbox("Season", list(range(2026, 2017, -1)), index=0)
views = ["Leaderboards", "Pitch Search", "Catcher Profiles", "SL+ Leaderboard"]
default_view = "SL+ Leaderboard" if st.query_params.get("view") == "sl-plus" else "Leaderboards"
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
        show_catcher_profile(season)
    else:
        show_sl_plus_leaderboard(season)
except pd.errors.EmptyDataError:
    st.info("Baseball Savant returned no data for this selection.")
except (requests.RequestException, ValueError, KeyError):
    st.error("Unable to load this Baseball Savant dataset. Try again or change the selection.")

st.caption("Data source: MLB Baseball Savant")
