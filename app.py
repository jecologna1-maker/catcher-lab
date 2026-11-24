
import json

import streamlit as st
import pandas as pd
import requests
from datetime import date
from io import StringIO
from pathlib import Path
from unicodedata import combining, normalize
from streamlit.components.v1 import declare_component

from blocking_stats import (
    BLOCKING_BUCKETS, DIFFICULT_CUTOFF, ROUTINE_CUTOFF, blocking_pitches, blocking_rate_totals,
)
from catcher_profiles import catcher_profile_url, has_receiving_data, show_catcher_profile
from receiving_bands import receiving_band_totals
from sl_plus_leaderboard import show_sl_plus_leaderboard
from framing_video_review import show_blocking_review, show_framing_review

st.set_page_config(
    page_title="Catcher Lab",
    page_icon="⚾",
    layout="wide"
)

st.html(Path(__file__).with_name("catcher_lab.css"))

receiving_table = declare_component(
    "receiving_table", path=str(Path(__file__).with_name("receiving_table")),
)
SEASONS = list(range(2026, 2019, -1))


def open_catcher_profile(catcher, season):
    st.session_state["catcher_nav"] = "Catcher Profiles"
    st.session_state["profile_catcher_id"] = catcher
    if "profile_year_range" not in st.session_state:
        st.session_state["profile_year_range"] = (season, season)


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


def show_results(df, identity_columns, filename, widget_key, filters):
    statistics = [column for column in df.columns if column not in identity_columns]
    visible = filters.multiselect(
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


def show_blocking(season, qualification):
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
            "passed balls. xMiss adds up their predicted probabilities. "
            "BLAA is xMiss minus Missed Blocks, and Miss Surplus is Missed Blocks minus xMiss. "
            "Block Rate is the percentage of opportunities with no advancement recorded as a "
            "wild pitch or passed ball."
        )
        st.write(
            f"Routine+ uses risk below {ROUTINE_CUTOFF:.2%}, Extended+ uses "
            f"{ROUTINE_CUTOFF:.2%} to below {DIFFICULT_CUTOFF:.2%}, and Difficult+ uses "
            f"{DIFFICULT_CUTOFF:.2%} and up. Each bucket contains about a third of the "
            "league’s expected misses. Each + score uses the Block+ formula within that bucket."
        )
    minimum = qualification.number_input("Minimum Block Opportunities", min_value=0, value=50, step=10)
    frame = pd.read_csv(Path(__file__).with_name(f"dirt_blocking_{season}.csv"))
    frame = frame.loc[frame["opportunities"].ge(minimum)].copy()
    frame["blocks"] = frame["opportunities"] - frame["actual_misplays"]
    frame["expected_blocks"] = frame["opportunities"] - frame["expected_misplays"]
    frame["block_plus"] = 100 * frame["blocks"] / frame["expected_blocks"]
    frame["miss_surplus"] = frame["actual_misplays"] - frame["expected_misplays"]
    frame["blocks_above_average"] = -frame["miss_surplus"]
    rates = blocking_rate_totals(blocking_pitches(season))
    for metric, totals in rates.items():
        totals = totals.reindex(frame["catcher_id"])
        frame[metric] = (100 * totals["actual"] / totals["denominator"]).to_numpy()
    qualified = frame
    columns = {
        "player": "Player",
        "opportunities": "Block Opportunities",
        "block_plus": "Block+",
        "blocks_above_average": "BLAA",
        "Block Rate": "Block Rate",
        "actual_misplays": "Missed Blocks",
        "expected_misplays": "xMiss",
        "miss_surplus": "Miss Surplus",
        **{metric: metric for metric in BLOCKING_BUCKETS},
    }
    frame = frame.sort_values("block_plus", ascending=False)
    st.caption(
        "Select a player for their profile or a number for pitch video and location, lowest WP/PB "
        "probability first. Missed Blocks opens only scored wild pitches and passed balls."
    )
    show_catcher_table(
        frame, qualified, columns, season, minimum, category="Blocking",
        integer_columns=["Block Opportunities", "Missed Blocks"], review=show_blocking_review, bands=rates,
    )


def show_leaderboard(season, category_control, qualification, extra_filters):
    category = category_control.selectbox("Category", ["Receiving", "Blocking", *DATASETS])
    st.header(f"{season} MLB {category}")
    if category == "Blocking":
        show_blocking(season, qualification)
        return
    if category == "Receiving":
        if has_receiving_data(season):
            show_sl_plus(season, qualification)
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
        selected = extra_filters.multiselect(
            "Teams", teams, key=f"teams_{category}_{season}"
        )
        if selected:
            df = df[df[team_column].isin(selected)]

    show_results(df, identity_columns, f"{category}_{season}.csv", f"columns_{category}", extra_filters)




@st.cache_data(max_entries=7)
def decisive_pitch_counts(season, data_version):
    pitches = pd.read_parquet(
        Path(__file__).with_name(f"catcher_pitches_{season}.parquet"),
        columns=["fielder_2", "expected"],
    )
    return pitches.loc[pitches["expected"].between(0.05, 0.95)].groupby("fielder_2").size()


def show_sl_plus(season, qualification):
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
    minimum = qualification.number_input(
        "Minimum Decisive Pitches Received", min_value=0, value=500, step=50,
        help="Decisive pitches are takes with a 5% through 95% predicted called-strike probability.",
    )
    df = pd.read_csv(Path(__file__).with_name(f"sl_plus_{season}.csv"))
    data_file = Path(__file__).with_name(f"catcher_pitches_{season}.parquet")
    counts = decisive_pitch_counts(season, data_file.stat().st_mtime_ns)
    df["decisive_pitches"] = df["catcher_id"].map(counts).fillna(0).astype(int)
    df = df.loc[df["decisive_pitches"].ge(minimum)]
    bands = receiving_band_totals(season, data_file.stat().st_mtime_ns)
    for metric, totals in bands.items():
        totals = totals.reindex(df["catcher_id"])
        df[metric] = (100 * totals["actual"] / totals["denominator"]).to_numpy()
    qualified = df
    df = df.sort_values("SL+", ascending=False)
    columns = {
        "player": "Player",
        "decisive_pitches": "Decisive Pitches",
        "SL+": "jcSL+",
        "SLAA": "jcSLAA",
        "Framing Runs Above Average": "jcFRMAA",
        **{metric: metric for metric in ("50/50%", "Gimme+", "Win+")},
    }
    st.caption(
        "Select a player for their profile or a number for pitch video and location "
        "50/50% is the actual strike rate from 25% to under 75% predicted probability. "
        "Win+ (24.9% and below) and Gimme+ (75.1% and above) compare actual to expected strikes."
    )
    show_catcher_table(
        df, qualified, columns, season, minimum, category="Receiving",
        integer_columns=["Decisive Pitches"], review=show_framing_review, bands=bands,
    )


def show_catcher_table(df, qualified, columns, season, minimum, *, category, integer_columns, review, bands=None):
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
        key=f"{category}_{season}_{minimum}",
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


def show_pitch_search(season, extra_filters):
    st.header(f"{season} Statcast Pitch Search")
    st.caption("Regular-season pitches. Choose a date range of up to 7 days.")

    default_pitch_day = date(season, 7, 23) if season == 2020 else date(season, 6, 1)
    with extra_filters.form(f"pitch_dates_{season}"):
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
        selected = extra_filters.multiselect(
            column, choices, key=f"{start}_{end}_{column}"
        )
        if selected:
            pitches = pitches[pitches[column].isin(selected)]

    show_results(
        pitches,
        ["Game date", "Catcher MLB ID", "Fielding team"],
        f"statcast_pitches_{start}_{end}.csv",
        "pitch_columns", extra_filters,
    )


def show_glossary():
    st.header("Catcher Lab Glossary")

    st.subheader("Receiving")
    st.markdown(
        """
        - **Decisive Pitches** — Tracked takes with a predicted called-strike probability from 5% through 95%, inclusive. The Receiving leaderboard defaults to a minimum of 500.
        - **jcSL+** — 100 × actual called strikes ÷ expected called strikes across eligible takes. A score of 100 means the catcher received exactly as many strikes as expected.
        - **jcSLAA** — Strikes Looking Above Average. Actual called strikes minus expected called strikes, summed across eligible takes.
        - **jcFRMAA** — Framing Runs Above Average. Each pitch’s actual strike result (1 for a called strike, 0 for a ball) minus its predicted strike probability, weighted by the run value of a strike versus a ball for that pitch’s runners, outs, and count.
        - **50/50%** — The percentage of takes actually called strikes among pitches with a predicted called-strike probability from 25% to under 75%.
        - **Gimme+** — 100 × actual ÷ expected called strikes for takes at 75.1% predicted strike probability or higher. It measures how well the catcher keeps expected strikes.
        - **Win+** — 100 × actual ÷ expected called strikes for takes at 24.9% predicted strike probability or lower. It measures how often the catcher earns a strike on pitches that tend to be called balls.
        - **Called strikes, expected strikes, and eligible takes** — Called strikes count the observed calls. Expected strikes sum the individual called-strike probabilities. Eligible takes are tracked pitches scored by the model, and include pitches outside the Decisive Pitches band.
        """
    )
    st.caption(
        "Receiving models are fit separately for each season from 2020 through 2026. "
        "The 2026 scores use the umpire’s initial call before any ABS overturn."
    )

    st.subheader("Blocking")
    st.markdown(
        f"""
        - **Block Opportunities** — Tracked pitches projected to hit dirt by the back edge of the batter’s box, with a runner on base or a two-strike swing-and-miss with the bases empty.
        - **Block+** — 100 × actual blocks ÷ expected blocks. Actual blocks are opportunities minus Missed Blocks. Expected blocks are opportunities minus xMiss. A score of 100 means the catcher blocked as many pitches as expected.
        - **BLAA** — Blocks Above Average. xMiss minus Missed Blocks. Positive means fewer misses than expected.
        - **Block Rate** — Percentage of block opportunities without a wild pitch or passed ball recorded. It measures recorded outcomes when a runner could advance.
        - **Missed Blocks** — Number of qualifying pitches with a scored wild pitch or passed ball. Each pitch counts once.
        - **xMiss** — Expected misses. The sum of the modeled wild-pitch or passed-ball probabilities across qualifying opportunities.
        - **Miss Surplus** — Missed Blocks minus xMiss. Positive means more misses than expected.
        - **Routine+** — Block+ on opportunities below {ROUTINE_CUTOFF:.2%} predicted wild-pitch or passed-ball risk.
        - **Extended+** — Block+ on opportunities from {ROUTINE_CUTOFF:.2%} to below {DIFFICULT_CUTOFF:.2%} predicted risk.
        - **Difficult+** — Block+ on opportunities at {DIFFICULT_CUTOFF:.2%} predicted risk or higher.
        """
    )
    st.caption(
        "Blocking is available for 2026. The three difficulty bands each contain about a third "
        "of the league’s expected misses. The displayed cutoffs are rounded."
    )


def query_number(name):
    value = st.query_params.get(name, "")
    return int(value) if value.isascii() and value.isdecimal() and len(value) <= 10 else None


requested_season = query_number("season")
default_season = requested_season if requested_season in SEASONS else SEASONS[0]
views = ["Leaderboards", "Pitch Search", "Catcher Profiles", "jcSL+ Leaderboard", "Glossary"]
requested_view = st.query_params.get("view")
default_view = {"sl-plus": "jcSL+ Leaderboard", "catcher": "Catcher Profiles", "glossary": "Glossary"}.get(
    requested_view, "Leaderboards"
)
if pending := st.session_state.pop("pending_profile", None):
    open_catcher_profile(*pending)
def normalize_player_name(name):
    return "".join(char for char in normalize("NFKD", name.casefold()) if not combining(char))


def player_matches(year_range, query):
    start, end = year_range
    names = pd.concat([
        pd.read_csv(Path(__file__).with_name(f"sl_plus_{year}.csv"), usecols=["catcher_id", "player"])
        for year in range(start, end + 1)
    ]).drop_duplicates("catcher_id", keep="last")
    normalized = names["player"].map(normalize_player_name)
    query = normalize_player_name(query)
    matches = names.loc[normalized.eq(query)]
    if matches.empty:
        matches = names.loc[normalized.str.contains(query, regex=False)]
    return matches.set_index("catcher_id")["player"].sort_values()


def player_search_years():
    season = st.session_state.get("active_catcher_season", default_season)
    if st.session_state.get("catcher_nav") == "Catcher Profiles":
        return st.session_state.get("profile_year_range", (season, season))
    return season, season


def search_players():
    query = st.session_state["player_search"].strip()
    if query:
        years = player_search_years()
        matches = player_matches(years, query)
        if len(matches) == 1:
            open_catcher_profile(int(matches.index[0]), years[1])
            st.session_state["player_search"] = ""


def open_search_result(season):
    catcher = st.session_state.get("player_search_result")
    if catcher is not None:
        open_catcher_profile(int(catcher), season)
        st.session_state["player_search"] = ""


with st.container(key="catcher_top_band"):
    identity, player_search = st.columns([2, 1], gap="small", vertical_alignment="center")
    with identity:
        st.html(
            '<header class="catcher-brand" aria-label="Catcher Lab">'
            '<span class="catcher-brand__mark" aria-hidden="true">JC</span>'
            '<div class="catcher-brand__identity">'
            '<span class="catcher-brand__name">Cologna Catching Development</span>'
            '<h1>Catcher Lab</h1></div>'
            '</header>'
        )
    search = player_search.text_input(
        "Player Search", placeholder="Find a catcher’s profile", key="player_search",
        on_change=search_players,
    ).strip()
    if search:
        years = player_search_years()
        matches = player_matches(years, search)
        if matches.empty:
            player_search.caption(f"No catcher profiles match “{search}” in the selected seasons.")
        else:
            player_search.selectbox(
                "Matching catcher profiles", matches.index.tolist(), index=None,
                format_func=matches.get, placeholder="Open a catcher profile",
                key="player_search_result", on_change=open_search_result, args=(years[1],),
            )

view = st.radio(
    "View", views, index=views.index(default_view), horizontal=True,
    label_visibility="collapsed", key="catcher_nav",
)

if view == "Catcher Profiles":
    if "profile_year_range" not in st.session_state:
        initial = st.session_state.get("active_catcher_season", default_season)
        st.session_state["profile_year_range"] = (initial, initial)
    with st.container(key="catcher_top_filters"):
        profile_range = st.select_slider(
            "Season Range", options=SEASONS[::-1], key="profile_year_range",
            value=st.session_state["profile_year_range"],
        )
        profile_filters = st.container()
elif view != "Glossary":
    with st.container(key="catcher_top_filters"):
        season_control, category_control, qualification = st.columns([1, 1, 2], gap="small")
        season = season_control.selectbox(
            "Season", SEASONS,
            index=SEASONS.index(st.session_state.get("active_catcher_season", default_season)),
            key="catcher_season",
        )
        st.session_state["active_catcher_season"] = season
        extra_filters = st.container()

try:
    if view == "Leaderboards":
        show_leaderboard(season, category_control, qualification, extra_filters)
    elif view == "Pitch Search":
        show_pitch_search(season, extra_filters)
    elif view == "Catcher Profiles":
        show_catcher_profile(
            profile_range, st.session_state.get("profile_catcher_id", query_number("catcher")),
            profile_filters,
        )
    elif view == "Glossary":
        show_glossary()
    else:
        show_sl_plus_leaderboard(season)
except pd.errors.EmptyDataError:
    st.info("Baseball Savant returned no data for this selection.")
except (requests.RequestException, ValueError, KeyError):
    st.error("Unable to load this Baseball Savant dataset. Try again or change the selection.")

st.caption("Data source: MLB Baseball Savant")
