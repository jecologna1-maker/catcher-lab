from pathlib import Path
from uuid import UUID

import altair as alt
import pandas as pd
import requests
import streamlit as st


@st.cache_data
def load_profile_data():
    folder = Path(__file__).parent
    catchers = pd.read_csv(folder / "sl_plus_2026.csv").sort_values("player")
    pitches = pd.read_parquet(folder / "catcher_pitches_2026.parquet")
    return catchers, pitches


@st.cache_data(ttl=21600, max_entries=128)
def game_video_links(game_id):
    response = requests.get(
        "https://baseballsavant.mlb.com/gf", params={"game_pk": game_id}, timeout=15
    )
    response.raise_for_status()
    game = response.json()
    links = {}
    ambiguous = set()
    for side in ("home_pitchers", "away_pitchers"):
        for pitcher_pitches in game[side].values():
            for pitch in pitcher_pitches:
                try:
                    key = tuple(
                        int(pitch[field])
                        for field in ("ab_number", "pitch_number", "catcher", "batter")
                    )
                    play_id = UUID(pitch["play_id"])
                except (KeyError, ValueError, TypeError, AttributeError):
                    continue
                url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
                if key in links and links[key] != url:
                    ambiguous.add(key)
                links[key] = url
    return {key: url for key, url in links.items() if key not in ambiguous}


def zone_outline():
    zone = pd.DataFrame({"left": [-17 / 24], "right": [17 / 24], "bottom": [0], "top": [1]})
    return alt.Chart(zone).mark_rect(
        fillOpacity=0, stroke="#68778b", strokeWidth=1.5, strokeDash=[4, 3]
    ).encode(x="left:Q", x2="right:Q", y="bottom:Q", y2="top:Q")


def heat_map(pitches):
    # Align the vertical coordinates to each batter's measured strike-zone bounds.
    visible = pitches.loc[
        pitches["plate_x"].between(-2, 2, inclusive="left")
        & pitches["zone_height"].between(-0.8, 1.8, inclusive="left")
    ].copy()
    visible["x0"] = (visible["plate_x"] * 4).floordiv(1) / 4
    visible["z0"] = (visible["zone_height"] * 5).floordiv(1) / 5
    cells = visible.groupby(["x0", "z0"], as_index=False).agg(
        takes=("called", "size"), actual=("called", "mean"), expected=("expected", "mean")
    )
    cells["x1"] = cells["x0"] + 0.25
    cells["z1"] = cells["z0"] + 0.20
    cells["difference"] = cells["actual"] - cells["expected"]
    chart = alt.Chart(cells).mark_rect().encode(
        x=alt.X("x0:Q", title="Horizontal location (ft)", scale=alt.Scale(domain=[-2, 2])),
        x2="x1:Q",
        y=alt.Y("z0:Q", title="Batter strike-zone height", scale=alt.Scale(domain=[-0.8, 1.8])),
        y2="z1:Q",
        color=alt.Color(
            "difference:Q",
            title="Actual minus expected",
            scale=alt.Scale(domain=[-0.25, 0.25], scheme="redblue", clamp=True),
            legend=alt.Legend(format="+.0%"),
        ),
        tooltip=[
            alt.Tooltip("takes:Q", title="Takes"),
            alt.Tooltip("actual:Q", title="Called strike rate", format=".1%"),
            alt.Tooltip("expected:Q", title="Expected rate", format=".1%"),
            alt.Tooltip("difference:Q", title="Actual minus expected", format="+.1%"),
        ],
    )
    return (chart + zone_outline()).properties(height=330)


def pitch_chart(pitches):
    chart = alt.Chart(pitches).mark_circle(size=90, opacity=0.78, cursor="pointer").encode(
        x=alt.X("plate_x:Q", title="Horizontal location (ft)", scale=alt.Scale(zero=False)),
        y=alt.Y("zone_height:Q", title="Batter strike-zone height", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "Initial call:N",
            scale=alt.Scale(domain=["Ball", "Called strike"], range=["#c95f4a", "#2376ad"]),
        ),
        href="Video:N",
        tooltip=[
            "Pitch:N", "Inning:Q", "Count:N", "Plate appearance:Q", "Pitch number:Q",
            "Initial call:N",
            alt.Tooltip("Called-strike probability:Q", format=".1%"),
        ],
    )
    return (zone_outline() + chart).properties(height=330)


def show_catcher_profile(season):
    if season != 2026:
        st.info("Catcher profiles are available for the 2026 regular season.")
        return

    catchers, pitches = load_profile_data()
    roster = catchers.set_index("catcher_id")
    catcher = st.sidebar.selectbox(
        "Catcher", roster.index.tolist(), format_func=lambda player: roster.at[player, "player"]
    )
    player = roster.loc[catcher]
    profile = pitches.loc[pitches["fielder_2"].eq(catcher)]
    st.header(player["player"])
    st.caption("2026 regular season · Catcher profile")
    columns = st.columns(4)
    columns[0].metric("SL+", f"{player['SL+']:.2f}")
    columns[1].metric("Called strikes", f"{player['called_strikes']:,}")
    columns[2].metric("Expected strikes", f"{player['expected_strikes']:,.1f}")
    columns[3].metric("Eligible takes", f"{player['eligible_takes']:,}")

    st.subheader("Called strikes by location")
    st.caption(
        "Blue means more called strikes than expected. Red means fewer. "
        "Hover over a cell for the actual rate, expected rate, and pitch count. "
        "Height 0 is the bottom of the batter’s zone and 1 is the top."
    )
    st.altair_chart(heat_map(profile), use_container_width=True)

    st.subheader("Pitch video")
    games = (
        profile.drop_duplicates("game_pk").sort_values(["game_date", "game_pk"], ascending=False)
        .set_index("game_pk")
    )
    labels = {
        game_id: f"{r.game_date} · {r.away_team} at {r.home_team} · {game_id}"
        for game_id, r in games.iterrows()
    }
    game_id = st.selectbox("Game", games.index.tolist(), format_func=labels.get, key=f"game_{catcher}")
    game_pitches = profile.loc[profile["game_pk"].eq(game_id)].copy()
    try:
        links = game_video_links(game_id)
    except (requests.RequestException, ValueError, KeyError, AttributeError):
        links = {}
        st.info("Savant video links are temporarily unavailable for this game.")

    game_pitches["Video"] = [
        links.get((row.at_bat_number, row.pitch_number, catcher, row.batter))
        for row in game_pitches.itertuples(index=False)
    ]
    game_pitches["Initial call"] = game_pitches["called"].map({True: "Called strike", False: "Ball"})
    game_pitches["Count"] = game_pitches["balls"].astype(str) + "–" + game_pitches["strikes"].astype(str)
    game_pitches = game_pitches.rename(columns={
        "pitch_name": "Pitch", "inning": "Inning", "at_bat_number": "Plate appearance",
        "pitch_number": "Pitch number", "expected": "Called-strike probability",
    })
    st.caption("Click a pitch dot or its Watch link to open the clip on Baseball Savant.")
    st.altair_chart(pitch_chart(game_pitches), use_container_width=True)
    if game_pitches["Video"].isna().any():
        st.caption("Pitches without a Watch link could not be matched to a Savant clip.")
    st.dataframe(
        game_pitches[[
            "Inning", "Plate appearance", "Pitch number", "Count", "Pitch", "Initial call",
            "Called-strike probability", "Video",
        ]],
        width="stretch",
        hide_index=True,
        column_config={
            "Video": st.column_config.LinkColumn("Savant video", display_text="Watch"),
            "Called-strike probability": st.column_config.NumberColumn(format="percent"),
        },
    )
