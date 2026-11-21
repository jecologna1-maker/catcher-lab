from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def heat_map(pitches, overlay=None):
    # Smooth counts and strike totals separately before calculating rates.
    x_edges = np.linspace(-2.5, 2.5, 101)
    z_edges = np.linspace(-1.6, 3.25, 98)
    x = (x_edges[:-1] + x_edges[1:]) / 2
    z = (z_edges[:-1] + z_edges[1:]) / 2
    smooth_x = np.exp(-0.5 * ((x[:, None] - x) / 0.17) ** 2)
    smooth_z = np.exp(-0.5 * ((z[:, None] - z) / 0.14) ** 2)

    def density(weights=None):
        grid, _, _ = np.histogram2d(
            pitches["plate_x"], pitches["zone_height"],
            bins=(x_edges, z_edges), weights=weights,
        )
        return smooth_x @ grid @ smooth_z.T

    support = density()
    denominator = np.where(support > 0, support, 1)
    actual = density(pitches["called"].astype(float)) / denominator
    expected = density(pitches["expected"]) / denominator
    difference = np.where(support >= 3, actual - expected, np.nan)
    figure = go.Figure(go.Heatmap(
        x=x, y=z, z=difference.T, zsmooth="best", hoverongaps=False,
        customdata=np.stack([actual.T, expected.T, support.T], axis=-1),
        colorscale="RdBu", zmin=-0.25, zmax=0.25,
        colorbar={"title": "Actual − expected", "tickformat": "+.0%"},
        hovertemplate=(
            "Smoothed called strike rate %{customdata[0]:.1%}<br>"
            "Smoothed expected rate %{customdata[1]:.1%}<br>"
            "Actual minus expected %{z:+.1%}<br>"
            "Weighted nearby takes %{customdata[2]:.1f}<extra></extra>"
        ),
    ))
    if overlay is not None:
        for call, color in (("Ball", "#bb4b37"), ("Called strike", "#17629c")):
            dots = overlay.loc[overlay["Initial call"].eq(call)]
            figure.add_scattergl(
                x=dots["plate_x"], y=dots["zone_height"], mode="markers", name=call,
                customdata=dots["pitch_id"].tolist(),
                marker={"size": 6, "color": color, "opacity": 0.72, "line": {"color": "white", "width": 0.5}},
                text=[
                    f"{date} · {pitch} · {call}<br>Count {count} · Probability {probability:.1%}"
                    for date, pitch, count, probability in dots[[
                        "Date", "Pitch", "Count", "Called-strike probability",
                    ]].itertuples(index=False, name=None)
                ],
                hovertemplate="%{text}<extra></extra>",
            )
    figure.add_shape(
        type="rect", x0=-17 / 24, x1=17 / 24, y0=0, y1=1,
        line={"color": "#68778b", "width": 1.5, "dash": "dash"},
    )
    figure.update_layout(
        height=485, clickmode="event+select", hovermode="closest",
        margin={"l": 40, "r": 10, "t": 24, "b": 45},
        legend={"orientation": "h", "y": 1.04},
        xaxis={"title": "Horizontal location (ft)", "range": [x_edges[0], x_edges[-1]]},
        yaxis={"title": "Batter strike-zone height", "range": [z_edges[0], z_edges[-1]]},
    )
    return figure


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
        "Blue shows more called strikes than expected. Red shows fewer. "
        "The surface averages nearby pitches and uses the full season. "
        "Height 0 is the bottom of the batter’s zone and 1 is the top."
    )
    if not st.toggle("Overlay individual pitches", key="show_pitch_overlay"):
        st.plotly_chart(heat_map(profile), use_container_width=True)
        return

    overlay = profile.loc[profile["expected"].between(0.05, 0.95, inclusive="neither")].copy()
    if overlay.empty:
        st.caption("No eligible pitches have a called-strike probability between 5% and 95%.")
        st.plotly_chart(heat_map(profile), use_container_width=True)
        return

    overlay = overlay.reset_index(drop=True)
    overlay["pitch_id"] = overlay.index
    overlay["Initial call"] = overlay["called"].map({True: "Called strike", False: "Ball"})
    overlay["Count"] = overlay["balls"].astype(str) + "–" + overlay["strikes"].astype(str)
    overlay = overlay.rename(columns={
        "pitch_name": "Pitch", "inning": "Inning", "at_bat_number": "Plate appearance",
        "pitch_number": "Pitch number", "expected": "Called-strike probability",
        "game_date": "Date",
    })
    st.caption(
        f"{len(overlay):,} pitches from the full season with estimated probabilities "
        "above 5% and below 95%. Select a dot or a table row to find its Savant clip."
    )
    chart_key = f"pitch_chart_{catcher}"
    table_key = f"pitch_table_{catcher}"
    pitch_key = f"selected_pitch_{catcher}"

    def select_dot():
        points = st.session_state[chart_key]["selection"]["points"]
        pitch_ids = [point["customdata"] for point in points if "customdata" in point]
        st.session_state[pitch_key] = pitch_ids[-1] if pitch_ids else None

    def select_row():
        rows = st.session_state[table_key]["selection"]["rows"]
        st.session_state[pitch_key] = rows[0] if rows else None

    chart_columns = [
        "pitch_id", "plate_x", "zone_height", "Date", "Inning", "Plate appearance",
        "Pitch number", "Count", "Pitch", "Initial call", "Called-strike probability",
    ]
    st.plotly_chart(
        heat_map(profile, overlay[chart_columns]), use_container_width=True,
        key=chart_key, on_select=select_dot, selection_mode="points",
    )
    details = st.container()
    st.dataframe(
        overlay[chart_columns[3:]], width="stretch", hide_index=True,
        key=table_key, on_select=select_row, selection_mode="single-row",
        column_config={
            "Called-strike probability": st.column_config.NumberColumn(format="percent"),
        },
    )
    selected = st.session_state.get(pitch_key)
    if selected is not None and selected in overlay.index:
        pitch = overlay.loc[selected]
        with details:
            st.caption(
                f"{pitch['Date']} · {pitch['away_team']} at {pitch['home_team']} · "
                f"Inning {pitch['Inning']} · Pitch {pitch['Pitch number']} · "
                f"{pitch['Pitch']} · {pitch['Initial call']} · "
                f"{pitch['Called-strike probability']:.1%} estimated called-strike probability"
            )
            try:
                links = game_video_links(int(pitch["game_pk"]))
                url = links.get((pitch["Plate appearance"], pitch["Pitch number"], catcher, pitch["batter"]))
            except (requests.RequestException, ValueError, KeyError, AttributeError):
                st.info("Savant video links are temporarily unavailable for this pitch.")
            else:
                if url:
                    st.link_button("Watch on Baseball Savant", url)
                else:
                    st.info("Savant has no matching video link for this pitch.")
