from datetime import date
from pathlib import Path

import pandas as pd

import plotly.graph_objects as go
import requests
import streamlit as st

from catcher_profiles import LOCATIONS, PITCH_FAMILIES, profile_data
from sl_plus_leaderboard import (
    BASE_STATES, HANDEDNESS, KEY_FIELDS, filter_pitches, load_video_source,
    savant_clip_url, savant_game_pitches,
)


def reset_framing_review():
    st.session_state["framing_review_version"] = st.session_state.get("framing_review_version", 0) + 1


@st.cache_data(max_entries=20, show_spinner=False)
def review_pitches(catcher, season=2026):
    _, pitches = profile_data(season)
    return pitches.loc[
        pitches["fielder_2"].eq(catcher)
        & pitches["expected"].between(0.05, 0.95, inclusive="both")
    ].sort_values(["game_date", *KEY_FIELDS], ascending=[False, True, True, True, True, True])


@st.cache_data(show_spinner=False)
def load_blocking_pitches(path, version):
    return pd.read_parquet(path)


def blocking_review_pitches(catcher, season, *, actual_only=False):
    path = Path(__file__).with_name(f"blocking_pitches_{season}.parquet")
    file = path.stat()
    pitches = load_blocking_pitches(path, (file.st_mtime_ns, file.st_size))
    matching = pitches["fielder_2"].eq(catcher)
    if actual_only:
        matching &= pitches["misplay"]
    return pitches.loc[matching].sort_values(
        ["expected_misplays", "game_date", *KEY_FIELDS],
        ascending=[True, False, *([True] * len(KEY_FIELDS))],
    )


def pitch_location_diagram(pitch, *, favorable=None):
    """Plot plate coordinates and batter-relative zone height from the catcher’s view."""
    half_plate = 17 / 24
    x, height = pitch.plate_x, pitch.zone_height
    if favorable is None:
        favorable = pitch.called
    figure = go.Figure()
    figure.add_shape(
        type="rect", x0=-half_plate, x1=half_plate, y0=0, y1=1,
        fillcolor="#f2edf5", line={"color": "#503363", "width": 2},
    )
    for y in (1 / 3, 2 / 3):
        figure.add_shape(type="line", x0=-half_plate, x1=half_plate, y0=y, y1=y,
                         line={"color": "#cbbbd5", "width": 1})
    for line_x in (-half_plate / 3, half_plate / 3):
        figure.add_shape(type="line", x0=line_x, x1=line_x, y0=0, y1=1,
                         line={"color": "#cbbbd5", "width": 1})
    figure.add_shape(
        type="path", path="M -0.48,-0.40 L 0.48,-0.40 L 0.48,-0.47 L 0,-0.61 L -0.48,-0.47 Z",
        fillcolor="#eee8f1", line={"color": "#503363", "width": 1},
    )
    figure.add_scatter(
        x=[x], y=[height], mode="markers", hoverinfo="skip",
        marker={
            "size": 13, "color": "#d8b64c" if favorable else "#6b35a1",
            "line": {"color": "#342044", "width": 1.5},
        },
    )
    x_bound = max(1.25, abs(x) + 0.18)
    figure.update_layout(
        height=215, margin={"l": 6, "r": 6, "t": 6, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False, "range": [-x_bound, x_bound], "fixedrange": True},
        yaxis={"visible": False, "range": [min(-0.67, height - 0.12), max(1.20, height + 0.12)],
               "fixedrange": True},
        dragmode=False, showlegend=False,
    )
    return figure


@st.dialog("Framing Pitch Review", width="large", on_dismiss=reset_framing_review)
def show_framing_review(catcher, player, metric, season=2026):
    st.subheader(player)
    st.caption(f"{metric} · 5%–95% called-strike probability · Newest games first")
    show_pitch_review(review_pitches(catcher, season), catcher, metric, season, blocking=False)


@st.dialog("Blocking Pitch Review", width="large", on_dismiss=reset_framing_review)
def show_blocking_review(catcher, player, metric, season):
    st.subheader(player)
    actual_only = metric == "Actual WP + PB"
    pitches = blocking_review_pitches(catcher, season, actual_only=actual_only)
    st.caption(f"{metric} · {len(pitches):,} pitches · Lowest WP/PB probability first")
    if actual_only and pitches.empty:
        st.info("No wild pitches or passed balls were scored on this catcher’s qualifying pitches.")
        return
    show_pitch_review(pitches, catcher, metric, season, blocking=True, actual_only=actual_only)


def show_pitch_review(pitches, catcher, metric, season, *, blocking, actual_only=False):
    if pitches.empty:
        st.info("No pitches are available for review.")
        return

    outcomes = (
        {"No Scored Misplay": False, "Wild Pitch or Passed Ball": True}
        if blocking else {"Called Ball": False, "Called Strike": True}
    )
    outcome_column = "misplay" if blocking else "called"
    probability_column = "expected_misplays" if blocking else "expected"
    probability_label = "WP/PB Probability" if blocking else "Called Strike Probability"
    namespace = (
        f"review_{blocking}_{st.session_state.get('framing_review_version', 0)}_"
        f"{season}_{catcher}_{metric}"
    )

    def reset_pitch():
        st.session_state.pop(f"{namespace}_pitch", None)

    with st.expander("Pitch Filters"):
        left, right = st.columns(2)
        with left:
            dates = st.date_input(
                "Date Range",
                value=(date.fromisoformat(pitches["game_date"].min()),
                       date.fromisoformat(pitches["game_date"].max())),
                key=f"{namespace}_dates", on_change=reset_pitch,
            )
            calls = []
            if not actual_only:
                calls = st.pills(
                    "Outcome" if blocking else "Call", list(outcomes), selection_mode="multi",
                    key=f"{namespace}_calls", on_change=reset_pitch,
                )
            families = st.pills(
                "Pitch Group", list(PITCH_FAMILIES)[1:], selection_mode="multi",
                key=f"{namespace}_families", on_change=reset_pitch,
            )
        with right:
            locations = st.pills(
                "Pitch Location", LOCATIONS[1:], selection_mode="multi",
                key=f"{namespace}_locations", on_change=reset_pitch,
            )
            pitchers = st.pills(
                "Pitcher Handedness", list(HANDEDNESS), selection_mode="multi",
                key=f"{namespace}_pitchers", on_change=reset_pitch,
            )
            hitters = st.pills(
                "Hitter Handedness", list(HANDEDNESS), selection_mode="multi",
                key=f"{namespace}_hitters", on_change=reset_pitch,
            )
            bases = st.pills(
                "Base State", list(BASE_STATES), selection_mode="multi",
                key=f"{namespace}_bases", on_change=reset_pitch,
            )
    if len(dates) == 2:
        pitches = pitches.loc[pitches["game_date"].between(*(day.isoformat() for day in dates))]
    if calls:
        pitches = pitches.loc[pitches[outcome_column].isin([outcomes[call] for call in calls])]
    pitches = filter_pitches(pitches, families, locations, pitchers, hitters, bases)
    if pitches.empty:
        st.info("No pitches match these filters.")
        return

    outcome_labels = {value: label for label, value in outcomes.items()}

    def pitch_label(position):
        pitch = pitches.iloc[position]
        call = outcome_labels[bool(pitch[outcome_column])]
        return (f"{position + 1:,}. {pitch.game_date} · {pitch.pitch_name} · {call} · "
                f"{pitch[probability_column]:.1%}")

    pitch_widget = f"{namespace}_pitch"

    def move_pitch(offset):
        st.session_state[pitch_widget] += offset

    selected = st.selectbox(
        f"Pitch ({len(pitches):,} available)", range(len(pitches)), format_func=pitch_label,
        key=pitch_widget,
    )
    previous, following = st.columns(2)
    previous.button("Previous Pitch", on_click=move_pitch, args=(-1,), disabled=selected == 0,
                    key=f"{namespace}_previous")
    following.button("Next Pitch", on_click=move_pitch, args=(1,), disabled=selected == len(pitches) - 1,
                     key=f"{namespace}_next")
    pitch = pitches.iloc[selected]
    game, *key = (int(pitch[field]) for field in KEY_FIELDS)
    try:
        play_id, pitcher, hitter = savant_game_pitches(game).get(tuple(key), (None, None, None))
    except (requests.RequestException, KeyError, ValueError, TypeError):
        play_id, pitcher, hitter = None, None, None
    video, details = st.columns([2, 1])
    with video:
        source = None
        if play_id:
            try:
                source = load_video_source(play_id)
            except (requests.RequestException, ValueError):
                pass
        if source:
            st.video(source, format="video/mp4")
        else:
            st.info("An embedded Savant video is unavailable for this pitch.")
        if play_id:
            st.link_button("Open on Baseball Savant", savant_clip_url(play_id))
    with details:
        outcome = bool(pitch[outcome_column])
        call = outcome_labels[outcome]
        st.markdown(f"**{call} · {pitch.pitch_name}**")
        st.text(f"Date  {pitch.game_date}")
        st.text(f"Pitcher  {pitcher or 'Unavailable'}")
        st.text(f"Hitter  {hitter or f'MLB ID {pitch.batter}'}")
        st.text(f"Count  {pitch.balls}-{pitch.strikes}")
        st.text(f"{probability_label}  {pitch[probability_column]:.1%}")
        if pd.notna(pitch.zone_height):
            st.caption("Pitch Location · Catcher’s View")
            st.plotly_chart(
                pitch_location_diagram(pitch, favorable=outcome != blocking),
                config={"displayModeBar": False, "staticPlot": True},
                key=f"{namespace}_location_chart",
            )
