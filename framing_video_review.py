from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

import plotly.graph_objects as go
import requests
import streamlit as st

from catcher_profiles import HALF_PLATE, LOCATIONS, PITCH_FAMILIES, home_plate_path, profile_data
from receiving_bands import RECEIVING_BANDS, receiving_band_mask
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


# These limits keep the strike zone the same size on every clip.
PITCH_X_RANGE = (-2.5, 2.5)
PITCH_HEIGHT_RANGE = (-1.65, 1.65)


def pitch_location_diagram(pitch, *, favorable=None):
    """Plot a pitch location from the catcher’s view, with a fixed viewport."""
    half_plate = HALF_PLATE
    x, height = pitch.plate_x, pitch.zone_height
    clipped = not (PITCH_X_RANGE[0] < x < PITCH_X_RANGE[1]
                   and PITCH_HEIGHT_RANGE[0] < height < PITCH_HEIGHT_RANGE[1])
    if clipped:
        x = min(max(x, PITCH_X_RANGE[0] + 0.12), PITCH_X_RANGE[1] - 0.12)
        height = min(max(height, PITCH_HEIGHT_RANGE[0] + 0.12), PITCH_HEIGHT_RANGE[1] - 0.12)
    if favorable is None:
        favorable = pitch.called
    figure = go.Figure()
    figure.add_shape(
        type="rect", x0=-half_plate, x1=half_plate, y0=0, y1=1,
        layer="below", fillcolor="#f2edf5", line={"color": "#503363", "width": 2},
    )
    for y in (1 / 3, 2 / 3):
        figure.add_shape(type="line", x0=-half_plate, x1=half_plate, y0=y, y1=y,
                         layer="below", line={"color": "#cbbbd5", "width": 1})
    for line_x in (-half_plate / 3, half_plate / 3):
        figure.add_shape(type="line", x0=line_x, x1=line_x, y0=0, y1=1,
                         layer="below", line={"color": "#cbbbd5", "width": 1})
    figure.add_shape(
        type="path", path=home_plate_path(-0.34, 0.26),
        layer="below", fillcolor="#eee8f1", line={"color": "#503363", "width": 1},
    )
    figure.add_scatter(
        x=[x], y=[height], mode="markers", hoverinfo="skip",
        marker={
            "size": 12, "symbol": "circle-open" if clipped else "circle", "color": "#d8b64c" if favorable else "#6b35a1",
            "line": {"color": "#342044", "width": 1.5},
        },
    )
    figure.update_layout(
        height=245, margin={"l": 6, "r": 6, "t": 6, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "visible": False, "range": PITCH_X_RANGE, "fixedrange": True,
            "scaleanchor": "y", "scaleratio": 0.59, "constrain": "domain",
        },
        yaxis={
            "visible": False, "range": PITCH_HEIGHT_RANGE, "fixedrange": True,
            "constrain": "domain",
        },
        dragmode=False, showlegend=False,
    )
    return figure


@st.dialog("Framing Pitch Review", width="large", on_dismiss=reset_framing_review)
def show_framing_review(catcher, player, metric, season=2026):
    st.subheader(player)
    band = "5%–95%"
    if metric in RECEIVING_BANDS:
        _, season_pitches = profile_data(season)
        pitches = season_pitches.loc[
            season_pitches["fielder_2"].eq(catcher)
            & receiving_band_mask(season_pitches["expected"], metric)
        ].sort_values(["game_date", *KEY_FIELDS], ascending=[False, True, True, True, True, True])
        band = RECEIVING_BANDS[metric][3]
    else:
        pitches = review_pitches(catcher, season)
    st.caption(f"{metric} · {band} called-strike probability · Newest games first")
    show_pitch_review(pitches, catcher, metric, season, blocking=False)


@st.dialog("Blocking Pitch Review", width="large", on_dismiss=reset_framing_review)
def show_blocking_review(catcher, player, metric, season):
    st.subheader(player)
    actual_only = metric == "Missed Blocks"
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

    stage = st.container(key="pitch_review_workspace", gap="small")
    with st.container(border=True, key="pitch_review_queue", gap="small"):
        st.markdown("**Pitch Queue**")
        selected = st.selectbox(
            f"Pitch ({len(pitches):,} available)", range(len(pitches)), format_func=pitch_label,
            key=pitch_widget, label_visibility="collapsed",
        )
    pitch = pitches.iloc[selected]
    game, *key = (int(pitch[field]) for field in KEY_FIELDS)
    try:
        play_id, pitcher, hitter = savant_game_pitches(game).get(tuple(key), (None, None, None))
    except (requests.RequestException, KeyError, ValueError, TypeError):
        play_id, pitcher, hitter = None, None, None
    with stage:
        video, details = st.columns([2.35, 1], gap="small")
    with video.container(border=True, key="pitch_review_video", gap="small"):
        st.markdown("**Pitch Video**")
        previous, position, following = st.columns([1, 1, 1], gap="small", vertical_alignment="center")
        previous.button(
            "← Previous Pitch", on_click=move_pitch, args=(-1,), disabled=selected == 0,
            key=f"{namespace}_previous", width="stretch",
        )
        position.html(
            f'<div class="pitch-review-position">{selected + 1:,} / {len(pitches):,}</div>'
        )
        following.button(
            "Next Pitch →", on_click=move_pitch, args=(1,), disabled=selected == len(pitches) - 1,
            key=f"{namespace}_next", width="stretch",
        )
        source = None
        if play_id:
            try:
                source = load_video_source(play_id)
            except (requests.RequestException, ValueError):
                pass
        if source:
            st.video(source, format="video/mp4", autoplay=True, muted=True)
        else:
            st.info("An embedded Savant video is unavailable for this pitch.")
        if play_id:
            st.link_button("Open on Baseball Savant", savant_clip_url(play_id))
    with details.container(border=True, key="pitch_review_location", gap="small"):
        st.markdown("**Pitch Location**")
        outcome = bool(pitch[outcome_column])
        call = outcome_labels[outcome]
        pitcher_name = escape(pitcher or "Unavailable")
        hitter_name = escape(hitter or f"MLB ID {pitch.batter}")
        st.html(
            '<div class="pitch-review-facts" style="font-size:.82rem;line-height:1.4">'
            f'<strong>{escape(call)} · {escape(str(pitch.pitch_name))}</strong><br>'
            f'{pitch.game_date} · Count {pitch.balls}-{pitch.strikes}<br>'
            f'Pitcher {pitcher_name} · Hitter {hitter_name}<br>'
            f'<strong>{pitch[probability_column]:.1%}</strong> {probability_label}'
            '</div>'
        )
        if pd.notna(pitch.zone_height):
            diagram = pitch_location_diagram(pitch, favorable=outcome != blocking)
            location_label = "Pitch Location · Catcher’s View"
            if diagram.data[0].marker.symbol == "circle-open":
                location_label += " · Beyond chart edge"
            st.caption(location_label)
            st.plotly_chart(
                diagram, config={"displayModeBar": False, "staticPlot": True},
                key=f"{namespace}_location_chart",
            )
