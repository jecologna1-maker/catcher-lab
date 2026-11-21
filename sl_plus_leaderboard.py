from concurrent.futures import ThreadPoolExecutor
from html import escape
from uuid import UUID

import pandas as pd
import requests
import streamlit as st

from catcher_profiles import LOCATIONS, PITCH_FAMILIES, pitch_location_mask, profile_data

HANDEDNESS = {"Left": "L", "Right": "R"}
BASE_STATES = {"Bases Empty": False, "Runners On Base": True}
CLIP_COUNT = 10
KEY_FIELDS = ["game_pk", "at_bat_number", "pitch_number", "fielder_2", "batter"]


def filter_pitches(pitches, families, locations, pitchers, hitters, base_states=()):
    if families:
        codes = [code for family in families for code in PITCH_FAMILIES[family]]
        pitches = pitches.loc[pitches["pitch_type"].isin(codes)]
    if locations:
        masks = [pitch_location_mask(pitches, location) for location in locations]
        pitches = pitches.loc[pd.concat(masks, axis=1).any(axis=1)]
    if pitchers:
        pitches = pitches.loc[pitches["p_throws"].isin([HANDEDNESS[hand] for hand in pitchers])]
    if hitters:
        pitches = pitches.loc[pitches["stand"].isin([HANDEDNESS[hand] for hand in hitters])]
    if base_states:
        pitches = pitches.loc[pitches["runners_on"].isin([BASE_STATES[state] for state in base_states])]
    return pitches


def rank_catchers(pitches, minimum, bottom=False):
    ranked = pitches.groupby("fielder_2").agg(
        takes=("called", "size"), actual=("called", "sum"), expected=("expected", "sum"),
    )
    ranked = ranked.loc[ranked["takes"].ge(minimum) & ranked["expected"].gt(0)].copy()
    ranked["sl_plus"] = 100 * ranked["actual"] / ranked["expected"]
    return ranked.sort_values(["sl_plus", "takes"], ascending=[bottom, False]).head(5)


@st.cache_data(ttl=21600, max_entries=256, show_spinner=False)
def savant_game_videos(game_id):
    reply = requests.get(
        "https://baseballsavant.mlb.com/gf", params={"game_pk": game_id}, timeout=12,
    )
    reply.raise_for_status()
    game = reply.json()
    videos = {}
    for side in ("home_pitchers", "away_pitchers"):
        for pitcher in game[side].values():
            for pitch in pitcher:
                try:
                    key = tuple(int(pitch[field]) for field in (
                        "ab_number", "pitch_number", "catcher", "batter",
                    ))
                    play_id = str(UUID(pitch["play_id"]))
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
                videos.setdefault(key, set()).add(play_id)
    return {key: ids.pop() for key, ids in videos.items() if len(ids) == 1}


def available_videos(game_id):
    try:
        return savant_game_videos(game_id)
    except (requests.RequestException, KeyError, ValueError, TypeError, AttributeError):
        return {}


def compile_clips(pitches):
    # The largest differences between the actual call and its probability come first.
    impact = (pitches["called"].astype(float) - pitches["expected"]).abs()
    candidates = pitches.assign(impact=impact).sort_values(
        ["impact", "game_date", *KEY_FIELDS], ascending=[False, False, True, True, True, True, True],
    )
    clips = []
    feeds = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        for start in range(0, min(len(candidates), 60), 10):
            batch = candidates.iloc[start:start + 10]
            missing = [int(game) for game in batch["game_pk"].unique() if game not in feeds]
            feeds.update(zip(missing, pool.map(available_videos, missing)))
            for pitch in batch.itertuples(index=False):
                key = tuple(int(getattr(pitch, field)) for field in KEY_FIELDS)
                play_id = feeds[key[0]].get(key[1:])
                if play_id:
                    clips.append((pitch, play_id))
                    if len(clips) == CLIP_COUNT:
                        return clips
    return clips


def show_clip_reel(clips):
    if len(clips) < CLIP_COUNT:
        st.caption(f"Found {len(clips)} matching Savant clips.")
    for number, (pitch, play_id) in enumerate(clips, 1):
        call = "Called Strike" if pitch.called else "Ball"
        label = f"{number}. {call} · {pitch.pitch_name or 'Pitch'} · {pitch.expected:.1%}"
        st.link_button(
            label, f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}", width="stretch",
            help=f"{pitch.game_date} · {pitch.away_team} at {pitch.home_team}",
        )


def show_sl_plus_leaderboard(season):
    st.header("SL+ Leaderboard")
    if season != 2026:
        st.info("Interactive SL+ rankings are available for the 2026 regular season.")
        return

    st.caption("Tap filters to combine them. An empty row includes all options.")
    with st.container(border=True, key="sl_filter_panel"):
        families = st.pills(
            "Pitch Group", list(PITCH_FAMILIES)[1:], selection_mode="multi", format_func=str.title,
        )
        locations = st.pills(
            "Pitch Location", LOCATIONS[1:], selection_mode="multi", format_func=str.title,
        )
        pitchers = st.pills("Pitcher Handedness", list(HANDEDNESS), selection_mode="multi")
        hitters = st.pills("Hitter Handedness", list(HANDEDNESS), selection_mode="multi")
        base_states = st.pills("Base State", list(BASE_STATES), selection_mode="multi")
        minimum = st.number_input("Minimum Eligible Takes", min_value=1, value=50, step=10)

    ranking = st.radio("Ranking", ["Top 5", "Bottom 5"], horizontal=True, key="sl_ranking")
    bottom = ranking == "Bottom 5"
    catchers, pitches = profile_data()
    matches = filter_pitches(pitches, families, locations, pitchers, hitters, base_states)
    ranked = rank_catchers(matches, minimum, bottom=bottom)
    if ranked.empty:
        st.info("No catchers meet the minimum for these filters.")
        return

    st.subheader(f"{'Bottom' if bottom else 'Top'} {len(ranked)} by SL+")
    names = catchers.set_index("catcher_id")["player"]
    filters_key = (tuple(families), tuple(locations), tuple(pitchers), tuple(hitters), tuple(base_states))
    for position, (catcher, row) in enumerate(ranked.iterrows(), 1):
        with st.container(border=True, key=f"sl_rank_{catcher}"):
            name = escape(names.at[catcher])
            st.html(
                f'<div class="sl-ranked-row"><div class="sl-ranked-player">'
                f'<h3>{position}. {name}</h3>'
                f'<span>{int(row["takes"]):,} Takes · {int(row["actual"]):,} Called Strikes'
                f' · {row["expected"]:,.1f} Expected</span></div>'
                f'<div class="sl-rank-score"><span>SL+</span><strong>{row["sl_plus"]:.2f}</strong></div>'
                f'</div>'
            )
            reel_key = f"sl_reel_{catcher}"
            with st.expander("Savant Clips"):
                cached = st.session_state.get(reel_key)
                if cached and cached[0] == filters_key:
                    show_clip_reel(cached[1])
                elif st.button("Load 10 Clips", key=f"sl_load_{catcher}", width="stretch"):
                    with st.spinner("Finding the matching pitches on Baseball Savant…"):
                        clips = compile_clips(matches.loc[matches["fielder_2"].eq(catcher)])
                    st.session_state[reel_key] = (filters_key, clips)
                    show_clip_reel(clips)
