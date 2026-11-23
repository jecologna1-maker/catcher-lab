from concurrent.futures import ThreadPoolExecutor
from html import escape
from html.parser import HTMLParser
from urllib.parse import urlparse
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
        slaa=("slaa", "sum"), framing_runs=("framing_runs_aa", "sum"),
    )
    ranked = ranked.loc[ranked["takes"].ge(minimum) & ranked["expected"].gt(0)].copy()
    ranked["sl_plus"] = 100 * ranked["actual"] / ranked["expected"]
    return ranked.sort_values(["sl_plus", "takes"], ascending=[bottom, False]).head(5)


@st.cache_data(ttl=21600, max_entries=256, show_spinner=False)
def savant_game_pitches(game_id):
    reply = requests.get(
        "https://baseballsavant.mlb.com/gf", params={"game_pk": game_id}, timeout=12,
    )
    reply.raise_for_status()
    game = reply.json()
    pitches = {}
    for side in ("home_pitchers", "away_pitchers"):
        for pitcher in game[side].values():
            for pitch in pitcher:
                try:
                    key = tuple(int(pitch[field]) for field in (
                        "ab_number", "pitch_number", "catcher", "batter",
                    ))
                except (KeyError, TypeError, ValueError):
                    continue
                try:
                    play_id = str(UUID(pitch["play_id"]))
                except (KeyError, TypeError, ValueError, AttributeError):
                    play_id = None
                candidates = pitches.setdefault(key, {})
                candidates[play_id] = (pitch.get("pitcher_name"), pitch.get("batter_name"))
    result = {}
    for key, candidates in pitches.items():
        clips = [play_id for play_id in candidates if play_id]
        play_id = clips[0] if len(clips) == 1 else None
        pitcher, hitter = candidates.get(play_id) or next(iter(candidates.values()))
        result[key] = (play_id, pitcher, hitter)
    return result


def savant_game_videos(game_id):
    return {
        key: play_id
        for key, (play_id, _, _) in savant_game_pitches(game_id).items()
        if play_id
    }


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


class SavantVideoSource(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_player = False
        self.url = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "video":
            self.in_player = attrs.get("id") == "sporty"
        if self.in_player and tag == "source" and attrs.get("type") == "video/mp4":
            url = attrs.get("src") or ""
            parsed = urlparse(url)
            if (
                parsed.scheme == "https"
                and parsed.netloc == "sporty-clips.mlb.com"
                and parsed.path.endswith(".mp4")
            ):
                self.url = url

    def handle_endtag(self, tag):
        if tag == "video":
            self.in_player = False


def savant_clip_url(play_id):
    return f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"


@st.cache_data(ttl=3600, max_entries=256, show_spinner=False)
def load_video_source(play_id):
    reply = requests.get(savant_clip_url(play_id), timeout=10)
    reply.raise_for_status()
    parser = SavantVideoSource()
    parser.feed(reply.text)
    return parser.url


def show_clip_reel(clips, catcher):
    if len(clips) < CLIP_COUNT:
        st.caption(f"Found {len(clips)} matching Savant clips.")
    if not clips:
        return

    labels = {}
    for number, (pitch, play_id) in enumerate(clips, 1):
        call = "Called Strike" if pitch.called else "Ball"
        labels[play_id] = f"{number}. {call} · {pitch.pitch_name or 'Pitch'} · {pitch.expected:.1%}"
    selected = st.selectbox(
        "Choose a Clip", list(labels), format_func=labels.get, index=None,
        key=f"sl_clip_{catcher}", placeholder="Choose a Clip",
    )
    if selected is None:
        return
    try:
        source = load_video_source(selected)
    except (requests.RequestException, ValueError):
        source = None
    if source:
        st.video(source, format="video/mp4")
    else:
        st.caption("This clip is available on Baseball Savant.")
    st.link_button("Open on Baseball Savant", savant_clip_url(selected), width="stretch")


def reset_filters():
    for key in ("sl_family", "sl_location", "sl_pitcher", "sl_hitter", "sl_base"):
        st.session_state[key] = []
    st.session_state["sl_min"] = 50


def show_sl_plus_leaderboard(season):
    st.header("jcSL+ Leaderboard")
    if season != 2026:
        st.info("Interactive jcSL+ rankings are available for the 2026 regular season.")
        return

    with st.container(key="sl_workspace"):
        filters, results = st.columns([1, 2.25], gap="medium")
    with filters:
        ranking = st.radio("Ranking", ["Top 5", "Bottom 5"], horizontal=True, key="sl_ranking")
        with st.container(key="sl_filter_panel"):
            with st.expander("Pitch Filters", expanded=False):
                families = st.pills(
                    "Pitch Group", list(PITCH_FAMILIES)[1:], selection_mode="multi",
                    format_func=str.title, key="sl_family",
                )
                locations = st.pills(
                    "Pitch Location", LOCATIONS[1:], selection_mode="multi",
                    format_func=str.title, key="sl_location",
                )
                pitchers = st.pills(
                    "Pitcher Handedness", list(HANDEDNESS), selection_mode="multi", key="sl_pitcher",
                )
                hitters = st.pills(
                    "Hitter Handedness", list(HANDEDNESS), selection_mode="multi", key="sl_hitter",
                )
                base_states = st.pills("Base State", list(BASE_STATES), selection_mode="multi", key="sl_base")
                minimum = st.number_input(
                    "Minimum Eligible Takes", min_value=1, value=50, step=10, key="sl_min",
                )
                st.button("Reset Filters", on_click=reset_filters, key="sl_reset")
        active = [
            f"{label}: {', '.join(value.title() for value in values)}"
            for label, values in (
                ("Pitches", families), ("Location", locations), ("Pitcher", pitchers),
                ("Hitter", hitters), ("Bases", base_states),
            )
            if values
        ]
        st.caption(" · ".join(active) if active else "All pitch groups, locations, hands, and base states.")
        st.caption(f"Minimum {minimum:,} Eligible Takes")
    bottom = ranking == "Bottom 5"
    catchers, pitches = profile_data()
    matches = filter_pitches(pitches, families, locations, pitchers, hitters, base_states)
    ranked = rank_catchers(matches, minimum, bottom=bottom)
    with results, st.container(border=True, key="sl_report"):
        if ranked.empty:
            st.info("No catchers meet the minimum for these filters.")
            return

        st.subheader(f"{'Bottom' if bottom else 'Top'} {len(ranked)} by jcSL+")
        st.caption(f"{len(matches):,} matching takes · 2026 regular season")
        st.html(
            '<div class="sl-report-labels" aria-hidden="true">'
            '<span>Rank / Catcher</span><span>Filtered jcSL+</span><span>Overall jcSL+</span></div>'
        )
        season_stats = catchers.set_index("catcher_id")
        filters_key = (tuple(families), tuple(locations), tuple(pitchers), tuple(hitters), tuple(base_states))
        for position, (catcher, row) in enumerate(ranked.iterrows(), 1):
            with st.container(key=f"sl_rank_{catcher}"):
                name = escape(season_stats.at[catcher, "player"])
                overall = season_stats.at[catcher, "SL+"]
                photo = (
                    "https://img.mlbstatic.com/mlb-photos/image/upload/"
                    f"w_160,q_auto:best/v1/people/{int(catcher)}/headshot/67/current"
                )
                st.html(
                    f'<div class="sl-ranked-row">'
                    f'<span class="sl-rank-position" aria-label="Rank {position}">{position:02d}</span>'
                    f'<img class="sl-player-photo" src="{photo}" alt="" width="46" height="46" loading="lazy">'
                    f'<div class="sl-ranked-player">'
                    f'<h3>{name}</h3>'
                    f'<span>{int(row["takes"]):,} Takes · {int(row["actual"]):,} Called Strikes'
                    f' · {row["expected"]:,.1f} Expected</span>'
                    f'<span>{row["slaa"]:+.2f} jcSLAA · {row["framing_runs"]:+.2f} jcFRAA</span></div>'
                    f'<div class="sl-score-pair">'
                    f'<div class="sl-rank-score"><span>Filtered jcSL+</span><strong>{row["sl_plus"]:.2f}</strong></div>'
                    f'<div class="sl-rank-overall"><span>Overall jcSL+</span><strong>{overall:.2f}</strong></div>'
                    f'</div>'
                    f'</div>'
                )
                reel_key = f"sl_reel_{catcher}"
                with st.expander("Savant Clips"):
                    cached = st.session_state.get(reel_key)
                    if cached and cached[0] == filters_key:
                        show_clip_reel(cached[1], catcher)
                    elif st.button("Load 10 Clips", key=f"sl_load_{catcher}", width="stretch"):
                        with st.spinner("Finding the matching pitches on Baseball Savant…"):
                            clips = compile_clips(matches.loc[matches["fielder_2"].eq(catcher)])
                        st.session_state[reel_key] = (filters_key, clips)
                        show_clip_reel(clips, catcher)
