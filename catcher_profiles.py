from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps
from streamlit.components.v1 import html


@st.cache_data(max_entries=2)
def load_profile_data(season, data_version):
    folder = Path(__file__).parent
    catchers = pd.read_csv(folder / f"sl_plus_{season}.csv").sort_values("player")
    pitches = pd.read_parquet(folder / f"catcher_pitches_{season}.parquet")
    return catchers, pitches


HANDS = {"All": None, "Right-handed": "R", "Left-handed": "L"}
LOCATIONS = ["All locations", "Arm side", "Glove side", "Top", "Bottom"]
PITCH_FAMILIES = {
    "All pitch types": (),
    "Fastballs": ("FF", "FT", "SI", "FC"),
    "Breaking balls": ("SL", "ST", "SV", "CU", "KC", "CS"),
    "Offspeed": ("CH", "FS", "FO", "KN"),
}


def filter_profile(profile, pitcher, hitter, location, family):
    if HANDS[pitcher]:
        profile = profile.loc[profile["p_throws"].eq(HANDS[pitcher])]
    if HANDS[hitter]:
        profile = profile.loc[profile["stand"].eq(HANDS[hitter])]
    if location != "All locations":
        profile = profile.loc[pitch_location_mask(profile, location)]
    if PITCH_FAMILIES[family]:
        profile = profile.loc[profile["pitch_type"].isin(PITCH_FAMILIES[family])]
    return profile


def pitch_location_mask(pitches, location):
    height = pitches["zone_height"]
    if location == "Top":
        return height.ge(0.75)
    if location == "Bottom":
        return height.le(0.25)
    middle = height.between(0.25, 0.75, inclusive="neither")
    horizontal = pitches["plate_x"]
    return middle & (horizontal.gt(0) if location == "Arm side" else horizontal.lt(0))


def has_receiving_data(season):
    folder = Path(__file__).parent
    return all((folder / name).is_file() for name in (
        f"sl_plus_{season}.csv", f"catcher_pitches_{season}.parquet",
        f"called_strike_model_{season}.json",
    ))


def profile_data(season=2026):
    folder = Path(__file__).parent
    files = [folder / f"catcher_pitches_{season}.parquet", folder / f"sl_plus_{season}.csv"]
    version = tuple(path.stat().st_mtime_ns for path in files)
    return load_profile_data(season, version)


@st.cache_data
def hitter_silhouette():
    with Image.open(Path(__file__).with_name("hitter-silhouette.png")) as source:
        image = source.convert("RGBA")
    return image.crop(image.getbbox())


def heat_map(pitches, overlay=None, hitter=None):
    # Smooth counts and strike totals separately before calculating rates.
    x_edges = np.linspace(-2.5, 2.5, 101)
    z_edges = np.linspace(-2.25, 3.25, 111)
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
    palette = [
        (0.00, "#243ba3"), (0.20, "#2b83cb"), (0.38, "#71d7d3"),
        (0.50, "#dcf0c4"), (0.62, "#f6d55e"), (0.80, "#ef8a40"),
        (1.00, "#b83431"),
    ]
    difference = actual - expected
    color_index = (difference + 0.15) / 0.30
    stops = [stop for stop, _ in palette]
    colors = np.array([
        [int(color[channel:channel + 2], 16) for channel in (1, 3, 5)]
        for _, color in palette
    ])
    rgb = [np.interp(color_index, stops, channel) for channel in colors.T]
    alpha = 255 * np.clip((support - 3) / 18, 0, 1)
    pixels = np.stack([*rgb, alpha], axis=-1).transpose(1, 0, 2)[::-1].astype("uint8")
    surface = Image.fromarray(pixels).resize((600, 660), Image.Resampling.BILINEAR)
    figure = go.Figure(go.Heatmap(
        x=x, y=z, z=np.where(support >= 3, difference, np.nan).T,
        opacity=0.001, hoverongaps=False,
        customdata=np.stack([actual.T, expected.T, support.T], axis=-1),
        colorscale=palette,
        zmin=-0.15, zmax=0.15,
        colorbar={
            "title": "Called strikes<br>vs. expected",
            "tickvals": [-0.15, -0.075, 0, 0.075, 0.15],
            "ticktext": ["−15 pp", "−7.5 pp", "0", "+7.5 pp", "+15 pp"],
            "len": 0.76, "thickness": 14,
        },
        hovertemplate=(
            "Smoothed called strike rate %{customdata[0]:.1%}<br>"
            "Smoothed expected rate %{customdata[1]:.1%}<br>"
            "Actual minus expected %{z:+.1%}<br>"
            "Weighted nearby takes %{customdata[2]:.1f}<extra></extra>"
        ),
    ))
    figure.add_layout_image(
        source=surface, xref="x", yref="y", x=x_edges[0], y=z_edges[-1],
        sizex=x_edges[-1] - x_edges[0], sizey=z_edges[-1] - z_edges[0],
        sizing="stretch", layer="below",
    )
    silhouette = hitter_silhouette()
    # Catcher's perspective: a right-handed hitter stands on the viewer's left.
    for side, left, image in (("R", -2.48, silhouette), ("L", 0.88, ImageOps.mirror(silhouette))):
        if hitter is None or hitter == side:
            figure.add_layout_image(
                source=image, xref="x", yref="y", x=left, y=1.37,
                sizex=1.60, sizey=2.18, sizing="stretch", layer="below", opacity=0.60,
            )
    if overlay is not None:
        for call, color in (("Ball", "#bb4b37"), ("Called strike", "#17629c")):
            dots = overlay.loc[overlay["Initial call"].eq(call)]
            figure.add_scattergl(
                x=dots["plate_x"], y=dots["zone_height"], mode="markers", name=call,
                customdata=dots[[
                    "game_pk", "Plate appearance", "Pitch number", "fielder_2", "batter",
                ]].to_numpy(dtype=int).tolist(),
                marker={"size": 6, "color": color, "opacity": 0.72, "line": {"color": "white", "width": 0.5}},
                text=[
                    f"{date} · {pitch} · {call}<br>Count {count} · Probability {probability:.1%}"
                    for date, pitch, count, probability in dots[[
                        "Date", "Pitch", "Count", "Called-strike probability",
                    ]].itertuples(index=False, name=None)
                ],
                hovertemplate="%{text}<extra></extra>",
            )
    for width, color in ((4, "#102540"), (1.4, "#ffffff")):
        figure.add_shape(
            type="rect", x0=-17 / 24, x1=17 / 24, y0=0, y1=1,
            line={"color": color, "width": width},
        )
    figure.add_shape(
        type="path", path="M -0.69,-0.77 L 0.69,-0.77 L 0.69,-0.89 L 0,-1.05 L -0.69,-0.89 Z",
        line={"color": "#64748b", "width": 1.5}, fillcolor="#e2e8f0",
    )
    visible_y = [-1.15, 2.05]
    if overlay is not None and not overlay.empty:
        visible_y = [
            min(visible_y[0], overlay["zone_height"].min() - 0.08),
            max(visible_y[1], overlay["zone_height"].max() + 0.08),
        ]
    figure.update_layout(
        template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font={"color": "#342044"},
        height=615, clickmode="event", hovermode="closest",
        margin={"l": 40, "r": 10, "t": 24, "b": 45},
        legend={"orientation": "h", "y": 1.04},
        xaxis={"title": "Horizontal location (ft)", "range": [x_edges[0], x_edges[-1]]},
        yaxis={"title": "Batter strike-zone height", "range": visible_y},
    )
    return figure


@st.cache_data
def load_stance_usage(file_version):
    source = Path(__file__).with_name("catcher_stance_2026.csv")
    return pd.read_csv(source).set_index("id")


def show_stance_usage(catcher):
    st.subheader("Catcher stance usage")
    source = Path(__file__).with_name("catcher_stance_2026.csv")
    stance = load_stance_usage(source.stat().st_mtime_ns)
    if catcher not in stance.index:
        st.caption("Baseball Savant has no stance data for this catcher.")
        return
    row = stance.loc[catcher]
    total = int(row["pitches"])
    st.caption(
        f"{total:,} pitches in Savant’s 2026 regular-season stance breakdown. "
        "Stance is measured at pitch release."
    )
    traditional, left, right = st.columns(3)
    with traditional:
        st.markdown("**Traditional**")
        st.metric("Both knees up", f"{row['traditional'] / total:.1%}")
    for column, side in ((left, "Left"), (right, "Right")):
        with column:
            st.markdown(f"**{side} knee down**")
            st.metric("No leg extended", f"{row[f'{side.lower()}_no_extension'] / total:.1%}")
            st.metric("Leg extended", f"{row[f'{side.lower()}_extended'] / total:.1%}")
    st.caption("Both knees down are counted under Right knee down · Leg extended.")
    st.caption(
        "Source: [Baseball Savant Catcher Stance](https://baseballsavant.mlb.com/"
        "leaderboard/catcher-stance?type=catcher&seasonStart=2026&seasonEnd=2026&groupBy=knee_code)."
    )


def catcher_profile_url(season, catcher):
    return f"https://cologna-catcher-lab.streamlit.app/?view=catcher&season={season}&catcher={int(catcher)}"


def show_catcher_profile(season, requested_catcher=None):
    if not has_receiving_data(season):
        st.info(f"Catcher profiles are unavailable for the {season} season.")
        return

    catchers, pitches = profile_data(season)
    roster = catchers.set_index("catcher_id")
    player_ids = roster.index.tolist()
    profile_key = f"catcher_profile_{season}"
    if profile_key not in st.session_state and requested_catcher in player_ids:
        st.session_state[profile_key] = requested_catcher
    catcher = st.sidebar.selectbox(
        "Catcher", player_ids,
        format_func=lambda player: roster.at[player, "player"], key=profile_key,
    )
    player = roster.loc[catcher]
    profile = pitches.loc[pitches["fielder_2"].eq(catcher)]
    st.header(player["player"])
    st.caption(f"{season} regular season · Catcher profile")
    show_receiving_profile(profile)
    if season == 2026:
        with st.container(border=True, key="stance_panel"):
            show_stance_usage(catcher)


def show_receiving_profile(profile):
    metrics = st.container(key="receiving_summary")

    st.subheader("Called strikes by location")
    hand_filters = st.columns(2)
    pitcher = hand_filters[0].selectbox("Pitcher handedness", HANDS, key="profile_pitcher")
    hitter = hand_filters[1].selectbox("Hitter handedness", HANDS, key="profile_hitter")
    pitch_filters = st.columns(2)
    location = pitch_filters[0].selectbox("Pitch location", LOCATIONS)
    family = pitch_filters[1].selectbox("Pitch family", PITCH_FAMILIES)
    profile = filter_profile(profile, pitcher, hitter, location, family)
    actual = int(profile["called"].sum())
    expected = profile["expected"].astype("float64").sum()
    with metrics:
        columns = st.columns(4)
        columns[0].metric("jcSL+", f"{100 * actual / expected:.2f}" if expected > 0 else "—")
        columns[1].metric("Called strikes", f"{actual:,}")
        columns[2].metric("Expected strikes", f"{expected:,.1f}")
        columns[3].metric("Eligible takes", f"{len(profile):,}")
        strikes, runs = st.columns(2)
        strikes.metric("jcSLAA", f"{profile['slaa'].sum():+.2f}")
        runs.metric("jcFRAA", f"{profile['framing_runs_aa'].sum():+.2f}")
    st.caption(
        "Warm colors show more called strikes than expected. Blue shows fewer. "
        "The surface averages nearby takes that match your filters. "
        "Height 0 is the bottom of the batter’s zone and 1 is the top."
    )
    if location != "All locations":
        st.caption(
            "Arm and glove side cover the middle half of zone height, split left and right from the catcher’s view. "
            "Top and bottom use the outer quarters of zone height and include pitches beyond the zone."
        )
    if profile.empty:
        st.info("No eligible takes match these filters.")
        st.container(width=690).plotly_chart(heat_map(profile, hitter=HANDS[hitter]), use_container_width=True, theme=None)
        return
    if not st.toggle("Overlay individual pitches", key="show_pitch_overlay"):
        st.container(width=690).plotly_chart(heat_map(profile, hitter=HANDS[hitter]), use_container_width=True, theme=None)
        return

    overlay = profile.loc[profile["expected"].between(0.05, 0.95, inclusive="neither")].copy()
    outcome = st.radio(
        "Pitch outcome", ["All pitches", "Balls only", "Called strikes only"], horizontal=True
    )
    if outcome != "All pitches":
        overlay = overlay.loc[overlay["called"].eq(outcome == "Called strikes only")]
    if overlay.empty:
        st.caption("No pitches match this outcome at probabilities above 5% and below 95%.")
        st.container(width=690).plotly_chart(heat_map(profile, hitter=HANDS[hitter]), use_container_width=True, theme=None)
        return

    overlay["Initial call"] = overlay["called"].map({True: "Called strike", False: "Ball"})
    overlay["Count"] = overlay["balls"].astype(str) + "–" + overlay["strikes"].astype(str)
    overlay = overlay.rename(columns={
        "pitch_name": "Pitch", "inning": "Inning", "at_bat_number": "Plate appearance",
        "pitch_number": "Pitch number", "expected": "Called-strike probability",
        "game_date": "Date",
    })
    st.caption(
        f"{len(overlay):,} matching pitches with estimated probabilities "
        "above 5% and below 95%. Click a pitch to open its Savant clip in a new tab."
    )
    figure = heat_map(profile, overlay, hitter=HANDS[hitter])
    script = Path(__file__).with_name("pitch_video.js").read_text(encoding="utf-8")
    contents = figure.to_html(
        full_html=False, include_plotlyjs="cdn", post_script=script,
        config={"responsive": True, "displaylogo": False},
    )
    with st.container(width=690):
        html('<style>body{margin:0} .scatterlayer{cursor:pointer}</style>' + contents, height=625)
