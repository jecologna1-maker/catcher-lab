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


HALF_PLATE = 17 / 24


def home_plate_path(front, depth):
    side = front - depth / 2
    tip = front - depth
    return (f"M {-HALF_PLATE},{front} L {HALF_PLATE},{front} "
            f"L {HALF_PLATE},{side} L 0,{tip} L {-HALF_PLATE},{side} Z")


@st.cache_data
def hitter_silhouette():
    with Image.open(Path(__file__).with_name("hitter-silhouette.png")) as source:
        image = source.convert("RGBA")
    alpha = image.getchannel("A")
    silhouette = Image.new("RGBA", image.size, "#141019")
    silhouette.putalpha(alpha)
    return silhouette.crop(alpha.getbbox())


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
    actual = density(pitches["called"].astype(float))
    expected = density(pitches["expected"])
    jcsl = np.divide(100 * actual, expected, out=np.full_like(expected, 100.0), where=expected > 0)
    palette = [
        (0.00, "#503068"), (0.25, "#9e76b1"), (0.50, "#f5f1f3"),
        (0.72, "#f4d280"), (1.00, "#aa5d17"),
    ]
    color_min, color_max = 70, 130
    color_index = (jcsl - color_min) / (color_max - color_min)
    stops = [stop for stop, _ in palette]
    colors = np.array([
        [int(color[channel:channel + 2], 16) for channel in (1, 3, 5)]
        for _, color in palette
    ])
    rgb = [np.interp(color_index, stops, channel) for channel in colors.T]
    # Fade sparse areas and locations where the expected-strike denominator is tiny.
    opacity = np.minimum(np.clip((support - 3) / 18, 0, 1), np.clip((expected - 2) / 8, 0, 1))
    alpha = 255 * opacity
    pixels = np.stack([*rgb, alpha], axis=-1).transpose(1, 0, 2)[::-1].astype("uint8")
    surface = Image.fromarray(pixels).resize((600, 660), Image.Resampling.BILINEAR)
    figure = go.Figure(go.Heatmap(
        x=x, y=z, z=np.where(opacity > 0, jcsl, np.nan).T,
        opacity=0.001, hoverongaps=False,
        customdata=np.stack([actual.T, expected.T, support.T], axis=-1),
        colorscale=palette,
        zmin=color_min, zmax=color_max,
        colorbar={
            "title": {"text": "jcSL+", "side": "top"},
            "orientation": "h", "x": 0.5, "xanchor": "center",
            "y": 1.02, "yanchor": "bottom",
            "tickvals": [70, 85, 100, 115, 130],
            "ticktext": ["≤70", "85", "100", "115", "≥130"],
            "len": 0.66, "thickness": 12, "outlinewidth": 0,
        },
        hovertemplate=(
            "Local jcSL+ %{z:.1f}<br>"
            "Weighted called strikes %{customdata[0]:.1f}<br>"
            "Weighted expected strikes %{customdata[1]:.1f}<br>"
            "Weighted nearby takes %{customdata[2]:.1f}<extra></extra>"
        ),
    ))
    figure.add_layout_image(
        source=surface, xref="x", yref="y", x=x_edges[0], y=z_edges[-1],
        sizex=x_edges[-1] - x_edges[0], sizey=z_edges[-1] - z_edges[0],
        sizing="stretch", layer="below",
    )
    silhouette = hitter_silhouette()
    plate_front = -0.77
    hitter_height = 2.992  # In units of the normalized, one-unit strike-zone height.
    # Catcher's perspective: a right-handed hitter stands on the viewer's left.
    for side, left, image in (("R", -2.48, silhouette), ("L", 0.88, ImageOps.mirror(silhouette))):
        if hitter is None or hitter == side:
            figure.add_layout_image(
                source=image, xref="x", yref="y", x=left, y=plate_front + hitter_height,
                sizex=1.60, sizey=hitter_height, sizing="stretch", layer="below", opacity=0.84,
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
    figure.add_shape(
        type="rect", x0=-HALF_PLATE, x1=HALF_PLATE, y0=0, y1=1,
        line={"color": "#4b2d62", "width": 3},
    )
    figure.add_shape(
        type="path", path=home_plate_path(plate_front, 0.28),
        line={"color": "#64748b", "width": 1.5}, fillcolor="#e2e8f0",
    )
    visible_y = [-1.15, plate_front + hitter_height + 0.10]
    chart_height = round(69 + (615 - 69) * (visible_y[1] - visible_y[0]) / 3.20)
    if overlay is not None and not overlay.empty:
        visible_y = [
            min(visible_y[0], overlay["zone_height"].min() - 0.08),
            max(visible_y[1], overlay["zone_height"].max() + 0.08),
        ]
    figure.update_layout(
        template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font={"color": "#342044"},
        height=chart_height + 62, clickmode="event", hovermode="closest",
        margin={"l": 12, "r": 12, "t": 86, "b": 45},
        legend={"orientation": "h", "y": -0.01, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        xaxis={"visible": False, "range": [x_edges[0], x_edges[-1]]},
        yaxis={"visible": False, "range": visible_y},
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


@st.cache_data(max_entries=12)
def load_catcher_season(catcher, season, data_version):
    file = Path(__file__).with_name(f"catcher_pitches_{season}.parquet")
    return pd.read_parquet(file, filters=[("fielder_2", "==", catcher)])


def show_catcher_profile(year_range, catcher, filters):
    if catcher is None:
        st.info("Use Player Search to open a catcher’s profile.")
        return

    start, end = year_range
    seasons = [season for season in range(start, end + 1) if has_receiving_data(season)]
    player = None
    slices = []
    for season in seasons:
        folder = Path(__file__).parent
        roster = pd.read_csv(folder / f"sl_plus_{season}.csv").set_index("catcher_id")
        if catcher in roster.index:
            player = roster.at[catcher, "player"]
            data = folder / f"catcher_pitches_{season}.parquet"
            slices.append(load_catcher_season(catcher, season, data.stat().st_mtime_ns))
    if not slices:
        st.info("This catcher has no tracked pitches in the selected season range.")
        return

    profile = pd.concat(slices, ignore_index=True)
    st.header(player)
    label = str(start) if start == end else f"{start}–{end}"
    st.caption(f"{label} regular season · Catcher profile")
    show_receiving_profile(profile, filters)
    if 2026 in seasons:
        with st.container(border=True, key="stance_panel"):
            show_stance_usage(catcher)


def show_receiving_profile(profile, filters=None):
    with (filters if filters is not None else st.container()):
        pitcher_column, hitter_column, location_column, family_column = st.columns(4, gap="small")
        pitcher = pitcher_column.selectbox("Pitcher handedness", HANDS, key="profile_pitcher")
        hitter = hitter_column.selectbox("Hitter handedness", HANDS, key="profile_hitter")
        location = location_column.selectbox("Pitch location", LOCATIONS)
        family = family_column.selectbox("Pitch family", PITCH_FAMILIES)
        show_overlay = st.toggle("Overlay individual pitches", key="show_pitch_overlay")
        outcome = (
            st.radio("Pitch outcome", ["All pitches", "Balls only", "Called strikes only"], horizontal=True)
            if show_overlay else "All pitches"
        )
    profile = filter_profile(profile, pitcher, hitter, location, family)
    metrics = st.container(key="receiving_summary")
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
    st.subheader("jcSL+ by pitch location")
    st.caption(
        "Gold shows more called strikes than expected, and purple shows fewer. "
        "Local jcSL+ compares actual and expected strikes on nearby pitches that match your filters. "
        "Areas with too few expected strikes fade out."
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
    if not show_overlay:
        st.container(width=690).plotly_chart(heat_map(profile, hitter=HANDS[hitter]), use_container_width=True, theme=None)
        return

    overlay = profile.loc[profile["expected"].between(0.05, 0.95, inclusive="neither")].copy()
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
        html('<style>body{margin:0} .scatterlayer{cursor:pointer}</style>' + contents, height=figure.layout.height)
