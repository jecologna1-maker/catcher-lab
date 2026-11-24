"""Build a 2026 dirt-only blocking leaderboard from Statcast and MLB play-by-play.

Usage: python build_blocking.py STATCAST_CSV_DIR PLAY_BY_PLAY_DIR MLB_SCHEDULE_JSON
The play-by-play directory contains one <gamePk>.json per completed game, from
https://statsapi.mlb.com/api/v1/game/<gamePk>/playByPlay (full response or allPlays array).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_sl_plus import KEY, SEASON_GAMES, names_for
from called_strike_model import fit_logistic, sigmoid

BALL_RADIUS = 1.45 / 12
# The six-foot box is centered at the midpoint of the plate (OBR Appendix 2).
CUTOFF_BEHIND_MID_PLATE_FT = 3
TAKES_AND_MISSES = {
    "ball", "blocked_ball", "called_strike", "swinging_strike",
    "swinging_strike_blocked", "missed_bunt",
}
SWINGING_STRIKES = {"swinging_strike", "swinging_strike_blocked"}
THIRD_STRIKES = {*SWINGING_STRIKES, "missed_bunt", "called_strike"}
MISPLAYS = {"wild_pitch", "passed_ball"}
FIELDS = [
    *KEY, "game_type", "game_date", "fielder_2", "description", "plate_x",
    "plate_z", "vy0", "vz0", "ay", "az", "pfx_x", "pfx_z", "release_speed",
    "stand", "p_throws", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "batter", "balls", "pitch_type", "pitch_name", "sz_bot", "sz_top",
]


def official_misplays(path):
    plays = json.loads(path.read_text())
    if isinstance(plays, dict):
        plays = plays["allPlays"]
    for appearance in plays:
        event_index = appearance["about"]["atBatIndex"] + 1
        marked = {
            runner["details"]["playIndex"] for runner in appearance["runners"]
            if runner["details"].get("eventType") in MISPLAYS
        }
        pitch_number = None
        for event in appearance["playEvents"]:
            if event.get("isPitch"):
                pitch_number = event["pitchNumber"]
            if event["index"] in marked or event["details"].get("eventType") in MISPLAYS:
                if pitch_number is None:
                    raise ValueError(f"Misplay without a preceding pitch in {path.stem}, PA {event_index}")
                yield int(path.stem), event_index, pitch_number


def cutoff_height(pitches):
    """Project 2026 mid-plate tracking coordinates to the dirt cutoff."""
    vy, vz, ay, az = (pitches[field].to_numpy() for field in ("vy0", "vz0", "ay", "az"))

    def time_at(distance):
        with np.errstate(invalid="ignore", divide="ignore"):
            return 2 * distance / (-vy + np.sqrt(vy**2 - 2 * ay * distance))

    middle = time_at(50 - 17 / 24)
    cutoff = time_at(50 - 17 / 24 + CUTOFF_BEHIND_MID_PLATE_FT)
    return pitches["plate_z"] + (cutoff - middle) * (vz + az * (cutoff + middle) / 2)


def model_inputs(pitches):
    x = pitches["plate_x"].clip(-4, 4).to_numpy()
    depth = np.clip(BALL_RADIUS - pitches["cutoff_height"].to_numpy(), 0, 3)
    speed = (pitches["release_speed"].fillna(85).to_numpy() - 85) / 10
    horizontal = pitches["pfx_x"].fillna(0).clip(-3, 3).to_numpy()
    vertical = pitches["pfx_z"].fillna(0).clip(-3, 3).to_numpy()
    return np.column_stack([
        10 * np.ones(len(x)), x, np.abs(x), x*x, depth, depth*depth, np.abs(x)*depth,
        speed, horizontal, vertical, speed*depth,
        pitches["stand"].eq("R"), pitches["p_throws"].eq("R"),
        pitches["on_1b"].notna(), pitches["on_2b"].notna(), pitches["on_3b"].notna(),
        pitches["batter_can_reach"], pitches["outs_when_up"].eq(2),
    ])


def cross_fit(pitches):
    inputs = model_inputs(pitches)
    outcomes = pitches["misplay"].to_numpy(dtype=float)
    predictions = np.empty(len(pitches))
    baseline = np.empty(len(pitches))
    folds = pitches["game_pk"].to_numpy() % 5
    penalties = (3, 10, 30, 100)
    for fold in range(5):
        test = folds == fold
        training_folds = [inner for inner in range(5) if inner != fold]
        errors = np.zeros(len(penalties))
        for split in training_folds:
            validation = folds == split
            training = ~(test | validation)
            for i, penalty in enumerate(penalties):
                coefficients = fit_logistic(inputs[training], outcomes[training], penalty)
                errors[i] += np.square(outcomes[validation] - sigmoid(inputs[validation] @ coefficients)).sum()
        penalty = penalties[int(errors.argmin())]
        coefficients = fit_logistic(inputs[~test], outcomes[~test], penalty)
        predictions[test] = sigmoid(inputs[test] @ coefficients)
        baseline[test] = outcomes[~test].mean()
    brier = np.mean((outcomes - predictions)**2)
    reference = np.mean((outcomes - baseline)**2)
    print(f"Held-out Brier score {brier:.5f}; constant-rate baseline {reference:.5f}")
    if not (0 < predictions.min() <= predictions.max() < 1 and brier < reference):
        raise ValueError("Blocking model failed held-out validation")
    return predictions


def dirt_opportunities(raw, misplays):
    eligible = raw.loc[raw["description"].isin(TAKES_AND_MISSES)].copy()
    eligible["cutoff_height"] = cutoff_height(eligible)
    eligible["batter_can_reach"] = (
        eligible["strikes"].eq(2) & eligible["description"].isin(THIRD_STRIKES)
        & (eligible["on_1b"].isna() | eligible["outs_when_up"].eq(2))
    )
    runners_on = eligible[["on_1b", "on_2b", "on_3b"]].notna().any(axis=1)
    swinging_third_strike = eligible["strikes"].eq(2) & eligible["description"].isin(SWINGING_STRIKES)
    eligible = eligible.loc[
        (runners_on | swinging_third_strike)
        & eligible["fielder_2"].notna() & eligible["plate_x"].notna()
        & eligible["vy0"].lt(0) & np.isfinite(eligible["cutoff_height"])
        & eligible["cutoff_height"].le(BALL_RADIUS)
    ].copy()
    eligible["misplay"] = pd.MultiIndex.from_frame(eligible[KEY]).isin(misplays)
    return eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("statcast", type=Path)
    parser.add_argument("play_by_play", type=Path)
    parser.add_argument("schedule", type=Path)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    games = {
        game["gamePk"] for day in schedule["dates"] for game in day["games"]
        if game["status"]["detailedState"] in {"Final", "Completed Early"}
    }
    if len(games) != SEASON_GAMES[2026]:
        raise ValueError("Unexpected 2026 schedule coverage")
    misplays = {
        pitch for game in sorted(games)
        for pitch in official_misplays(args.play_by_play / f"{game}.json")
    }
    batches = [pd.read_csv(path, usecols=FIELDS) for path in sorted(args.statcast.glob("*.csv"))]
    raw = pd.concat([batch for batch in batches if not batch.empty], ignore_index=True)
    if (not raw["game_type"].eq("R").all() or not raw["game_date"].str.startswith("2026-").all()
            or raw.duplicated(KEY).any() or set(raw["game_pk"]) != games):
        raise ValueError("Incomplete or invalid 2026 Statcast pitch coverage")
    matched = len(misplays.intersection(pd.MultiIndex.from_frame(raw[KEY])))
    if matched < 0.99 * len(misplays):
        raise ValueError(f"Only {matched} of {len(misplays)} official misplays matched a pitch")
    chances = dirt_opportunities(raw, misplays)
    print(f"{matched}/{len(misplays)} official misplays matched; {len(chances):,} qualifying chances; "
          f"{chances['misplay'].sum():,} official misplays in the dirt-only set")
    chances["expected_misplays"] = cross_fit(chances)
    review = chances[KEY + [
        "game_date", "fielder_2", "batter", "balls", "strikes", "pitch_type",
        "pitch_name", "p_throws", "stand", "plate_x", "misplay", "expected_misplays",
    ]].copy()
    review["zone_height"] = (
        (chances["plate_z"] - chances["sz_bot"]) / (chances["sz_top"] - chances["sz_bot"])
    ).where(chances["sz_top"].gt(chances["sz_bot"]))
    review["runners_on"] = chances[["on_1b", "on_2b", "on_3b"]].notna().any(axis=1)
    review.to_parquet(Path(__file__).with_name("blocking_pitches_2026.parquet"), index=False)
    summaries = chances.groupby(chances["fielder_2"].astype(int)).agg(
        opportunities=("misplay", "size"), actual_misplays=("misplay", "sum"),
        expected_misplays=("expected_misplays", "sum"),
    )
    summaries.index.name = "catcher_id"
    # Use the existing season roster so a rebuild can run from cached data.
    roster_path = Path(__file__).with_name("sl_plus_2026.csv")
    known_names = pd.read_csv(roster_path).set_index("catcher_id")["player"].to_dict()
    missing = summaries.index.difference(known_names)
    if len(missing):
        known_names.update(names_for(missing))
    summaries["player"] = summaries.index.map(known_names)
    summaries["blocks_above_average"] = summaries["expected_misplays"] - summaries["actual_misplays"]
    summaries.to_csv(Path(__file__).with_name("dirt_blocking_2026.csv"), float_format="%.6f")


if __name__ == "__main__":
    main()
