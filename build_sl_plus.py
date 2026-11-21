r"""Build 2026 SL+ from regular-season Statcast CSV batches and ABS overturns.

Usage: python build_sl_plus.py PITCH_CSV_DIRECTORY ABS_OVERTURNS_CSV MLB_SCHEDULE_JSON
Download pitch details from Baseball Savant Statcast Search with type=details,
all=true, hfGT=R|, and inclusive game_date_gt/game_date_lt in three-day batches.
For the overturns CSV query the same season with hfABSFlag set to
is\.\.challengeabs\.\.overturned| (URL-encode the value).
Use the MLB Stats API schedule with sportId=1, season=2026, and gameType=R.
The schedule verifies coverage and the output is written beside this script.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

KEY = ["game_pk", "at_bat_number", "pitch_number"]
FIELDS = KEY + [
    "game_type", "fielder_2", "description", "plate_x", "plate_z", "sz_bot",
    "sz_top", "stand", "p_throws", "balls", "strikes", "game_date",
    "home_team", "away_team", "batter", "pitch_name", "pitch_type", "inning",
    "on_1b", "on_2b", "on_3b",
]
CALLS = ("called_strike", "ball", "blocked_ball")


def load_takes(paths, overturn_path, schedule_path):
    batches = [pd.read_csv(path, usecols=FIELDS) for path in paths]
    if any(len(batch) >= 24999 for batch in batches):
        raise ValueError("A pitch query may have reached the Statcast export limit")
    pitches = pd.concat(batches, ignore_index=True)
    schedule = json.loads(schedule_path.read_text())
    games = {
        game["gamePk"] for day in schedule["dates"] for game in day["games"]
        if game["status"]["detailedState"] in {"Final", "Completed Early"}
    }
    if len(games) != 2429 or set(pitches["game_pk"]) != games:
        raise ValueError("Pitch batches do not match the complete MLB schedule")
    if pitches.duplicated(KEY).any() or not pitches["game_type"].eq("R").all():
        raise ValueError("Expected unique regular-season pitches")

    overturns = pd.read_csv(overturn_path, usecols=KEY + ["description"])
    if len(overturns) != 5686 or overturns.duplicated(KEY).any():
        raise ValueError("Unexpected ABS overturn coverage")
    calls = pitches.set_index(KEY)["description"]
    overturned_calls = calls.reindex(pd.MultiIndex.from_frame(overturns[KEY]))
    if not overturned_calls.reset_index(drop=True).eq(overturns["description"]).all():
        raise ValueError("Savant feeds disagree on the final ABS calls")
    if overturned_calls.isna().any() or not overturned_calls.isin(CALLS).all():
        raise ValueError("Could not match every ABS overturn to a Statcast take")

    takes = pitches.loc[pitches["description"].isin(CALLS)].copy()
    takes["called"] = takes["description"].eq("called_strike")
    overturned = pd.MultiIndex.from_frame(takes[KEY]).isin(overturned_calls.index)
    takes.loc[overturned, "called"] = ~takes.loc[overturned, "called"]
    required = ["fielder_2", "plate_x", "plate_z", "sz_bot", "sz_top", "balls", "strikes"]
    zone_height = takes["sz_top"] - takes["sz_bot"]
    valid = (
        np.isfinite(takes[required]).all(axis=1)
        & zone_height.gt(0)
        & takes["stand"].isin(["L", "R"])
        & takes["p_throws"].isin(["L", "R"])
        & takes["balls"].between(0, 3)
        & takes["strikes"].between(0, 2)
    )
    print(f"{len(games)} games, {len(overturns)} reversals, {(~valid).sum()} untracked takes excluded")
    return takes.loc[valid].copy()


def spatial_prior(counts, strikes):
    # Pool nearby locations, with more weight on the center of the 3x3 grid.
    def smooth(grid):
        padded = np.pad(grid.sum(axis=(0, 1)), 1, mode="edge")
        return sum(
            weights[i] * weights[j] * padded[i:i + 40, j:j + 40]
            for i in range(3) for j in range(3)
        )
    weights = [1, 2, 1]
    return (smooth(strikes) + 0.5) / (smooth(counts) + 1)


def score(takes):
    # Separate slots for batter/pitcher handedness and every pre-pitch count.
    hands = takes["stand"].eq("R").astype(int) * 2 + takes["p_throws"].eq("R")
    count = takes["balls"].astype(int) * 3 + takes["strikes"].astype(int)
    height = (takes["plate_z"] - takes["sz_bot"]) / (takes["sz_top"] - takes["sz_bot"])
    x = np.floor((takes["plate_x"] + 2) * 10).clip(0, 39).astype(int)
    z = np.floor((height + 0.9) * 14).clip(0, 39).astype(int)
    shape = (4, 12, 40, 40)
    cells = np.ravel_multi_index((hands, count, x, z), shape)
    called = takes["called"].to_numpy(dtype=float)
    folds = takes["game_pk"].to_numpy() % 5
    predictions = np.empty(len(takes))
    for fold in range(5):
        train = folds != fold
        counts = np.bincount(cells[train], minlength=np.prod(shape)).reshape(shape)
        strikes = np.bincount(
            cells[train], weights=called[train], minlength=np.prod(shape)
        ).reshape(shape)
        prior = spatial_prior(counts, strikes)
        # Shrink sparse count/handedness cells toward the league's location baseline.
        probabilities = (strikes + 16 * prior) / (counts + 16)
        predictions[~train] = probabilities.ravel()[cells[~train]]
    print(f"{len(takes):,} scored takes; held-out Brier {(np.square(predictions - called)).mean():.4f}")
    print(f"League called rate {called.mean():.4f}; mean estimate {predictions.mean():.4f}")
    calibration = pd.DataFrame({"estimate": predictions, "actual": called})
    calibration["bucket"] = pd.cut(predictions, np.linspace(0, 1, 11))
    print(calibration.groupby("bucket", observed=True).agg(
        pitches=("actual", "size"), estimate=("estimate", "mean"), actual=("actual", "mean")
    ).round(3).to_string())
    return predictions


def names_for(player_ids):
    response = requests.get(
        "https://statsapi.mlb.com/api/v1/people",
        params={"personIds": ",".join(map(str, player_ids))},
        timeout=30,
    )
    response.raise_for_status()
    names = {person["id"]: person["fullName"] for person in response.json()["people"]}
    if set(player_ids) - names.keys():
        raise ValueError("Missing names for scored catchers")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pitch_directory", type=Path)
    parser.add_argument("overturn_csv", type=Path)
    parser.add_argument("schedule_json", type=Path)
    args = parser.parse_args()
    takes = load_takes(sorted(args.pitch_directory.glob("*.csv")), args.overturn_csv, args.schedule_json)
    takes["expected"] = score(takes)
    result = takes.groupby(takes["fielder_2"].astype(int)).agg(
        eligible_takes=("called", "size"),
        called_strikes=("called", "sum"),
        expected_strikes=("expected", "sum"),
    )
    result.index.name = "catcher_id"
    result["player"] = result.index.map(names_for(result.index))
    result["SL+"] = 100 * result["called_strikes"] / result["expected_strikes"]
    result = result[["player", "SL+", "called_strikes", "expected_strikes", "eligible_takes"]]
    result.to_csv(Path(__file__).with_name("sl_plus_2026.csv"), float_format="%.6f")
    profiles = takes[KEY + [
        "game_date", "fielder_2", "batter", "home_team", "away_team", "inning",
        "pitch_name", "pitch_type", "p_throws", "stand", "balls", "strikes",
        "plate_x", "called", "expected",
    ]].copy()
    profiles["runners_on"] = takes[["on_1b", "on_2b", "on_3b"]].notna().any(axis=1)
    profiles["zone_height"] = (
        (takes["plate_z"] - takes["sz_bot"]) / (takes["sz_top"] - takes["sz_bot"])
    ).astype("float32")
    # The Search CSV uses another zone numbering scheme. Estimate the eight
    # framing shadow zones using a one-ball-width band at the plate and zone edges.
    half_plate = 17 / 24
    ball_width = 2.9 / 12
    horizontal = np.select(
        [takes["plate_x"] < -half_plate + ball_width,
         takes["plate_x"] > half_plate - ball_width],
        [-1, 1], default=0,
    )
    vertical = np.select(
        [takes["plate_z"] > takes["sz_top"] - ball_width,
         takes["plate_z"] < takes["sz_bot"] + ball_width],
        [0, 2], default=1,
    )
    in_band_bounds = (
        takes["plate_x"].abs().le(half_plate + ball_width)
        & takes["plate_z"].ge(takes["sz_bot"] - ball_width)
        & takes["plate_z"].le(takes["sz_top"] + ball_width)
    )
    zones = 11 + 3 * vertical + horizontal + 1
    profiles["framing_zone"] = np.where(in_band_bounds & (zones != 15), zones, 0).astype("int8")
    for field in ("fielder_2", "batter"):
        profiles[field] = profiles[field].fillna(0).astype("int32")
    profiles["expected"] = profiles["expected"].astype("float32")
    profiles["plate_x"] = profiles["plate_x"].astype("float32")
    profiles.sort_values(["fielder_2", "game_date", *KEY]).to_parquet(
        Path(__file__).with_name("catcher_pitches_2026.parquet"), index=False
    )
    print(f"{len(result)} catchers")


if __name__ == "__main__":
    main()
