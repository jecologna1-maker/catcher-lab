r"""Build season-specific jcSL+ from regular-season Statcast CSV batches.

Usage: python build_sl_plus.py PITCH_CSV_DIRECTORY ABS_OVERTURNS_CSV_OR_DASH MLB_SCHEDULE_JSON --season YEAR --enhanced
Download pitch details from Baseball Savant Statcast Search with type=details,
all=true, hfGT=R|, and inclusive game_date_gt/game_date_lt in three-day batches.
For 2026, query the overturns CSV with hfABSFlag set to
is\.\.challengeabs\.\.overturned| (URL-encode the value).
Pass a dash in place of the overturns CSV for 2020–2025.
Use the MLB Stats API schedule with sportId=1, the selected season, and gameType=R.
The schedule verifies coverage and the output is written beside this script.
For --review or --enhanced add hydrate=officials to the schedule request. The
enhanced build adds pitch trajectory and the
official home-plate umpire. Use --review to evaluate without replacing scores.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from framing_metrics import RUN_FIELDS, strike_values

from called_strike_model import (
    TRAJECTORY_FIELDS, chronological_review, cross_fit, metrics, model_features,
)

KEY = ["game_pk", "at_bat_number", "pitch_number"]
FIELDS = KEY + [
    "game_type", "fielder_2", "description", "plate_x", "plate_z", "sz_bot",
    "sz_top", "stand", "p_throws", "balls", "strikes", "game_date",
    "home_team", "away_team", "batter", "pitch_name", "pitch_type", "inning",
    "on_1b", "on_2b", "on_3b",
]
CALLS = ("called_strike", "ball", "blocked_ball")
SEASON_GAMES = {2020: 898, 2021: 2429, 2022: 2430, 2023: 2430, 2024: 2429, 2025: 2430, 2026: 2429}


def load_takes(paths, overturn_path, schedule_path, extra_fields=(), season=2026):
    fields = [*FIELDS, *extra_fields]
    batches = [pd.read_csv(path, usecols=fields) for path in paths]
    if any(len(batch) >= 24999 for batch in batches):
        raise ValueError("A pitch query may have reached the Statcast export limit")
    pitches = pd.concat([batch for batch in batches if not batch.empty], ignore_index=True)
    schedule = json.loads(schedule_path.read_text())
    games = {
        game["gamePk"] for day in schedule["dates"] for game in day["games"]
        if game["status"]["detailedState"] in {"Final", "Completed Early"}
    }
    pitch_games = set(pitches["game_pk"])
    if len(games) != SEASON_GAMES[season] or pitch_games != games:
        raise ValueError(
            f"{season} game coverage: {len(pitch_games)}/{len(games)}; "
            f"missing {sorted(games - pitch_games)}, unexpected {sorted(pitch_games - games)}"
        )
    if not pitches["game_date"].str.startswith(f"{season}-").all():
        raise ValueError("Pitch dates must be from the selected season")
    if pitches.duplicated(KEY).any() or not pitches["game_type"].eq("R").all():
        raise ValueError("Expected unique regular-season pitches")

    takes = pitches.loc[pitches["description"].isin(CALLS)].copy()
    takes["called"] = takes["description"].eq("called_strike")
    reversal_count = 0
    if season == 2026:
        overturns = pd.read_csv(overturn_path, usecols=KEY + ["description"])
        if len(overturns) != 5686 or overturns.duplicated(KEY).any():
            raise ValueError("Unexpected ABS overturn coverage")
        calls = pitches.set_index(KEY)["description"]
        overturned_calls = calls.reindex(pd.MultiIndex.from_frame(overturns[KEY]))
        if not overturned_calls.reset_index(drop=True).eq(overturns["description"]).all():
            raise ValueError("Savant feeds disagree on the final ABS calls")
        if not overturned_calls.isin(CALLS).all():
            raise ValueError("Could not match every ABS overturn to a Statcast take")
        overturned = pd.MultiIndex.from_frame(takes[KEY]).isin(overturned_calls.index)
        takes.loc[overturned, "called"] = ~takes.loc[overturned, "called"]
        reversal_count = len(overturns)
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
    print(f"{len(games)} games, {reversal_count} reversals, {(~valid).sum()} untracked takes excluded")
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


def baseline_predictor(takes):
    # Separate slots for batter/pitcher handedness and every pre-pitch count.
    hands = takes["stand"].eq("R").astype(int) * 2 + takes["p_throws"].eq("R")
    count = takes["balls"].astype(int) * 3 + takes["strikes"].astype(int)
    height = (takes["plate_z"] - takes["sz_bot"]) / (takes["sz_top"] - takes["sz_bot"])
    x = np.floor((takes["plate_x"] + 2) * 10).clip(0, 39).astype(int)
    z = np.floor((height + 0.9) * 14).clip(0, 39).astype(int)
    shape = (4, 12, 40, 40)
    cells = np.ravel_multi_index((hands, count, x, z), shape)
    called = takes["called"].to_numpy(dtype=float)
    def predict(train, test):
        counts = np.bincount(cells[train], minlength=np.prod(shape)).reshape(shape)
        strikes = np.bincount(cells[train], weights=called[train], minlength=np.prod(shape)).reshape(shape)
        prior = spatial_prior(counts, strikes)
        probabilities = (strikes + 16 * prior) / (counts + 16)
        return probabilities.ravel()[cells[test]]

    return predict


def score(takes):
    predict = baseline_predictor(takes)
    called = takes["called"].to_numpy(dtype=float)
    folds = takes["game_pk"].to_numpy() % 5
    predictions = np.empty(len(takes))
    for fold in range(5):
        test = folds == fold
        predictions[test] = predict(~test, test)
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
    parser.add_argument("--season", type=int, choices=SEASON_GAMES, default=2026)
    parser.add_argument("--review", action="store_true", help="Compare models without updating outputs")
    parser.add_argument("--enhanced", action="store_true", help="Build with the enhanced model")
    args = parser.parse_args()
    extra_fields = TRAJECTORY_FIELDS if args.review or args.enhanced else ()
    takes = load_takes(sorted(args.pitch_directory.glob("*.csv")), args.overturn_csv,
                       args.schedule_json, extra_fields=[*extra_fields, *RUN_FIELDS], season=args.season)
    if extra_fields:
        features = model_features(takes, args.schedule_json, season=args.season)
        review = chronological_review(takes, features, baseline_predictor(takes), season=args.season)
        if args.review:
            return
        if any(
            review["challenger"][group][metric] >= review["baseline"][group][metric]
            for group in ("all", "edge") for metric in ("brier", "log_loss")
        ):
            raise ValueError("Enhanced model did not improve both held-out probability scores")
        takes["expected"] = cross_fit(takes, features)
        review["cross_fitted"] = metrics(
            takes["called"].to_numpy(dtype=float), takes["expected"].to_numpy(),
            features["signed_edge_distance"].abs().le(0.25).to_numpy(),
        )
        Path(__file__).with_name(f"called_strike_model_{args.season}.json").write_text(
            json.dumps(review, indent=2) + "\n", encoding="utf-8"
        )
    else:
        takes["expected"] = score(takes)
    takes["strike_run_value"] = strike_values(takes)
    takes["slaa"] = takes["called"].astype(float) - takes["expected"].astype("float32").astype(float)
    takes["framing_runs_aa"] = takes["slaa"] * takes["strike_run_value"]
    result = takes.groupby(takes["fielder_2"].astype(int)).agg(
        eligible_takes=("called", "size"),
        called_strikes=("called", "sum"),
        expected_strikes=("expected", "sum"),
    )
    result.index.name = "catcher_id"
    result["player"] = result.index.map(names_for(result.index))
    result["SL+"] = 100 * result["called_strikes"] / result["expected_strikes"]
    result = result[["player", "SL+", "called_strikes", "expected_strikes", "eligible_takes"]]
    contextual = takes.groupby(takes["fielder_2"].astype(int))[["slaa", "framing_runs_aa"]].sum()
    result["SLAA"] = contextual["slaa"]
    result["Framing Runs Above Average"] = contextual["framing_runs_aa"]
    result.to_csv(Path(__file__).with_name(f"sl_plus_{args.season}.csv"), float_format="%.6f")
    profiles = takes[KEY + [
        "game_date", "fielder_2", "batter", "home_team", "away_team", "inning",
        "pitch_name", "pitch_type", "p_throws", "stand", "balls", "strikes",
        "plate_x", "called", "expected", "strike_run_value", "slaa", "framing_runs_aa",
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
        Path(__file__).with_name(f"catcher_pitches_{args.season}.parquet"), index=False
    )
    print(f"{len(result)} catchers")


if __name__ == "__main__":
    main()
