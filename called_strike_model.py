"""Fit and evaluate a catcher-neutral called-strike probability model."""

import json

import numpy as np
import pandas as pd

TRAJECTORY_FIELDS = [
    "release_speed", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "pfx_x", "pfx_z", "release_spin_rate", "spin_axis",
]
HALF_PLATE_FT = 17 / 24
UMPIRE_FIELD = "home_plate_umpire"


def plate_umpires(schedule_path):
    schedule = json.loads(schedule_path.read_text())
    assignments = {}
    for date in schedule["dates"]:
        for game in date["games"]:
            if game["status"]["detailedState"] not in {"Final", "Completed Early"}:
                continue
            officials = [
                official["official"]["id"] for official in game.get("officials", [])
                if official["officialType"] == "Home Plate"
            ]
            if len(officials) != 1:
                raise ValueError(f"Expected one plate umpire for game {game['gamePk']}")
            previous = assignments.setdefault(game["gamePk"], officials[0])
            if previous != officials[0]:
                raise ValueError(f"Conflicting plate umpire assignments for game {game['gamePk']}")
    return assignments


def approach_angles(pitches, plate_depth=HALF_PLATE_FT):
    """Estimate approach angles at the Statcast measurement plane, in degrees."""
    vx, vy, vz, ax, ay, az = (pitches[field].to_numpy() for field in (
        "vx0", "vy0", "vz0", "ax", "ay", "az",
    ))
    # CSV velocity is evaluated 50 feet from the back point of the plate.
    travel = 50 - plate_depth
    with np.errstate(invalid="ignore", divide="ignore"):
        final_y = -np.sqrt(vy**2 - 2 * ay * travel)
        time = 2 * travel / -(vy + final_y)
        haa = np.degrees(np.arctan2(vx + ax * time, -final_y))
        vaa = np.degrees(np.arctan2(vz + az * time, -final_y))
    valid = (vy < 0) & (final_y < 0) & np.isfinite(time)
    return np.where(valid, haa, np.nan), np.where(valid, vaa, np.nan)


def model_features(pitches, schedule_path, season=2026):
    assignments = plate_umpires(schedule_path)
    umpires = pitches["game_pk"].map(assignments)
    if umpires.isna().any():
        raise ValueError("Some scored pitches have no registered home-plate umpire")
    x = pitches["plate_x"]
    z = pitches["plate_z"]
    bottom = z - pitches["sz_bot"]
    top = pitches["sz_top"] - z
    side = HALF_PLATE_FT - x.abs()
    edge = np.where(
        (side >= 0) & (bottom >= 0) & (top >= 0),
        np.minimum(np.minimum(side, bottom), top),
        -np.hypot(np.minimum(side, 0), np.minimum(np.minimum(bottom, top), 0)),
    )
    plate_depth = HALF_PLATE_FT if season >= 2026 else 2 * HALF_PLATE_FT
    haa, vaa = approach_angles(pitches, plate_depth)
    axis = np.deg2rad(pitches["spin_axis"])
    features = pd.DataFrame({
        "plate_x": x,
        "zone_height": bottom / (bottom + top),
        "vertical_zone_size": bottom + top,
        "from_lower_edge": bottom,
        "from_upper_edge": top,
        "from_side_edge": side,
        "signed_edge_distance": edge,
        "balls": pitches["balls"],
        "strikes": pitches["strikes"],
        "right_batter": pitches["stand"].eq("R"),
        "right_pitcher": pitches["p_throws"].eq("R"),
        "velocity_mph": pitches["release_speed"],
        "horizontal_approach_deg": haa,
        "vertical_approach_deg": vaa,
        "horizontal_movement_in": pitches["pfx_x"] * 12,
        "vertical_movement_in": pitches["pfx_z"] * 12,
        "spin_rpm": pitches["release_spin_rate"],
        "spin_axis_sin": np.sin(axis),
        "spin_axis_cos": np.cos(axis),
        UMPIRE_FIELD: umpires,
    }, index=pitches.index)
    print(f"{umpires.nunique()} plate umpires across {pitches['game_pk'].nunique():,} games", flush=True)
    print("Tracking coverage:", features.notna().mean().tail(9).round(4).to_dict(), flush=True)
    return features


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -25, 25)))


def fit_logistic(matrix, called, penalty, offset=0):
    """Solve a ridge-regularized logistic model with an optional fixed offset."""
    coefficients = np.zeros(matrix.shape[1], dtype=matrix.dtype)
    ridge = np.eye(matrix.shape[1], dtype=matrix.dtype) * penalty
    for _ in range(14):
        probability = sigmoid(matrix @ coefficients + offset)
        gradient = matrix.T @ (called - probability) - ridge @ coefficients
        weights = probability * (1 - probability)
        information = matrix.T @ (matrix * weights[:, None]) + ridge
        step = np.linalg.solve(information, gradient)
        coefficients += step
        if np.max(np.abs(step)) < 0.001:
            break
    return coefficients


def design_matrix(features):
    distance = features["signed_edge_distance"].to_numpy()
    x = features["plate_x"].to_numpy().clip(-2.5, 2.5)
    z = features["zone_height"].to_numpy().clip(-1, 2)
    edges = features[["from_side_edge", "from_lower_edge", "from_upper_edge"]].to_numpy()
    # Blend adjacent zone edges smoothly, including the corners.
    edge_weights = np.exp(-(edges - edges.min(axis=1, keepdims=True)) / 0.12)
    edge_weights /= edge_weights.sum(axis=1, keepdims=True)
    side, low, high = edge_weights.T
    left = side * (x < 0)
    right = side * (x >= 0)
    clock = np.column_stack([left, right, low, high])

    movement = np.column_stack([
        (features["velocity_mph"] - 90) / 8,
        features["horizontal_approach_deg"] / 5,
        (features["vertical_approach_deg"] + 5) / 3,
        features["horizontal_movement_in"] / 15,
        features["vertical_movement_in"] / 20,
        (features["spin_rpm"] - 2200) / 1000,
        features["spin_axis_sin"], features["spin_axis_cos"],
        (features["vertical_zone_size"] - 1.75) / 0.4,
    ])
    # Missing tracking features receive the centered reference value.
    movement = np.nan_to_num(movement, nan=0, posinf=0, neginf=0).clip(-4, 4)
    batter = features["right_batter"].to_numpy(dtype=float) - 0.5
    pitcher = features["right_pitcher"].to_numpy(dtype=float) - 0.5
    count = features["balls"].to_numpy() * 3 + features["strikes"].to_numpy()
    counts = count[:, None] == np.arange(1, 12)

    # Nonlinear geometry and pitch-shape interactions with edge and handedness.
    league = np.column_stack([
        np.ones(len(features)), distance.clip(-1.5, 1), np.tanh(distance * 5), np.tanh(distance * 13),
        x, z, x*x, z*z, x*z, np.abs(distance).clip(0, 1.5), low, high, right,
        clock * distance[:, None].clip(-0.6, 0.6), clock * batter[:, None], clock * pitcher[:, None],
        batter, pitcher, batter * pitcher, counts, movement, movement * movement,
        movement * low[:, None], movement * high[:, None], movement * side[:, None],
        movement * pitcher[:, None], movement * batter[:, None],
    ]).astype(np.float32)
    umpire = np.column_stack([np.ones(len(features)), left, right, low]).astype(np.float32)
    return league, umpire


def fit_probability(train_x, called, test_x):
    league_train, umpire_train = design_matrix(train_x)
    league_test, umpire_test = design_matrix(test_x)
    called = np.asarray(called, dtype=np.float32)
    coefficients = fit_logistic(league_train, called, penalty=8)
    train_logits = league_train @ coefficients
    predict_logits = league_test @ coefficients

    if UMPIRE_FIELD in train_x:
        officials = test_x[UMPIRE_FIELD].to_numpy()
        for official, rows in train_x.groupby(UMPIRE_FIELD, sort=False).indices.items():
            test = officials == official
            if test.any():
                # Shrink individual zones toward the league, especially for sparse umpires.
                adjustment = fit_logistic(
                    umpire_train[rows], called[rows], penalty=115, offset=train_logits[rows],
                )
                predict_logits[test] += umpire_test[test] @ adjustment
    return sigmoid(predict_logits).clip(1e-6, 1 - 1e-6)


def metrics(called, predictions, borderline):
    scores = {}
    for label, mask in (("all", np.full(len(called), True)), ("edge", borderline)):
        actual = called[mask]
        estimate = np.clip(predictions[mask], 1e-6, 1 - 1e-6)
        scores[label] = {
            "pitches": int(mask.sum()),
            "brier": float(np.square(actual - estimate).mean()),
            "log_loss": float(-np.mean(actual * np.log(estimate) + (1 - actual) * np.log1p(-estimate))),
            "bias": float(estimate.mean() - actual.mean()),
        }
    return scores


def chronological_review(takes, features, baseline_predict, season=2026):
    called = takes["called"].to_numpy(dtype=float)
    # Split by game, even if a suspended game has pitches listed on different dates.
    first_date = takes.groupby("game_pk")["game_date"].transform("min")
    if season == 2026:
        test_start = "2026-08-01"
    else:
        game_dates = first_date.groupby(takes["game_pk"]).first().sort_values()
        test_start = game_dates.iloc[int(len(game_dates) * 0.8)]
    test = first_date.ge(test_start).to_numpy()
    baseline = baseline_predict(~test, test)
    alternate = fit_probability(features.loc[~test], called[~test], features.loc[test])
    neutral = features.drop(columns=UMPIRE_FIELD)
    without_umpire = fit_probability(neutral.loc[~test], called[~test], neutral.loc[test])
    edge = features.loc[test, "signed_edge_distance"].abs().le(0.25).to_numpy()
    review = {
        "test_start": test_start,
        "baseline": metrics(called[test], baseline, edge),
        "without_umpire": metrics(called[test], without_umpire, edge),
        "challenger": metrics(called[test], alternate, edge),
    }
    print(json.dumps(review, indent=2), flush=True)
    return review


def cross_fit(takes, features):
    called = takes["called"].to_numpy(dtype=float)
    folds = takes["game_pk"].to_numpy() % 5
    probabilities = np.empty(len(takes))
    for fold in range(5):
        test = folds == fold
        probabilities[test] = fit_probability(features.loc[~test], called[~test], features.loc[test])
        print(f"Scored held-out game fold {fold + 1}/5", flush=True)
    return probabilities
