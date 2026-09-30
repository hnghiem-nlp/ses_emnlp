"""Synthetic applicant generation used by the published study.

The implementation preserves the original sampling sequence and scientific
constants while exposing testable functions and explicit inputs/outputs.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .constants import (
    FEE_WAIVER_LABELS,
    FIRST_GEN_LABELS,
    GPA_BINS,
    GPA_PROBS,
    INCOME_QUINTILE_DATA,
    PERFORMANCE_SCORE_WEIGHTS,
    SAT_INCOME_QUINTILE_DATA,
    SAT_QUINTILE_DATA,
    SCHOOL_TYPE_LABELS,
)

PROFILE_COLUMNS = (
    "income_quintile",
    "household_income",
    "gpa",
    "sat",
    "activity",
    "leadership",
    "award",
    "first_gen",
    "fee_waiver",
    "school_type",
    "zip_code",
    "zip_quintile",
    "ses_quintile",
    "perf_quintile",
)


def _generate_attribute_by_bin(
    n: int,
    bins: Sequence[tuple[float, float]],
    probabilities: Sequence[float],
    *,
    data_type: str,
    round_to: int = 2,
) -> list[int] | list[float]:
    sampled_bins = np.random.choice(len(bins), size=n, p=probabilities)
    if data_type == "int":
        return [int(np.random.randint(bins[i][0], bins[i][1] + 1)) for i in sampled_bins]
    if data_type == "float":
        values = [np.random.uniform(bins[i][0], bins[i][1]) for i in sampled_bins]
        return np.round(values, round_to).tolist()
    raise ValueError("data_type must be 'int' or 'float'")


def _generate_income(quintiles: Sequence[int], cap: int = 500_000) -> list[int]:
    incomes: list[int] = []
    for quintile in quintiles:
        if not 1 <= quintile <= 5:
            raise ValueError("Income quintiles must be between 1 and 5")
        low, mean, high = INCOME_QUINTILE_DATA[quintile - 1]
        low, mean, high = min(low, cap), min(mean, cap), min(high, cap)
        income = low if low == high else random.triangular(low, high, mean)
        incomes.append(int(round(income, -1)))
    return incomes


def _lookup_income_quintile(incomes: Sequence[int]) -> list[int]:
    result: list[int] = []
    for income in incomes:
        matched = -1
        for index, (low, _, high) in enumerate(SAT_INCOME_QUINTILE_DATA):
            if low <= income <= high:
                matched = index + 1
                break
        result.append(matched)
    return result


def _generate_sat(
    quintiles: Sequence[int],
    household_income: Sequence[int],
    *,
    target_correlation: float = 0.4,
    noise_factor: float = 0.25,
) -> list[int]:
    base_scores = np.array(
        [np.random.normal(*SAT_QUINTILE_DATA[quintile - 1]) for quintile in quintiles]
    )
    # The 3000 upper bound and final 1600 clipping preserve the original implementation.
    random_scores = np.random.uniform(400, 3000, size=len(quintiles))
    current_correlation = pearsonr(base_scores, household_income).statistic
    if np.isnan(current_correlation):
        current_correlation = 0.0
    alpha = target_correlation / current_correlation if current_correlation else 0.0
    alpha = min(1.0, max(0.0, alpha) + noise_factor)
    scores = alpha * base_scores + (1 - alpha) * random_scores
    return np.round(np.clip(scores, 400, 1600)).astype(int).tolist()


def _generate_gpa(
    n: int,
    income_quintiles: Sequence[int],
    *,
    target_correlation: float = 0.15,
) -> np.ndarray:
    preliminary = _generate_attribute_by_bin(
        n, GPA_BINS, GPA_PROBS, data_type="float"
    )
    income = np.asarray(income_quintiles)
    income_standardized = (income - income.mean()) / income.std()
    noise = np.random.normal(0, 1, size=n)
    noise = (noise - noise.mean()) / noise.std()
    correlated = (
        target_correlation * income_standardized
        + np.sqrt(1 - target_correlation**2) * noise
    )
    ranks = np.argsort(np.argsort(correlated))
    return np.sort(preliminary)[ranks]


def _generate_school_type(income_quintiles: Sequence[int]) -> np.ndarray:
    private_probabilities = np.array([0.13, 0.16, 0.20, 0.25, 0.35])
    indices = np.asarray(income_quintiles) - 1
    return np.random.binomial(1, private_probabilities[indices])


def _generate_activities(
    income_quintiles: Sequence[int], school_type: Sequence[int]
) -> np.ndarray:
    # This fixed seed is part of the original experiment.
    np.random.seed(42)
    activities = np.zeros(len(income_quintiles), dtype=int)
    categories = [0, 1, 2, 3]
    probabilities = [0.015, 0.23, 0.39, 0.365]
    boosts = {1: -1.2, 2: -0.6, 3: 0, 4: 0.6, 5: 1.2}
    for index, quintile in enumerate(income_quintiles):
        category = np.random.choice(categories, p=probabilities)
        if category == 0:
            count = 0
        elif category == 1:
            count = np.random.randint(1, 4)
        elif category == 2:
            count = np.random.randint(4, 8)
        else:
            count = 10 if np.random.rand() < 0.34 else np.random.randint(8, 10)
        count += boosts[quintile]
        if school_type[index] == 1:
            count *= 1.187
        activities[index] = int(round(np.clip(count, 0, 10)))
    return activities


def _generate_activity_outcomes(
    activities: Sequence[int],
    income_quintiles: Sequence[int],
    school_type: Sequence[int],
    *,
    base_probability: float,
    income_boost: np.ndarray,
    private_boost: float,
    maximum_probability: float,
) -> np.ndarray:
    result = np.zeros(len(activities), dtype=int)
    for index, activity_count in enumerate(activities):
        probability = base_probability + income_boost[income_quintiles[index] - 1]
        if school_type[index] == 1:
            probability += private_boost
        result[index] = np.random.binomial(
            n=activity_count,
            p=np.clip(probability, 0, maximum_probability),
        )
    return result


def _generate_fee_waiver(
    household_incomes: Sequence[int], *, flip_rate: float = 0.3
) -> list[int]:
    thresholds = {1: 25142, 2: 33874, 3: 42606, 4: 51338, 5: 60070, 6: 68802, 7: 77534}
    sizes = np.random.choice(
        [1, 2, 3, 4, 5, 6, 7],
        size=len(household_incomes),
        p=[0.15, 0.30, 0.30, 0.15, 0.05, 0.03, 0.02],
    )
    waivers = [
        int(income <= thresholds[size])
        for income, size in zip(household_incomes, sizes)  # noqa: B905
    ]
    flip_indices = np.random.choice(
        len(waivers), size=int(flip_rate * len(waivers)), replace=False
    )
    for index in flip_indices:
        waivers[index] = 1 - waivers[index]
    return waivers


def _consume_parent_education_draws(sat_scores: Sequence[int]) -> None:
    """Preserve RNG consumption from an unsaved variable in the original pipeline."""
    bins = [(400, 850), (850, 950), (950, 1050), (1050, 1150), (1150, 1600)]
    probabilities = {
        (400, 850): [0.60, 0.25, 0.05, 0.07, 0.03],
        (850, 950): [0.30, 0.40, 0.10, 0.15, 0.05],
        (950, 1050): [0.10, 0.30, 0.20, 0.30, 0.10],
        (1050, 1150): [0.05, 0.15, 0.15, 0.45, 0.20],
        (1150, 1600): [0.02, 0.05, 0.10, 0.30, 0.53],
    }
    for score in sat_scores:
        selected = next((item for item in bins if item[0] <= score < item[1]), None)
        selected = selected or (bins[0] if score < bins[0][0] else bins[-1])
        np.random.choice([0, 1, 2, 3, 4], p=probabilities[selected])


def _generate_first_gen(income_quintiles: Sequence[int], noise_std: float = 0.1) -> list[int]:
    base_probabilities = {1: 0.65, 2: 0.55, 3: 0.40, 4: 0.25, 5: 0.15}
    result: list[int] = []
    for quintile in income_quintiles:
        probability = np.clip(
            np.random.normal(base_probabilities.get(quintile, 0.34), noise_std),
            0.05,
            0.95,
        )
        result.append(int(np.random.rand() < probability))
    return result


def _generate_zip_codes(
    income_quintiles: Sequence[int], zip_map: Mapping[str, Sequence[str]]
) -> tuple[list[str], list[int]]:
    zip_codes: list[str] = []
    zip_quintiles: list[int] = []
    for quintile in income_quintiles:
        if np.random.rand() < 0.5:
            selected_quintile = quintile
        else:
            choices = [candidate for candidate in range(1, 6) if candidate != quintile]
            selected_quintile = int(np.random.choice(choices))
        zip_quintiles.append(selected_quintile)
        zip_codes.append(str(np.random.choice(zip_map[str(selected_quintile)])))
    return zip_codes, zip_quintiles


def _compute_performance_quintile(frame: pd.DataFrame) -> pd.Series:
    weights = PERFORMANCE_SCORE_WEIGHTS
    score = (
        weights["sat"] * frame["sat"].rank(pct=True)
        + weights["gpa"] * frame["gpa"].rank(pct=True)
        + weights["activity"] * ((frame["activity"] - frame["activity"].mean()) / frame["activity"].std())
        + weights["leadership"] * ((frame["leadership"] - frame["leadership"].mean()) / frame["leadership"].std())
        + weights["award"] * ((frame["award"] - frame["award"].mean()) / frame["award"].std())
    )
    return pd.qcut(score, 5, labels=False) + 1


def _compute_ses_quintile(frame: pd.DataFrame) -> pd.Series:
    variables = ("zip_quintile", "fee_waiver", "school_type", "first_gen")
    correlations = np.abs(
        [pearsonr(frame[name], frame["income_quintile"]).statistic for name in variables]
    )
    weights = correlations / (correlations.sum() + 1e-10)
    zip_weight, fee_weight, school_weight, first_gen_weight = np.round(weights, 3)
    score = (
        zip_weight * frame["zip_quintile"].rank(pct=True)
        + school_weight * frame["school_type"].rank(pct=True)
        + fee_weight * (1 - frame["fee_waiver"].rank(pct=True))
        + first_gen_weight * (1 - frame["first_gen"].rank(pct=True))
    )
    return pd.qcut(score, 5, labels=False) + 1


def _subsample_profiles(frame: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    keys = ["ses_quintile", "perf_quintile"]
    counts = frame.groupby(keys).size().to_dict()

    # pandas versions used during the original data generation grouped rows by
    # each join key's first appearance. Reconstruct that order explicitly so
    # both the selected profiles and their IDs are version-independent.
    chunks: list[pd.DataFrame] = []
    for ses_quintile, perf_quintile in frame[keys].drop_duplicates().itertuples(
        index=False, name=None
    ):
        mask = (frame["ses_quintile"] == ses_quintile) & (
            frame["perf_quintile"] == perf_quintile
        )
        chunk = frame.loc[mask].copy()
        chunk["cell_count"] = counts[(ses_quintile, perf_quintile)]
        chunks.append(chunk)
    weighted = pd.concat(chunks, ignore_index=True)
    weighted["weight"] = 1 / weighted["cell_count"]
    weighted["weight"] /= weighted["weight"].sum()
    sampled = weighted.sample(
        n=target_n, weights="weight", random_state=seed, replace=False
    )
    return sampled.drop(columns=["cell_count", "weight"])


def generate_conditional_profiles(
    num_students: int,
    seed: int,
    zip_map: Mapping[str, Sequence[str]],
    *,
    oversample: int = 5_000,
    subsample: bool = True,
) -> pd.DataFrame:
    """Generate one conditionally dependent synthetic applicant cohort."""
    if num_students <= 0:
        raise ValueError("num_students must be positive")
    missing_quintiles = {str(value) for value in range(1, 6)} - set(zip_map)
    if missing_quintiles:
        raise ValueError(f"zip_map is missing quintiles: {sorted(missing_quintiles)}")

    population_size = num_students + oversample
    np.random.seed(seed)
    random.seed(seed)

    income_quintile = [int(np.random.choice([1, 2, 3, 4, 5])) for _ in range(population_size)]
    household_income = _generate_income(income_quintile)
    sat_quintile = _lookup_income_quintile(household_income)
    sat = _generate_sat(sat_quintile, household_income)
    gpa = _generate_gpa(population_size, income_quintile)
    school_type = _generate_school_type(income_quintile)
    activity = _generate_activities(income_quintile, school_type)
    leadership = _generate_activity_outcomes(
        activity,
        income_quintile,
        school_type,
        base_probability=0.15,
        income_boost=np.linspace(-0.08, 0.08, 5),
        private_boost=0.237 / 6.86,
        maximum_probability=0.5,
    )
    award = _generate_activity_outcomes(
        activity,
        income_quintile,
        school_type,
        base_probability=0.22,
        income_boost=np.linspace(-0.10, 0.10, 5),
        private_boost=0.264 / 6.86,
        maximum_probability=0.6,
    )
    fee_waiver = _generate_fee_waiver(household_income)
    _consume_parent_education_draws(sat)
    first_gen = _generate_first_gen(income_quintile)
    zip_code, zip_quintile = _generate_zip_codes(income_quintile, zip_map)

    frame = pd.DataFrame(
        {
            "income_quintile": income_quintile,
            "household_income": household_income,
            "gpa": gpa,
            "sat": sat,
            "activity": activity,
            "leadership": leadership,
            "award": award,
            "first_gen": first_gen,
            "fee_waiver": fee_waiver,
            "school_type": school_type,
            "zip_code": zip_code,
            "zip_quintile": zip_quintile,
        }
    )
    frame["ses_quintile"] = _compute_ses_quintile(frame)
    frame["perf_quintile"] = _compute_performance_quintile(frame)
    if subsample:
        frame = _subsample_profiles(frame, num_students, seed)

    frame = frame.copy()
    frame["first_gen"] = frame["first_gen"].map(FIRST_GEN_LABELS)
    frame["fee_waiver"] = frame["fee_waiver"].map(FEE_WAIVER_LABELS)
    frame["school_type"] = frame["school_type"].map(SCHOOL_TYPE_LABELS)
    frame["zip_code"] = frame["zip_code"].astype(str).str.zfill(5)
    return frame.loc[:, list(PROFILE_COLUMNS)]


def save_profiles(frame: pd.DataFrame, output_path: str | Path) -> None:
    """Write profiles as nested JSON, JSONL, CSV, or Parquet based on the suffix."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        records = {str(index): row for index, row in frame.to_dict(orient="index").items()}
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    elif suffix in {".jsonl", ".ndjson"}:
        frame.reset_index(names="id").to_json(path, orient="records", lines=True)
    elif suffix == ".csv":
        frame.reset_index(names="id").to_csv(path, index=False)
    elif suffix == ".parquet":
        frame.reset_index(names="id").to_parquet(path, index=False)
    else:
        raise ValueError("Output must end in .json, .jsonl, .csv, or .parquet")


def load_zip_map(path: str | Path) -> dict[str, list[str]]:
    """Load the ZIP-code quintile mapping used during profile generation."""
    with Path(path).open(encoding="utf-8") as stream:
        data: dict[str, list[str]] = json.load(stream)
    return data


def validate_profiles(frame: pd.DataFrame) -> list[str]:
    """Return human-readable validation errors for a profile table."""
    errors: list[str] = []
    missing = set(PROFILE_COLUMNS) - set(frame.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
        return errors
    for column in ("income_quintile", "zip_quintile", "ses_quintile", "perf_quintile"):
        if not frame[column].between(1, 5).all():
            errors.append(f"{column} contains values outside 1..5")
    if not frame["gpa"].between(1, 5).all():
        errors.append("gpa contains values outside 1..5")
    if not frame["sat"].between(400, 1600).all():
        errors.append("sat contains values outside 400..1600")
    if not (frame["leadership"] <= frame["activity"]).all():
        errors.append("leadership exceeds activity for at least one profile")
    if not (frame["award"] <= frame["activity"]).all():
        errors.append("award exceeds activity for at least one profile")
    return errors
