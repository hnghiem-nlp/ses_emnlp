"""Scientific constants used in the published experiments.

Values are intentionally kept aligned with the original research implementation.
Changing them creates a new synthetic-data version and should be documented as such.
"""

from __future__ import annotations

ATTRIBUTE_LABELS = {
    "school_type": "high school type",
    "fee_waiver": "eligible for fee waiver",
    "first_gen": "first-generation student status",
    "income": "household income in US Dollars",
    "income_quintile": "income quintile",
    "activity": "number of extracurricular activities reported",
    "leadership": "number of leadership roles in extracurricular activities",
    "award": "number of awards received in extracurricular activities",
    "parent_marital_status": "parent marital status",
    "zip_code": "zip code",
    "zip_quintile": "zip quintile",
    "parent_education": "highest level of education completed by parent(s)",
}

# This full order mirrors the original Applicant dataclass. Missing and latent
# attributes still participate in seeded shuffling before being omitted.
CANONICAL_ATTRIBUTE_ORDER = (
    "race",
    "school_type",
    "gpa",
    "sat",
    "activity",
    "leadership",
    "award",
    "income",
    "income_quintile",
    "parent_education",
    "zip_code",
    "zip_quintile",
    "first_gen",
    "fee_waiver",
    "parent_marital_status",
)

DISPLAY_ATTRIBUTES = frozenset(
    {
        "school_type",
        "gpa",
        "sat",
        "activity",
        "leadership",
        "award",
        "zip_code",
        "first_gen",
        "fee_waiver",
    }
)

FROZEN_ACTIVITY_BLOCK = ("activity", "leadership", "award")

SAT_BINS = (
    (400, 999),
    (1000, 1099),
    (1100, 1199),
    (1200, 1299),
    (1300, 1399),
    (1400, 1499),
    (1500, 1600),
)
SAT_PROBS = (0.07, 0.11, 0.16, 0.18, 0.18, 0.17, 0.13)

GPA_BINS = (
    (1.0, 2.39),
    (2.4, 2.79),
    (2.8, 3.19),
    (3.2, 3.59),
    (3.6, 4.0),
    (4.0, 4.39),
    (4.4, 5.0),
)
GPA_PROBS = (0.02, 0.05, 0.12, 0.25, 0.37, 0.14, 0.05)

INCOME_QUINTILE_DATA = (
    (0, 16120, 30000),
    (30001, 43850, 58020),
    (58021, 74730, 94000),
    (94001, 119900, 153000),
    (153001, 277300, 500000),
)

SAT_INCOME_QUINTILE_DATA = (
    (0, -1, 51591),
    (51592, -1, 67083),
    (67084, -1, 83766),
    (83767, -1, 110244),
    (110245, -1, 500000),
)

SAT_QUINTILE_DATA = (
    (914, 5),
    (965, 5),
    (1007, 5),
    (1059, 5),
    (1161, 5),
)

PERFORMANCE_SCORE_WEIGHTS = {
    "sat": 0.35,
    "gpa": 0.35,
    "activity": 0.20,
    "leadership": 0.10,
    "award": 0.10,
}

FIRST_GEN_LABELS = {0: "No", 1: "Yes"}
FEE_WAIVER_LABELS = {0: "No", 1: "Yes"}
SCHOOL_TYPE_LABELS = {0: "Public", 1: "Private"}

TIER_DESCRIPTIONS = {
    "t1": ("highly selective", "less than 15%"),
    "t2": ("selective", "between 15% and 30%"),
    "t3": ("moderately selective", "between 30% and 50%"),
}

