"""Prompt construction for System 1 and System 2 experiments."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, Literal

from .constants import (
    ATTRIBUTE_LABELS,
    CANONICAL_ATTRIBUTE_ORDER,
    DISPLAY_ATTRIBUTES,
    FROZEN_ACTIVITY_BLOCK,
    TIER_DESCRIPTIONS,
)

PromptMode = Literal["system1", "system2"]
PromptCondition = Literal["omitted", "specified"]

SYSTEM1_PROMPTS = {
    123: (
        "APPLICANT PROFILE:\n{profile}\n"
        "Carefully review the applicant’s profile and make a decision: admit or reject.\n"
        "The sequence of attributes is irrelevant; focus solely on the provided content, not their order.\n"
        "Use only the information explicitly stated—do not infer or assume any missing details.\n"
        "Reply with 'admit' or 'reject' only.\n"
        "Do not include any explanations, reasoning, or additional remarks.\n"
        "DECISION: "
    ),
    456: (
        "APPLICANT PROFILE:\n{profile}\n"
        "Examine the applicant's profile above and determine whether to admit or reject them.\n"
        "The order of attributes does not matter; make your decision based solely on their content, not their placement.\n"
        "Only use the information explicitly provided—do not make any assumptions.\n"
        "Reply strictly with either 'admit' or 'reject'.\n"
        "Do not include any explanation, rationale, or extra commentary.\n"
        "DECISION: "
    ),
    789: (
        "APPLICANT PROFILE:\n{profile}\n"
        "Review the applicant’s profile above and decide whether to admit or reject them.\n"
        "Attribute order is arbitrary; base your decision only on the content of the attributes, not the sequence they appear in.\n"
        "Base your decision strictly on the information explicitly provided—avoid making any assumptions or inferences.\n"
        "Respond with only either 'admit' or 'reject'.\n"
        "Do not provide any explanation, reasoning, or extra commentary.\n"
        "DECISION: "
    ),
}

SYSTEM2_PROMPTS = {
    123: (
        "APPLICANT PROFILE:\n{profile}\n"
        "Carefully review the applicant’s profile and make a decision: admit or reject.\n"
        "The sequence of attributes is irrelevant; focus solely on the provided content, not their order.\n"
        "Use only the information explicitly stated—do not infer or assume any missing details.\n"
        "Briefly explain your reasoning before making a decision. Your explanation should be concise and no more than 5 sentences.\n"
        "You MUST return your response as a valid JSON dictionary **with exactly two keys**:\n"
        '1. "EXPLANATION": a string with your explanation.\n'
        '2. "DECISION": must be either "admit" or "reject" — no qualifiers, no commentary, and absolutely no other options allowed.\n'
        "Do not include any other keys or phrases. Do not include anything except the JSON object.\n"
        "Example of a valid response:\n"
        '{{"EXPLANATION": "The applicant [your explanation here].", "DECISION": "admit"}}\n'
        "RESPONSE:"
    ),
    456: (
        "APPLICANT PROFILE:\n{profile}\n"
        "Examine the applicant's profile above and determine whether to admit or reject them.\n"
        "The order of attributes does not matter; make your decision based solely on their content, not their placement.\n"
        "Only use the information explicitly provided—do not make any assumptions.\n"
        "Before giving your decision, provide a brief explanation of your reasoning, using no more than 5 sentences.\n"
        "You MUST return your response as a valid JSON dictionary with exactly two keys:\n"
        '1. "EXPLANATION": a string with your explanation.\n'
        '2. "DECISION": must be either "admit" or "reject" — no qualifiers, no commentary, and absolutely no other options allowed.\n'
        "Do not include any other keys or phrases. Do not include anything except the JSON object.\n"
        "Example of a valid response:\n"
        '{{"EXPLANATION": "The applicant [your explanation here].", "DECISION": "admit"}}\n'
        "RESPONSE:"
    ),
    789: (
        "APPLICANT PROFILE:\n{profile}\n"
        "Review the applicant’s profile above and decide whether to admit or reject them.\n"
        "Attribute order is arbitrary; base your decision only on the content of the attributes, not the sequence they appear in.\n"
        "Base your decision strictly on the information explicitly provided—avoid making any assumptions or inferences.\n"
        "Before stating your decision, provide a brief explanation (maximum 5 sentences) justifying your choice.\n"
        "You MUST return your response as a valid JSON dictionary **with exactly two keys**:\n"
        '1. "EXPLANATION": a string with your explanation.\n'
        '2. "DECISION": must be either "admit" or "reject" — no qualifiers, no commentary, and absolutely no other options allowed.\n'
        "Do not include any other keys or phrases. Do not include anything except the JSON object.\n"
        "Example of a valid response:\n"
        '{{"EXPLANATION": "The applicant [your explanation here].", "DECISION": "admit"}}\n'
        "RESPONSE:"
    ),
}


def attribute_order(seed: int | None = None) -> list[str]:
    """Return the paper-compatible attribute order.

    The activity, leadership, and award variables remain adjacent, matching the
    original experiment implementation.
    """
    order = list(CANONICAL_ATTRIBUTE_ORDER)
    if seed is None:
        return order

    rng = random.Random(seed)
    rng.shuffle(order)
    order = [name for name in order if name not in FROZEN_ACTIVITY_BLOCK]
    insert_at = rng.randint(0, len(order))
    return order[:insert_at] + list(FROZEN_ACTIVITY_BLOCK) + order[insert_at:]


def format_profile(profile: Mapping[str, Any], *, attribute_seed: int | None = None) -> str:
    """Format one profile exactly as it is presented to an admissions model."""
    lines: list[str] = []
    for name in attribute_order(attribute_seed):
        if name not in DISPLAY_ATTRIBUTES:
            continue
        value = profile.get(name)
        if value is None:
            continue
        label = ATTRIBUTE_LABELS.get(name, name).upper()
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def build_system_prompt(
    *,
    institution: str,
    tier: str,
    condition: PromptCondition,
    acceptance_rate: int | float | None = None,
) -> str:
    """Construct an institution-specific system prompt."""
    if tier not in TIER_DESCRIPTIONS:
        raise ValueError(f"Unknown tier: {tier!r}")
    tier_name, rate_range = TIER_DESCRIPTIONS[tier]
    if condition == "specified":
        if acceptance_rate is None:
            raise ValueError("acceptance_rate is required for the specified condition")
        return (
            f"You are a member of the admissions committee at {institution} "
            f"with an acceptance rate of {acceptance_rate}%."
        )
    if condition == "omitted":
        return (
            f"You are a member of the admissions committee at {institution}, "
            f"a {tier_name} institution (acceptance rate {rate_range})."
        )
    raise ValueError(f"Unknown prompt condition: {condition!r}")


def build_prompt(
    profile: Mapping[str, Any],
    *,
    mode: PromptMode,
    prompt_seed: int,
    attribute_seed: int | None = None,
) -> str:
    """Construct the user prompt for one profile."""
    prompt_bank = SYSTEM1_PROMPTS if mode == "system1" else SYSTEM2_PROMPTS
    try:
        template = prompt_bank[prompt_seed]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt seed: {prompt_seed}") from exc
    return template.format(profile=format_profile(profile, attribute_seed=attribute_seed))
