"""Public reproducibility utilities for the SES admissions study."""

from .profiles import generate_conditional_profiles
from .prompts import build_prompt
from .results import normalize_decision, parse_system2_response

__all__ = [
    "build_prompt",
    "generate_conditional_profiles",
    "normalize_decision",
    "parse_system2_response",
]

__version__ = "0.1.0"

