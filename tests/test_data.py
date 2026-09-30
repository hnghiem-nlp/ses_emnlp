import pytest

from ses_admissions.data import load_profiles


def test_loader_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        load_profiles()
    with pytest.raises(ValueError, match="exactly one"):
        load_profiles(input_path="profiles.json", repo_id="org/data", revision="v1")


def test_hub_loader_requires_revision_before_importing_optional_dependency() -> None:
    with pytest.raises(ValueError, match="revision"):
        load_profiles(repo_id="org/data")
