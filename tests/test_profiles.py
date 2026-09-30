import pandas as pd

from ses_admissions.data import load_local_profiles
from ses_admissions.profiles import (
    generate_conditional_profiles,
    save_profiles,
    validate_profiles,
)


ZIP_MAP = {str(quintile): [f"{quintile}0001", f"{quintile}0002"] for quintile in range(1, 6)}


def test_profile_generation_is_deterministic() -> None:
    first = generate_conditional_profiles(100, 123, ZIP_MAP, oversample=100)
    second = generate_conditional_profiles(100, 123, ZIP_MAP, oversample=100)
    pd.testing.assert_frame_equal(first, second)
    assert not validate_profiles(first)


def test_jsonl_round_trip_preserves_identifiers_and_zip_codes(tmp_path) -> None:
    profiles = generate_conditional_profiles(100, 456, ZIP_MAP, oversample=100)
    destination = tmp_path / "profiles.jsonl"
    save_profiles(profiles, destination)
    loaded = load_local_profiles(destination)
    assert len(loaded) == 100
    assert loaded["id"].iloc[0] == str(profiles.index[0])
    assert loaded["zip_code"].str.len().eq(5).all()


def test_nested_json_layout_is_loadable(tmp_path) -> None:
    profiles = generate_conditional_profiles(50, 789, ZIP_MAP, oversample=100)
    destination = tmp_path / "profiles.json"
    save_profiles(profiles, destination)
    loaded = load_local_profiles(destination)
    assert set(profiles.columns).issubset(loaded.columns)
    assert len(loaded) == 50
