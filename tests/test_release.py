import json

from ses_admissions.profiles import PROFILE_COLUMNS
from ses_admissions.release import build_hf_profiles


def _profile(zip_code: str) -> dict:
    values = {
        "income_quintile": 1,
        "household_income": 20_000,
        "gpa": 3.5,
        "sat": 1200,
        "activity": 3,
        "leadership": 1,
        "award": 0,
        "first_gen": "Yes",
        "fee_waiver": "Yes",
        "school_type": "Public",
        "zip_code": zip_code,
        "zip_quintile": 1,
        "ses_quintile": 1,
        "perf_quintile": 3,
    }
    assert set(values) == set(PROFILE_COLUMNS)
    return values


def test_build_hf_profiles(tmp_path) -> None:
    profiles_dir = tmp_path / "profiles"
    system2_dir = tmp_path / "system2"
    profiles_dir.mkdir()
    system2_dir.mkdir()
    cohort = {"9": _profile("601"), "4": _profile("02138")}
    subset = {"4": cohort["4"]}
    (profiles_dir / "applicant_profile_10000_123.json").write_text(json.dumps(cohort))
    (system2_dir / "cot_profile_10000_123.json").write_text(json.dumps(subset))

    output = tmp_path / "profiles.jsonl"
    metadata = build_hf_profiles(
        profiles_dir,
        system2_dir,
        output,
        seeds=(123,),
        expected_cohort_size=2,
        expected_system2_size=1,
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]

    assert metadata["rows"] == 2
    assert metadata["system2_rows"] == 1
    assert [record["profile_id"] for record in records] == ["123:9", "123:4"]
    assert [record["system2_subset"] for record in records] == [False, True]
    assert records[0]["zip_code"] == "00601"
