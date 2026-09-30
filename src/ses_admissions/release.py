"""Build the combined Hugging Face profile artifact from the paper cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .profiles import PROFILE_COLUMNS

DEFAULT_SEEDS = (123, 456, 789)


def _load_object(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(row, dict) for row in value.values()):
        raise ValueError(f"Expected a JSON object of profile records: {path}")
    return value


def build_hf_profiles(
    profiles_dir: str | Path,
    system2_dir: str | Path,
    output_path: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    expected_cohort_size: int | None = 10_000,
    expected_system2_size: int | None = 1_000,
) -> dict[str, Any]:
    """Combine paper cohorts into one JSONL table and return build metadata."""
    profiles_root = Path(profiles_dir)
    system2_root = Path(system2_dir)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")

    total_rows = 0
    total_system2 = 0
    seen_profile_ids: set[str] = set()
    hasher = hashlib.sha256()

    with temporary.open("w", encoding="utf-8") as stream:
        for seed in seeds:
            cohort_path = profiles_root / f"applicant_profile_10000_{seed}.json"
            subset_path = system2_root / f"cot_profile_10000_{seed}.json"
            cohort = _load_object(cohort_path)
            system2 = _load_object(subset_path)

            if expected_cohort_size is not None and len(cohort) != expected_cohort_size:
                raise ValueError(
                    f"Expected {expected_cohort_size} profiles for seed {seed}, got {len(cohort)}"
                )
            if expected_system2_size is not None and len(system2) != expected_system2_size:
                raise ValueError(
                    f"Expected {expected_system2_size} System 2 profiles for seed {seed}, "
                    f"got {len(system2)}"
                )

            missing_ids = set(system2) - set(cohort)
            changed_ids = {
                identifier
                for identifier in system2.keys() & cohort.keys()
                if system2[identifier] != cohort[identifier]
            }
            if missing_ids or changed_ids:
                raise ValueError(
                    f"System 2 data do not exactly match cohort {seed}: "
                    f"{len(missing_ids)} missing IDs, {len(changed_ids)} changed rows"
                )

            system2_ids = set(system2)
            for position, (source_index, profile) in enumerate(cohort.items()):
                missing_columns = set(PROFILE_COLUMNS) - set(profile)
                if missing_columns:
                    raise ValueError(
                        f"Profile {seed}:{source_index} is missing {sorted(missing_columns)}"
                    )
                profile_id = f"{seed}:{source_index}"
                if profile_id in seen_profile_ids:
                    raise ValueError(f"Duplicate profile_id: {profile_id}")
                seen_profile_ids.add(profile_id)

                record = {
                    "profile_id": profile_id,
                    "cohort_seed": seed,
                    "source_index": int(source_index),
                    "cohort_position": position,
                    "system2_subset": source_index in system2_ids,
                }
                record.update({column: profile[column] for column in PROFILE_COLUMNS})
                record["zip_code"] = str(record["zip_code"]).zfill(5)
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                stream.write(line)
                hasher.update(line.encode("utf-8"))
                total_rows += 1
                total_system2 += int(record["system2_subset"])

    temporary.replace(destination)
    return {
        "output": str(destination),
        "rows": total_rows,
        "system2_rows": total_system2,
        "seeds": list(seeds),
        "sha256": hasher.hexdigest(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the combined Hugging Face profile file")
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--system2-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = build_hf_profiles(args.profiles_dir, args.system2_dir, args.output)
    print(json.dumps(metadata, indent=2))
