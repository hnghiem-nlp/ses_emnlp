"""Profile loading with interchangeable local-file and Hugging Face backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_local_profiles(path: str | Path) -> pd.DataFrame:
    """Load profiles from the formats produced by :func:`save_profiles`.

    Both the original nested-JSON layout (``{"0": {...}}``) and record-oriented
    JSON/JSONL files are accepted. An ``id`` column is always returned.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            frame = pd.DataFrame.from_dict(payload, orient="index")
            frame.index.name = "id"
            frame = frame.reset_index()
        elif isinstance(payload, list):
            frame = pd.DataFrame(payload)
        else:
            raise ValueError("JSON profiles must be an object or a list of objects")
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(source, orient="records", lines=True, dtype={"zip_code": str})
    elif suffix == ".csv":
        frame = pd.read_csv(source, dtype={"zip_code": str})
    elif suffix == ".parquet":
        frame = pd.read_parquet(source)
    else:
        raise ValueError("Input must end in .json, .jsonl, .ndjson, .csv, or .parquet")

    if "id" not in frame.columns:
        frame.insert(0, "id", frame.index.astype(str))
    frame["id"] = frame["id"].astype(str)
    if "zip_code" in frame:
        frame["zip_code"] = frame["zip_code"].astype(str).str.zfill(5)
    return frame


def load_hf_profiles(
    repo_id: str,
    *,
    revision: str,
    config_name: str = "profiles",
    split: str = "train",
) -> pd.DataFrame:
    """Load a pinned profile split from the Hugging Face Hub.

    Requiring a revision prevents a future dataset update from silently changing
    an experiment. The revision may be a tag (for example ``v1.0.0``) or commit.
    """
    if not revision:
        raise ValueError("A pinned Hugging Face dataset revision is required")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError('Install Hugging Face support with: pip install -e ".[hf]"') from exc

    dataset = load_dataset(repo_id, config_name, split=split, revision=revision)
    frame = dataset.to_pandas()
    if "id" not in frame.columns:
        frame.insert(0, "id", frame.index.astype(str))
    frame["id"] = frame["id"].astype(str)
    if "zip_code" in frame:
        frame["zip_code"] = frame["zip_code"].astype(str).str.zfill(5)
    return frame


def load_profiles(
    *,
    input_path: str | Path | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    config_name: str = "profiles",
    split: str = "train",
) -> pd.DataFrame:
    """Load profiles from exactly one source using a common table schema."""
    if (input_path is None) == (repo_id is None):
        raise ValueError("Provide exactly one of input_path or repo_id")
    if input_path is not None:
        return load_local_profiles(input_path)
    return load_hf_profiles(
        str(repo_id), revision=str(revision or ""), config_name=config_name, split=split
    )
