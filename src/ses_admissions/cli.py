"""Command-line entry points for the public reproducibility package."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .data import load_profiles
from .inference import HFInferenceRunner, attach_responses, prepare_records
from .profiles import generate_conditional_profiles, load_zip_map, save_profiles, validate_profiles
from .prompts import build_prompt, build_system_prompt
from .results import parse_response, read_jsonl, write_jsonl


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _add_source_arguments(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    source = parser.add_mutually_exclusive_group(required=required)
    source.add_argument("--input", type=Path, help="Local JSON/JSONL/CSV/Parquet profiles")
    source.add_argument("--dataset-id", help="Hugging Face dataset repository ID")
    parser.add_argument("--dataset-revision", help="Required tag or commit for Hub data")
    parser.add_argument("--dataset-config", default="profiles")
    parser.add_argument("--split", default="train")


def _load_from_args(args: argparse.Namespace):
    return load_profiles(
        input_path=args.input,
        repo_id=args.dataset_id,
        revision=args.dataset_revision,
        config_name=args.dataset_config,
        split=args.split,
    )


def generate_profiles_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate paper-compatible synthetic profiles")
    parser.add_argument("--num-students", type=int, default=10_000)
    parser.add_argument("--seed", type=int, choices=(123, 456, 789), default=123)
    parser.add_argument("--zip-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oversample", type=int, default=5_000)
    parser.add_argument(
        "--no-subsample",
        action="store_true",
        help="Keep the full num-students + oversample candidate pool",
    )
    args = parser.parse_args(argv)
    profiles = generate_conditional_profiles(
        args.num_students,
        args.seed,
        load_zip_map(args.zip_map),
        oversample=args.oversample,
        subsample=not args.no_subsample,
    )
    save_profiles(profiles, args.output)
    print(f"Wrote {len(profiles):,} profiles to {args.output}")


def preview_prompts_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render prompts without loading a model")
    _add_source_arguments(parser)
    parser.add_argument("--mode", choices=("system1", "system2"), default="system1")
    parser.add_argument("--prompt-seed", type=int, choices=(123, 456, 789), default=123)
    parser.add_argument("--attribute-seed", type=int, choices=(101, 211, 307), default=101)
    parser.add_argument("--institution", required=True)
    parser.add_argument("--tier", choices=("t1", "t2", "t3"), required=True)
    parser.add_argument("--condition", choices=("omitted", "specified"), required=True)
    parser.add_argument("--acceptance-rate", type=float)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args(argv)

    frame = _load_from_args(args).head(args.limit)
    system_prompt = build_system_prompt(
        institution=args.institution,
        tier=args.tier,
        condition=args.condition,
        acceptance_rate=args.acceptance_rate,
    )
    print(f"SYSTEM\n{system_prompt}")
    for row in frame.to_dict(orient="records"):
        print(
            "\nUSER "
            f"[id={row['id']}]\n"
            + build_prompt(
                row,
                mode=args.mode,
                prompt_seed=args.prompt_seed,
                attribute_seed=args.attribute_seed,
            )
        )


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Experiment config must be a JSON object")
    return value


def run_inference_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a paper-compatible local HF model")
    _add_source_arguments(parser, required=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Model key from experiment.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("system1", "system2"), default="system1")
    parser.add_argument("--prompt-seed", type=int, choices=(123, 456, 789), default=123)
    parser.add_argument("--attribute-seed", type=int, choices=(101, 211, 307), default=101)
    parser.add_argument("--institution", required=True)
    parser.add_argument("--tier", choices=("t1", "t2", "t3"), required=True)
    parser.add_argument("--condition", choices=("omitted", "specified"), required=True)
    parser.add_argument("--acceptance-rate", type=float)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--quantization-bits", type=int, choices=(0, 4, 8))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    config = _load_config(args.config)
    try:
        model_config = config["models"][args.model]
        generation_config = config["generation"]
    except KeyError as exc:
        raise ValueError(f"Missing config entry: {exc}") from exc

    if args.input is None and args.dataset_id is None:
        dataset_config = config.get("dataset", {})
        args.dataset_id = dataset_config.get("repo_id")
        args.dataset_revision = dataset_config.get("revision")
        args.dataset_config = dataset_config.get("profiles_config", "profiles")
        args.split = dataset_config.get("profiles_split", "train")
    if args.input is None and not args.dataset_id:
        raise ValueError("Configure a dataset repo_id or pass --input/--dataset-id")

    frame = _load_from_args(args)
    if args.limit is not None:
        frame = frame.head(args.limit)
    profiles = frame.to_dict(orient="records")
    prompts, records = prepare_records(
        profiles,
        mode=args.mode,
        prompt_seed=args.prompt_seed,
        attribute_seed=args.attribute_seed,
    )
    system_prompt = build_system_prompt(
        institution=args.institution,
        tier=args.tier,
        condition=args.condition,
        acceptance_rate=args.acceptance_rate,
    )
    quantization_bits = (
        args.quantization_bits
        if args.quantization_bits is not None
        else generation_config.get("quantization_bits", 4)
    )
    runner = HFInferenceRunner(
        model_config["model_id"],
        revision=model_config.get("revision"),
        merge_system_prompt=model_config.get("merge_system_prompt", False),
        quantization_bits=quantization_bits,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    responses = runner.generate(
        prompts,
        system_prompt=system_prompt,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=generation_config.get("max_new_tokens", 512),
        seed=config.get("seeds", {}).get("system2_sampling", 123),
    )
    outputs = attach_responses(records, responses, mode=args.mode)
    metadata = {
        "model_id": model_config["model_id"],
        "model_revision": model_config.get("revision"),
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "dataset_config": args.dataset_config,
        "dataset_split": args.split,
        "input_path": str(args.input) if args.input else None,
        "mode": args.mode,
        "prompt_seed": args.prompt_seed,
        "attribute_seed": args.attribute_seed,
        "institution": args.institution,
        "tier": args.tier,
        "condition": args.condition,
        "acceptance_rate": args.acceptance_rate,
        "quantization_bits": quantization_bits,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": generation_config.get("max_new_tokens", 512),
    }
    for output in outputs:
        output["metadata"] = metadata
    write_jsonl(outputs, args.output)
    print(f"Wrote {len(outputs):,} results to {args.output}")


def parse_results_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parse raw responses in a JSONL result file")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("system1", "system2"), required=True)
    args = parser.parse_args(argv)
    parsed: list[dict[str, Any]] = []
    for record in read_jsonl(args.input):
        item = dict(record)
        item.update(parse_response(item.get("raw_response"), mode=args.mode))
        parsed.append(item)
    write_jsonl(parsed, args.output)
    print(f"Wrote {len(parsed):,} parsed results to {args.output}")


def validate_profiles_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a local synthetic profile table")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    errors = validate_profiles(load_profiles(input_path=args.input).drop(columns=["id"]))
    if errors:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(errors))
    print(f"Validated {args.input}")
