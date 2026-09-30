# Socioeconomic Reasoning in LLM Admissions Decisions

This directory is the release-ready code package for the paper **“From Data to
Decisions: The Impact of Socioeconomic Factors on Large Language Model-Based
College Admissions.”** It contains the basic scripts needed to regenerate
synthetic applicant profiles, construct the paper's System 1 and System 2
prompts, run the four open-source models, and parse their outputs.

The synthetic profile files will be published separately on the Hugging Face
Hub. The commands here deliberately support both local files and a pinned Hub
dataset revision, so moving to the public dataset will not require code changes.

## What is included

- Conditional synthetic-profile generation with the paper's seeds and sampling
  sequence.
- Exact prompt variants for System 1 (decision only) and System 2 (brief
  explanation followed by a decision).
- Institution selectivity prompts for omitted-range and specified-rate
  conditions.
- Batched Hugging Face inference with the model-specific system-prompt handling
  used in the study.
- Robust, auditable result parsing that retains every raw response.
- Tests covering deterministic generation, prompt construction, file loading,
  and response parsing.

The private analysis notebooks, intermediate artifacts, API integrations, and
cluster-specific launch machinery are intentionally not part of this basic
release.

## Installation

Python 3.10 or newer is required.

```bash
cd public
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Install optional Hugging Face dataset and inference dependencies as needed:

```bash
pip install -e ".[hf,inference]"
```

Four- or eight-bit inference uses `bitsandbytes` and a CUDA GPU. For a small CPU
smoke test, pass `--quantization-bits 0`; this is not the paper's inference
configuration.

## 1. Generate profiles

Profile generation requires the ZIP-code-to-income-quintile mapping. This map
will accompany the synthetic data release.

```bash
ses-generate-profiles \
  --num-students 10000 \
  --seed 123 \
  --zip-map path/to/zip_income_map.json \
  --output artifacts/profiles_123.jsonl
```

The paper uses profile seeds `123`, `456`, and `789`, an oversample pool of
5,000 applicants, and stratified weighted subsampling. These are the command
defaults. Changing them produces a different synthetic cohort.

## 2. Preview prompts

Previewing is a lightweight way to verify the data and prompt condition before
loading a model:

```bash
ses-preview-prompts \
  --input artifacts/profiles_123.jsonl \
  --institution "Example University" \
  --tier t2 \
  --condition omitted \
  --mode system1 \
  --prompt-seed 123 \
  --attribute-seed 101
```

Once the dataset is on Hugging Face, replace the local source arguments:

```bash
ses-preview-prompts \
  --dataset-id ORG/DATASET \
  --dataset-revision v1.0.0 \
  --dataset-config profiles \
  --split train \
  --institution "Example University" \
  --tier t2 \
  --condition omitted
```

A tag or commit in `--dataset-revision` is required to prevent later Hub
updates from silently changing a run.

## 3. Run inference

Edit [`configs/experiment.json`](configs/experiment.json) before a public run:

1. Replace the placeholder dataset ID.
2. Pin each model to the exact Hugging Face revision used for the release.
3. Confirm the dataset revision/tag after the Hub upload.

Then run a configured model by its key (`gemma`, `llama`, `mistral`, or `qwen`):

```bash
ses-run-inference \
  --config configs/experiment.json \
  --model qwen \
  --institution "Example University" \
  --tier t2 \
  --condition specified \
  --acceptance-rate 24 \
  --mode system2 \
  --prompt-seed 123 \
  --attribute-seed 101 \
  --output artifacts/qwen_system2.jsonl
```

The inference command reads the Hub source from `experiment.json`. Pass
`--input path/to/profiles.jsonl` to override it with a local file, or pass the
`--dataset-*` arguments to override the configured Hub source.

Every result contains its applicant ID, unmodified model response, parsed
decision, parse status, and run metadata. Reparse an output file after improving
the parser without rerunning the model:

```bash
ses-parse-results \
  --input artifacts/qwen_system2.jsonl \
  --output artifacts/qwen_system2_reparsed.jsonl \
  --mode system2
```

## Expected Hugging Face dataset layout

The scripts expect a dataset configuration named `profiles` with a `train`
split by default. Each row should have a stable string `id` and these profile
columns:

```text
income_quintile, household_income, gpa, sat, activity, leadership,
award, first_gen, fee_waiver, school_type, zip_code, zip_quintile,
ses_quintile, perf_quintile
```

`id` is the join key between profiles and outputs. ZIP codes must be stored as
strings so leading zeros are preserved.

## Reproducibility notes

- Generation and attribute-order seeds are recorded in
  [`configs/experiment.json`](configs/experiment.json).
- Greedy decoding uses `do_sample=false` and `max_new_tokens=512`.
- Gemma and Mistral receive the system instruction merged into the user message,
  matching the study implementation; Llama and Qwen use separate chat roles.
- The generator intentionally preserves the original random-number consumption
  and scientific constants. Refactoring those values creates a new data version.
- Model and dataset revisions must be filled in before release. Moving Hub tags
  are convenient for readers, but immutable commit hashes are best for archival
  reproduction.

## Responsible use

These are synthetic applicants used to audit model behavior. The generated
decisions are not valid admissions recommendations and must not be used to make
decisions about real people. The study demonstrates that model outputs can
encode socioeconomic biases even when explicit income fields are absent.

## License

The code in this repository is released under the
[Apache License 2.0](LICENSE).

## Before publishing

The maintainers still need to add the final Hugging Face dataset ID and immutable
revisions, add the final citation metadata, and replace “Example University” in
run commands with the institutions specified by the released experiment manifest.
