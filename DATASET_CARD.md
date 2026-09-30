---
pretty_name: Synthetic Socioeconomic College Admissions Profiles
license: apache-2.0
language:
  - en
tags:
  - synthetic
  - socioeconomic-status
  - college-admissions
  - llm-bias
  - responsible-ai
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/profiles.jsonl
---

# Synthetic Socioeconomic College Admissions Profiles

This dataset contains the 30,000 synthetic applicant profiles used in
**“From Data to Decisions: The Impact of Socioeconomic Factors on Large
Language Model-Based College Admissions.”** The profiles support audits of how
language models use socioeconomic signals when making simulated admissions
decisions.

The records describe synthetic applicants, not real people. They must not be
used to make admissions or other consequential decisions about individuals.

## Loading

```python
from datasets import load_dataset

profiles = load_dataset(
    "nghiemhnlp/ses_emnlp",
    split="train",
    revision="v1.0.0",
)
```

All three cohorts are combined in one split. Filter using `cohort_seed`:

```python
cohort_123 = profiles.filter(lambda row: row["cohort_seed"] == 123)
```

The profiles used in the System 2 experiments are marked directly:

```python
system2_profiles = profiles.filter(lambda row: row["system2_subset"])
```

## Dataset structure

There are 30,000 rows: 10,000 profiles for each generation seed (`123`, `456`,
and `789`). Exactly 3,000 rows—1,000 per cohort—have
`system2_subset = true`.

### Identifiers and experiment metadata

| Field | Type | Description |
|---|---|---|
| `profile_id` | string | Globally unique ID formatted as `cohort_seed:source_index`. |
| `cohort_seed` | integer | Profile-generation seed: `123`, `456`, or `789`. |
| `source_index` | integer | Original row identifier retained for joining legacy outputs. |
| `cohort_position` | integer | Row position within the original 10,000-profile cohort. |
| `system2_subset` | boolean | Whether the profile belongs to the paper's System 2 subset. |

### Applicant attributes

| Field | Type | Description |
|---|---|---|
| `income_quintile` | integer | Generated household-income quintile, 1–5. |
| `household_income` | integer | Generated annual household income in USD. |
| `gpa` | float | Generated GPA. |
| `sat` | integer | Generated SAT score. |
| `activity` | integer | Number of reported extracurricular activities. |
| `leadership` | integer | Number of activities involving leadership. |
| `award` | integer | Number of activities involving awards or honors. |
| `first_gen` | string | Synthetic first-generation status (`Yes` or `No`). |
| `fee_waiver` | string | Synthetic fee-waiver eligibility (`Yes` or `No`). |
| `school_type` | string | Synthetic high-school type (`Public` or `Private`). |
| `zip_code` | string | Sampled U.S. ZIP code, stored as a five-character string. |
| `zip_quintile` | integer | Income quintile associated with the sampled ZIP code. |
| `ses_quintile` | integer | Composite socioeconomic-status quintile, 1–5. |
| `perf_quintile` | integer | Composite academic-performance quintile, 1–5. |

The admissions prompts displayed `school_type`, `gpa`, `sat`, `activity`,
`leadership`, `award`, `first_gen`, `fee_waiver`, and `zip_code`. Income and
composite quintile fields were retained for analysis but were not shown to the
models.

## Construction

For each seed, 15,000 candidate profiles were generated from distributions and
dependencies grounded in public reports. A weighted stratified sample of 10,000
was then selected to improve coverage across the 5×5 SES–performance grid.

The System 2 subset was selected independently within each published cohort
using sampling seed `42`. The `system2_subset` flag was verified against the
original System 2 files, including exact equality of all applicant attributes.

The ZIP codes are real geographic codes grouped using public American Community
Survey income data, but their association with applicant attributes is entirely
synthetic.

## Validation

- 30,000 total rows and no missing attribute values.
- 10,000 rows per cohort seed.
- 3,000 System 2 rows, with 1,000 from each cohort.
- Globally unique `profile_id` values.
- ZIP codes retained as strings, including leading zeros.
- No duplicate complete profiles.

## Intended use

The dataset is intended for research on LLM behavior, socioeconomic reasoning,
fairness, robustness, and simulated high-stakes decision making. It is not a
model of any particular institution's admissions process and is not suitable
for evaluating or ranking real applicants.

## Limitations

The profiles contain a limited set of structured attributes and cannot capture
the complexity of real applications. Dependencies are literature-grounded but
synthetic. The data should not be interpreted as representative of every U.S.
applicant or institution. ZIP codes can expose models to geographic priors even
though no record corresponds to a real person.

## Code and citation

Generation, prompt, and inference scripts are available at
[hnghiem-nlp/ses_emnlp](https://github.com/hnghiem-nlp/ses_emnlp).

Please cite the associated EMNLP 2025 paper when using this dataset.
