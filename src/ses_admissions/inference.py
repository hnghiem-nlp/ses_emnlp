"""Minimal Hugging Face inference runner for the published prompt conditions."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .prompts import build_prompt, build_system_prompt
from .results import parse_response

LOGGER = logging.getLogger(__name__)


def chat_messages(system_prompt: str, user_prompt: str, *, merge_system: bool) -> list[dict[str, str]]:
    """Build chat-template messages, including the paper's model-specific merge."""
    if merge_system:
        return [{"role": "user", "content": f"{system_prompt} {user_prompt}"}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class HFInferenceRunner:
    """Load one causal language model and generate deterministic responses."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        merge_system_prompt: bool = False,
        quantization_bits: int | None = 4,
        cache_dir: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - optional heavyweight packages
            raise RuntimeError('Install inference support with: pip install -e ".[inference]"') from exc

        if quantization_bits not in {None, 0, 4, 8}:
            raise ValueError("quantization_bits must be 0, 4, 8, or None")
        if quantization_bits in {4, 8} and not torch.cuda.is_available():
            raise RuntimeError(
                "4/8-bit inference requires a CUDA GPU; use quantization_bits=0 for CPU"
            )

        load_kwargs: dict[str, Any] = {
            "revision": revision,
            "cache_dir": cache_dir,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if quantization_bits == 4:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
        elif quantization_bits == 8:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        self.merge_system_prompt = merge_system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

    def generate(
        self,
        user_prompts: Sequence[str],
        *,
        system_prompt: str,
        batch_size: int = 8,
        max_input_tokens: int = 384,
        max_new_tokens: int = 512,
        seed: int = 123,
    ) -> list[str]:
        """Generate completions in stable input order using greedy decoding."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.torch.manual_seed(seed)
        responses: list[str] = []
        terminators = [self.tokenizer.eos_token_id]
        if self.tokenizer.pad_token_id not in terminators:
            terminators.append(self.tokenizer.pad_token_id)

        for start in range(0, len(user_prompts), batch_size):
            batch = user_prompts[start : start + batch_size]
            rendered = []
            for prompt in batch:
                text = self.tokenizer.apply_chat_template(
                    chat_messages(system_prompt, prompt, merge_system=self.merge_system_prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )
                # Match the original runner's removal of tokenizer-injected date stubs.
                text = re.sub(
                    r"^(Cutting Knowledge Date:|Today Date:).*\n?",
                    "",
                    text,
                    flags=re.MULTILINE,
                )
                rendered.append(text)
            inputs = self.tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            ).to(self.model.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            prompt_width = inputs["input_ids"].shape[1]
            responses.extend(
                self.tokenizer.batch_decode(
                    generated[:, prompt_width:], skip_special_tokens=True
                )
            )
            LOGGER.info("Generated %d/%d responses", min(start + len(batch), len(user_prompts)), len(user_prompts))
        return [response.strip() for response in responses]


def prepare_records(
    profiles: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    prompt_seed: int,
    attribute_seed: int | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build prompts and retain the identifiers required to join results."""
    prompts: list[str] = []
    records: list[dict[str, Any]] = []
    for position, profile in enumerate(profiles):
        identifier = str(profile.get("id", position))
        prompt = build_prompt(
            profile,
            mode=mode,
            prompt_seed=prompt_seed,
            attribute_seed=attribute_seed,
        )
        prompts.append(prompt)
        records.append({"id": identifier})
    return prompts, records


def attach_responses(
    records: Sequence[Mapping[str, Any]], responses: Sequence[str], *, mode: str
) -> list[dict[str, Any]]:
    """Attach raw and parsed model outputs without dropping failures."""
    if len(records) != len(responses):
        raise ValueError("records and responses must have the same length")
    output: list[dict[str, Any]] = []
    for record, raw_response in zip(records, responses, strict=True):
        item = dict(record)
        item["raw_response"] = raw_response
        item.update(parse_response(raw_response, mode=mode))
        output.append(item)
    return output


__all__ = [
    "HFInferenceRunner",
    "attach_responses",
    "build_system_prompt",
    "chat_messages",
    "prepare_records",
]
