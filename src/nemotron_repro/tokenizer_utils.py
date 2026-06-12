# src/nemotron_repro/tokenizer_utils.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer


DEFAULT_MODEL_CONFIG_PATH = "configs/model_config.json"


def load_json_config(path: str | os.PathLike) -> Dict[str, Any]:
    """
    Load a JSON config file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_config(
    config_path: str | os.PathLike = DEFAULT_MODEL_CONFIG_PATH,
) -> Dict[str, Any]:
    """
    Load model/tokenizer config.
    """
    return load_json_config(config_path)


def resolve_tokenizer_args(
    model_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    use_fast: Optional[bool] = None,
    config_path: str | os.PathLike = DEFAULT_MODEL_CONFIG_PATH,
) -> Dict[str, Any]:
    """
    Resolve tokenizer arguments from CLI overrides and model_config.json.

    CLI arguments have priority over config values.
    """
    config = load_model_config(config_path)

    resolved = {
        "model_name": model_name or config.get("model_name"),
        "cache_dir": cache_dir or config.get("cache_dir"),
        "trust_remote_code": (
            trust_remote_code
            if trust_remote_code is not None
            else bool(config.get("trust_remote_code", True))
        ),
        "use_fast": (
            use_fast
            if use_fast is not None
            else bool(config.get("use_fast", True))
        ),
        "max_seq_len": int(config.get("max_seq_len", 8192)),
    }

    if not resolved["model_name"]:
        raise ValueError(
            "model_name is required. Set it in configs/model_config.json "
            "or pass --model-name."
        )

    return resolved


def load_tokenizer(
    model_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    use_fast: Optional[bool] = None,
    config_path: str | os.PathLike = DEFAULT_MODEL_CONFIG_PATH,
):
    """
    Load only the tokenizer.

    Important:
        This does not load model weights.
        It only downloads/loads tokenizer files.
    """
    args = resolve_tokenizer_args(
        model_name=model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
        config_path=config_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args["model_name"],
        cache_dir=args["cache_dir"],
        trust_remote_code=args["trust_remote_code"],
        use_fast=args["use_fast"],
    )

    return tokenizer


def load_messages(value) -> List[Dict[str, Any]]:
    """
    Parse messages from CSV.

    Expected CSV value:
        JSON string created from list[dict].

    Returns:
        list of message dictionaries.
    """
    if isinstance(value, list):
        return value

    if value is None:
        return []

    try:
        # pandas NaN check without importing pandas here
        if value != value:
            return []
    except Exception:
        pass

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            return []
        except Exception as e:
            raise ValueError(
                f"Could not parse messages JSON. Preview: {value[:200]}"
            ) from e

    return []


def fallback_format_messages(
    messages: List[Dict[str, Any]],
    include_assistant: bool = True,
    add_generation_prompt: bool = False,
) -> str:
    """
    Fallback chat formatting when tokenizer.apply_chat_template fails.

    This is not preferred for final tokenization, but keeps scripts usable
    if the tokenizer template has an unexpected issue.
    """
    chunks = []

    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))

        if role == "user":
            chunks.append(f"<|user|>\n{content}\n")

        elif role == "assistant" and include_assistant:
            reasoning = str(msg.get("reasoning_content", ""))

            if reasoning:
                chunks.append(
                    f"<|assistant|>\n"
                    f"<think>\n{reasoning}\n</think>\n"
                    f"{content}\n"
                )
            else:
                chunks.append(f"<|assistant|>\n{content}\n")

    if add_generation_prompt:
        chunks.append("<|assistant|>\n")

    return "".join(chunks)


def render_chat(
    tokenizer,
    messages: List[Dict[str, Any]],
    include_assistant: bool = True,
    add_generation_prompt: bool = False,
) -> str:
    """
    Render messages with the tokenizer's chat template.

    Args:
        include_assistant:
            If False, assistant messages are removed. Used to get prompt prefix.

        add_generation_prompt:
            If True, append assistant generation prompt.
    """
    if include_assistant:
        rendered_messages = messages
    else:
        rendered_messages = [
            msg for msg in messages
            if msg.get("role") != "assistant"
        ]

    try:
        return tokenizer.apply_chat_template(
            rendered_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return fallback_format_messages(
            rendered_messages,
            include_assistant=include_assistant,
            add_generation_prompt=add_generation_prompt,
        )


def assistant_text_from_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Extract assistant-side text from messages.

    This is useful for quick assistant-token budget estimates.
    """
    parts = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        if msg.get("role") != "assistant":
            continue

        reasoning = msg.get("reasoning_content", "")
        content = msg.get("content", "")

        if reasoning:
            parts.append(str(reasoning))

        if content:
            parts.append(str(content))

    return "\n".join(parts)


def count_text_tokens(
    tokenizer,
    text: str,
    add_special_tokens: bool = False,
) -> int:
    """
    Count tokens for plain text.
    """
    return len(
        tokenizer(
            str(text),
            add_special_tokens=add_special_tokens,
        )["input_ids"]
    )


def tokenize_messages_assistant_only(
    tokenizer,
    messages: List[Dict[str, Any]],
    max_seq_len: int = 8192,
    answer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert chat messages into causal-LM input_ids and assistant-only labels.

    Output:
        input_ids:
            Full prompt + assistant tokens.

        attention_mask:
            1 for real tokens.

        labels:
            Prompt tokens are masked as -100.
            Assistant tokens keep their token ids.

        truncated:
            True if full sequence exceeded max_seq_len.

        unmasked_tokens:
            Number of assistant-side tokens used for loss.

    Important:
        This function does not load or train a model.
        It only uses the tokenizer.
    """
    prefix_text = render_chat(
        tokenizer=tokenizer,
        messages=messages,
        include_assistant=False,
        add_generation_prompt=True,
    )

    full_text = render_chat(
        tokenizer=tokenizer,
        messages=messages,
        include_assistant=True,
        add_generation_prompt=False,
    )

    prefix_ids = tokenizer(
        prefix_text,
        add_special_tokens=False,
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
    )["input_ids"]

    if (
        tokenizer.eos_token_id is not None
        and len(full_ids) > 0
        and full_ids[-1] != tokenizer.eos_token_id
    ):
        full_ids = full_ids + [tokenizer.eos_token_id]

    original_seq_len = len(full_ids)
    truncated = original_seq_len > max_seq_len

    input_ids = full_ids[:max_seq_len]
    attention_mask = [1] * len(input_ids)

    labels = input_ids.copy()

    prompt_len = min(len(prefix_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    unmasked_tokens = sum(1 for token_id in labels if token_id != -100)

    answer_in_truncated_tail = None

    if answer is not None:
        answer_str = str(answer).strip()
        if answer_str:
            try:
                tail_text = tokenizer.decode(
                    input_ids[-512:],
                    skip_special_tokens=False,
                )
                answer_in_truncated_tail = answer_str in tail_text
            except Exception:
                answer_in_truncated_tail = None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_tokens": prompt_len,
        "seq_len": len(input_ids),
        "original_seq_len": original_seq_len,
        "truncated": bool(truncated),
        "unmasked_tokens": int(unmasked_tokens),
        "answer_in_truncated_tail": answer_in_truncated_tail,
    }