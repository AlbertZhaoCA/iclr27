#!/usr/bin/env python3
"""
Evaluate mathematical reasoning models with vLLM offline inference.

Main features
-------------
1. Direct single-stage evaluation (no PTA / no plan generation).
2. Supports MATH500, AIME24, AIME25, AIME26, AMC23, and MinervaMath.
3. Repeated runs with configurable generation seeds.
4. Reports mean accuracy, sample variance, standard deviation, standard error,
   and a 95% confidence interval across runs.
5. Optionally evaluates a baseline model on exactly the same examples and runs
   paired significance tests:
      - paired t-test across run-level accuracies
      - Wilcoxon signed-rank test across run-level accuracies
      - exact McNemar test across paired example outcomes
      - paired bootstrap confidence interval for the accuracy difference
      - Benjamini-Hochberg correction across datasets
6. Saves per-example predictions, per-run metrics, CSV/Markdown/LaTeX tables,
   and the exact sampled test sets.

Example
-------
python eval_vllm_accuracy_stats.py \
    --model  Qwen/Qwen3-8B \
    --model_name sft \
    --tensor_parallel_size 4 \
    --num_runs 1 \
    --temperature 0.6 \
    --out_dir runs/qwen3 \
    --max_tokens 32768 \

With a paired baseline comparison:

python eval_vllm_accuracy_stats.py \
    --model //scratch/pioneer/jobs/xxl1337/qwen25_opsd_merged \
    --model_name Ours \
    --baseline_model Qwen/Qwen2.5-7B-Instruct \
    --baseline_name Base \
    --tensor_parallel_size 4 \
    --num_runs 5 \
    --temperature 0.6 \
    --out_dir runs/ours_vs_base

Notes
-----
- With temperature=0 and a fixed test set, repeated runs are normally
  deterministic, so the observed run-to-run variance may be zero.
- Statistical significance is only meaningful relative to a comparator.
  Therefore significance tests are produced only when --baseline_model is set.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import os

os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"

try:
    from scipy import stats as scipy_stats
except ImportError:  # The script can still produce descriptive statistics.
    scipy_stats = None


DATASETS = ["MATH500", "AIME24", "AIME25", "AIME26", "AMC23", "MinervaMath"]

DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "MATH500": {
        "path": "HuggingFaceH4/MATH-500",
        "name": None,
        "split": "test",
        "filters": [],
    },
    "AIME24": {
        "path": "math-ai/aime24",
        "name": None,
        "split": "test",
    },
    "AIME25": {
        "path": "math-ai/aime25",
        "name": None,
        "split": "test",
    },
    "AIME26": {
        "path": "math-ai/aime26",
        "name": None,
        "split": "test",
        "filters": [],
    },
    "AMC23": {
        "path": "AI-MO/aimo-validation-amc",
        "name": None,
        "split": "train",
        "filters": [{"field": "url", "op": "contains", "value": "2023"}],
    },
    "MinervaMath": {
        "path": "math-ai/minervamath",
        "name": None,
        "split": "test",
        "filters": [],
    },
}


USER_PROMPT_TEMPLATE = r"""Return your final response within \boxed{{}}. {question}""".strip()


@dataclass(frozen=True)
class Example:
    dataset: str
    example_id: str
    question: str
    groundtruth: str


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def stable_hash_int(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned.strip("_") or "model"


def save_jsonl(items: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")


def append_jsonl(item: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_json(item: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(item), f, ensure_ascii=False, indent=2)


def bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """Add --foo / --no_foo flags in a Python-version-compatible way."""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    group.add_argument(f"--no_{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: default})


# -----------------------------------------------------------------------------
# Dataset loading and sampling
# -----------------------------------------------------------------------------


def load_hf_dataset_examples(
    cfg: Mapping[str, Any],
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face 'datasets' package is required. Install it with "
            "`pip install datasets`."
        ) from exc

    kwargs: Dict[str, Any] = {
        "path": cfg["path"],
        "split": cfg.get("split", "train"),
    }
    if cfg.get("name") is not None:
        kwargs["name"] = cfg["name"]
    if hf_token:
        kwargs["token"] = hf_token

    dataset = load_dataset(**kwargs)
    return [dict(row) for row in dataset]


def apply_filters(
    examples: Sequence[Mapping[str, Any]],
    filters: Sequence[Mapping[str, Any]],
    dataset_name: str,
) -> List[Dict[str, Any]]:
    if not filters:
        return [dict(x) for x in examples]

    selected: List[Dict[str, Any]] = []
    for raw_example in examples:
        keep = True
        for flt in filters:
            field = str(flt["field"])
            op = str(flt["op"])
            value = flt["value"]

            if field not in raw_example:
                keep = False
                break

            field_value = raw_example[field]
            if op == "eq":
                keep = str(field_value) == str(value)
            elif op == "contains":
                keep = str(value) in str(field_value)
            elif op == "regex":
                keep = re.search(str(value), str(field_value)) is not None
            else:
                raise ValueError(f"Unsupported filter operation: {op}")

            if not keep:
                break

        if keep:
            selected.append(dict(raw_example))

    if not selected:
        available_keys = list(examples[0].keys()) if examples else []
        raise RuntimeError(
            f"{dataset_name} filtering returned zero examples. "
            f"Available fields: {available_keys}"
        )
    return selected


def get_question(example: Mapping[str, Any]) -> str:
    for key in ["question", "problem", "prompt", "input", "content", "query"]:
        if key in example and example[key] is not None:
            return str(example[key])
    raise KeyError(f"Cannot find question field. Available keys: {list(example.keys())}")


def get_groundtruth(example: Mapping[str, Any]) -> str:
    for key in [
        "groundtruth",
        "answer",
        "final_answer",
        "gold",
        "target",
        "label",
        "solution"
    ]:
        if key in example and example[key] is not None:
            return str(example[key])
    raise KeyError(f"Cannot find answer field. Available keys: {list(example.keys())}")


def get_example_id(example: Mapping[str, Any], dataset_name: str, index: int, question: str) -> str:
    for key in ["id", "problem_id", "question_id", "uid", "url"]:
        if key in example and example[key] not in (None, ""):
            return f"{dataset_name}:{example[key]}"
    return f"{dataset_name}:{index}:{stable_hash_int(question):08x}"


def load_all_datasets(hf_token: Optional[str]) -> Dict[str, List[Example]]:
    loaded: Dict[str, List[Example]] = {}

    for dataset_name in DATASETS:
        cfg = DATASET_CONFIG[dataset_name]
        print(f"\n[Load] {dataset_name}: {cfg['path']}")
        raw = load_hf_dataset_examples(cfg, hf_token=hf_token)
        filtered = apply_filters(raw, cfg.get("filters", []), dataset_name)

        examples: List[Example] = []
        for idx, row in enumerate(filtered):
            question = get_question(row).strip()
            groundtruth = get_groundtruth(row).strip()
            example_id = get_example_id(row, dataset_name, idx, question)
            examples.append(
                Example(
                    dataset=dataset_name,
                    example_id=example_id,
                    question=question,
                    groundtruth=groundtruth,
                )
            )

        loaded[dataset_name] = examples
        print(f"[Load] {dataset_name}: {len(examples)} usable examples")

    return loaded


def sample_examples(
    examples: Sequence[Example],
    ratio: float,
    seed: int,
    max_examples: Optional[int],
) -> List[Example]:
    if not 0 < ratio <= 1:
        raise ValueError(f"sample_ratio must be in (0, 1], got {ratio}")

    n = len(examples)
    k = max(1, min(n, int(n * ratio + 0.5)))
    if max_examples is not None and max_examples > 0:
        k = min(k, max_examples)

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    chosen = sorted(indices[:k])
    return [examples[i] for i in chosen]


def build_run_testsets(
    all_examples: Mapping[str, Sequence[Example]],
    num_runs: int,
    sample_ratio: float,
    max_examples_per_dataset: Optional[int],
    base_seed: int,
    resample_each_run: bool,
    out_dir: Path,
) -> Dict[int, Dict[str, List[Example]]]:
    testsets: Dict[int, Dict[str, List[Example]]] = {}
    testset_dir = out_dir / "sampled_testsets"

    for run_idx in range(num_runs):
        run_testsets: Dict[str, List[Example]] = {}
        for dataset_name in DATASETS:
            sampling_run = run_idx if resample_each_run else 0
            sample_seed = (
                base_seed
                + 100_003 * sampling_run
                + stable_hash_int(dataset_name)
            )
            selected = sample_examples(
                all_examples[dataset_name],
                ratio=sample_ratio,
                seed=sample_seed,
                max_examples=max_examples_per_dataset,
            )
            run_testsets[dataset_name] = selected

            rows = [
                {
                    "run": run_idx,
                    "dataset": ex.dataset,
                    "example_id": ex.example_id,
                    "question": ex.question,
                    "groundtruth": ex.groundtruth,
                }
                for ex in selected
            ]
            save_jsonl(
                rows,
                testset_dir / f"run_{run_idx:03d}_{dataset_name}.jsonl",
            )

        testsets[run_idx] = run_testsets

    return testsets


# -----------------------------------------------------------------------------
# Prompting and answer verification
# -----------------------------------------------------------------------------


def build_conversation(question: str) -> List[Dict[str, str]]:
    return [
        # {
        #     "role": "system",
        #     "content": SYSTEM_PROMPT,
        # },
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(question=question.strip()),
        },
    ]


def extract_last_boxed(text: str) -> str:
    """Extract the content of the last \boxed{...}, supporting nested braces."""
    if not text:
        return ""

    matches = list(re.finditer(r"\\boxed\s*\{|(?<!\\)boxed\s*\{", text))
    if not matches:
        return ""

    start = text.find("{", matches[-1].start())
    if start < 0:
        return ""

    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx].strip()
    return ""


def extract_answer_from_response(response: str) -> str:
    boxed = extract_last_boxed(response)
    if boxed:
        return boxed.strip()

    # Common fallback patterns when a model fails to follow the boxed format.
    patterns = [
        r"(?:final answer|answer)\s*(?:is|:|=)\s*(.+)$",
        r"答案\s*(?:是|为|:|：)\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def strip_outer_braces(text: str) -> str:
    value = text.strip()
    while value.startswith("{") and value.endswith("}"):
        depth = 0
        encloses_all = True
        for idx, char in enumerate(value):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and idx != len(value) - 1:
                    encloses_all = False
                    break
        if not encloses_all:
            break
        value = value[1:-1].strip()
    return value


def normalize_answer(text: Any) -> str:
    if text is None:
        return ""

    value = str(text).strip()
    boxed = extract_last_boxed(value)
    if boxed:
        value = boxed

    value = value.replace("\n", " ")
    value = value.replace("−", "-").replace("–", "-")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\!", "")
    value = value.replace("\\;", "").replace("\\:", "")
    value = value.replace("\\cdot", "*").replace("\\times", "*")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("$", "")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(".。")
    value = strip_outer_braces(value)

    # Remove a simple variable assignment, e.g. x=3 -> 3.
    assignment = re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*=(.+)", value)
    if assignment:
        value = assignment.group(1)

    # Canonicalize integer-looking decimals.
    if re.fullmatch(r"[-+]?\d+\.0+", value):
        try:
            value = str(int(float(value)))
        except ValueError:
            pass

    return value


def latex_fraction_to_float(text: str) -> Optional[float]:
    match = re.fullmatch(r"([-+]?)\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    if not match:
        return None
    sign, numerator, denominator = match.groups()
    try:
        result = float(numerator) / float(denominator)
        return -result if sign == "-" else result
    except (ValueError, ZeroDivisionError):
        return None


def parse_numeric(text: str) -> Optional[float]:
    value = normalize_answer(text)
    value = value.replace(",", "")

    fraction_value = latex_fraction_to_float(value)
    if fraction_value is not None:
        return fraction_value

    slash_fraction = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)", value)
    if slash_fraction:
        numerator, denominator = slash_fraction.groups()
        try:
            return float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            return None

    percent_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)%", value)
    if percent_match:
        return float(percent_match.group(1)) / 100.0

    try:
        return float(value)
    except ValueError:
        return None


def split_top_level_items(text: str) -> Optional[List[str]]:
    """Parse simple unordered sets/tuples without splitting inside braces."""
    value = normalize_answer(text)
    if len(value) < 2:
        return None

    bracket_pairs = [("\\{", "\\}"), ("{", "}"), ("(", ")"), ("[", "]")]
    stripped: Optional[str] = None
    for left, right in bracket_pairs:
        if value.startswith(left) and value.endswith(right):
            stripped = value[len(left) : -len(right)]
            break
    if stripped is None or "," not in stripped:
        return None

    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for char in stripped:
        if char in "{([":
            depth += 1
        elif char in "})]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))

    normalized = sorted(normalize_answer(part) for part in parts if part != "")
    return normalized or None


def try_sympy_equal(prediction: str, groundtruth: str) -> bool:
    """Best-effort symbolic equivalence; safely returns False if unavailable."""
    try:
        import sympy as sp
    except ImportError:
        return False

    pred = normalize_answer(prediction)
    gold = normalize_answer(groundtruth)

    # First try ordinary SymPy parsing after a few LaTeX-to-text conversions.
    def simple_to_sympy(text: str) -> str:
        value = text
        value = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", value)
        value = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", value)
        value = value.replace("^", "**")
        value = value.replace("\\pi", "pi")
        return value

    try:
        pred_expr = sp.sympify(simple_to_sympy(pred))
        gold_expr = sp.sympify(simple_to_sympy(gold))
        return bool(sp.simplify(pred_expr - gold_expr) == 0)
    except Exception:
        pass

    # parse_latex is more capable but may require antlr4, so keep it optional.
    try:
        from sympy.parsing.latex import parse_latex

        pred_expr = parse_latex(pred)
        gold_expr = parse_latex(gold)
        return bool(sp.simplify(pred_expr - gold_expr) == 0)
    except Exception:
        return False


def answers_equal(prediction: str, groundtruth: str, tolerance: float = 1e-9) -> bool:
    pred_n = normalize_answer(prediction)
    gold_n = normalize_answer(groundtruth)

    if not pred_n or not gold_n:
        return False
    if pred_n == gold_n:
        return True

    pred_num = parse_numeric(pred_n)
    gold_num = parse_numeric(gold_n)
    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=tolerance, abs_tol=tolerance)

    pred_items = split_top_level_items(pred_n)
    gold_items = split_top_level_items(gold_n)
    if pred_items is not None and gold_items is not None:
        return pred_items == gold_items

    return try_sympy_equal(pred_n, gold_n)


# -----------------------------------------------------------------------------
# vLLM inference
# -----------------------------------------------------------------------------


def build_llm(
    model_path: str,
    tokenizer_path: Optional[str],
    args: argparse.Namespace,
    engine_seed: int,
):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed. Install a vLLM build compatible with your "
            "CUDA/ROCm environment before running this script."
        ) from exc

    kwargs: Dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": engine_seed,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "generation_config": args.generation_config,
    }
    if tokenizer_path:
        kwargs["tokenizer"] = tokenizer_path
    if args.max_model_len and args.max_model_len > 0:
        kwargs["max_model_len"] = args.max_model_len
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.cpu_offload_gb > 0:
        kwargs["cpu_offload_gb"] = args.cpu_offload_gb

    return LLM(**kwargs)


def release_llm(llm: Any) -> None:
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def make_sampling_params(args: argparse.Namespace, generation_seed: int):
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is required for inference.") from exc

    return SamplingParams(
        n=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        seed=generation_seed,
    )


def chunked(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        yield items
        return
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def prediction_path(out_dir: Path, model_name: str, run_idx: int, dataset: str) -> Path:
    safe_name = sanitize_name(model_name)
    return out_dir / "predictions" / safe_name / f"run_{run_idx:03d}_{dataset}.jsonl"


def load_completed_prediction_map(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row["example_id"]): row for row in rows if row.get("example_id")}


def generate_dataset_predictions(
    llm: Any,
    model_name: str,
    model_path: str,
    run_idx: int,
    dataset_name: str,
    examples: Sequence[Example],
    args: argparse.Namespace,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    pred_path = prediction_path(out_dir, model_name, run_idx, dataset_name)
    completed = load_completed_prediction_map(pred_path) if args.resume else {}
    pending = [ex for ex in examples if ex.example_id not in completed]

    print(
        f"[Infer] model={model_name}, run={run_idx}, dataset={dataset_name}, "
        f"pending={len(pending)}/{len(examples)}"
    )

    # Use the same run/dataset seed for the primary and baseline models so that
    # stochastic decoding is paired as closely as possible.
    generation_seed = args.seed + run_idx * 10_007 + stable_hash_int(dataset_name)
    sampling_params = make_sampling_params(args, generation_seed)

    for example_batch in chunked(pending, args.batch_size):
        conversations = [build_conversation(ex.question) for ex in example_batch]
        outputs = llm.chat(
            conversations,
            sampling_params=sampling_params,
            use_tqdm=args.use_tqdm,
        )

        if len(outputs) != len(example_batch):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(example_batch)} inputs."
            )

        for ex, output in zip(example_batch, outputs):
            response = output.outputs[0].text if output.outputs else ""
            extracted_answer = extract_answer_from_response(response)
            correct = answers_equal(extracted_answer, ex.groundtruth)
            finish_reason = output.outputs[0].finish_reason if output.outputs else None
            token_count = len(output.outputs[0].token_ids) if output.outputs else 0

            append_jsonl(
                {
                    "model_name": model_name,
                    "model_path": model_path,
                    "run": run_idx,
                    "generation_seed": generation_seed,
                    "dataset": dataset_name,
                    "example_id": ex.example_id,
                    "question": ex.question,
                    "groundtruth": ex.groundtruth,
                    "response": response,
                    "extracted_answer": extracted_answer,
                    "normalized_prediction": normalize_answer(extracted_answer),
                    "normalized_groundtruth": normalize_answer(ex.groundtruth),
                    "correct": bool(correct),
                    "finish_reason": finish_reason,
                    "output_tokens": token_count,
                },
                pred_path,
            )

    final_map = load_completed_prediction_map(pred_path)
    missing_ids = [ex.example_id for ex in examples if ex.example_id not in final_map]
    if missing_ids:
        raise RuntimeError(
            f"Missing {len(missing_ids)} predictions for {model_name}, "
            f"run={run_idx}, dataset={dataset_name}."
        )

    # Return rows in the exact test-set order.
    return [final_map[ex.example_id] for ex in examples]


def evaluate_model(
    model_path: str,
    model_name: str,
    tokenizer_path: Optional[str],
    testsets: Mapping[int, Mapping[str, Sequence[Example]]],
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    engine_seed = args.seed + stable_hash_int(model_name)
    print(f"\n[Model] Loading {model_name}: {model_path}")
    llm = build_llm(
        model_path=model_path,
        tokenizer_path=model_path,
        args=args,
        engine_seed=engine_seed,
    )

    all_rows: List[Dict[str, Any]] = []
    run_metric_rows: List[Dict[str, Any]] = []

    try:
        for run_idx in range(args.num_runs):
            dataset_accuracies: List[float] = []
            for dataset_name in DATASETS:
                rows = generate_dataset_predictions(
                    llm=llm,
                    model_name=model_name,
                    model_path=model_path,
                    run_idx=run_idx,
                    dataset_name=dataset_name,
                    examples=testsets[run_idx][dataset_name],
                    args=args,
                    out_dir=out_dir,
                )
                all_rows.extend(rows)

                correct = int(sum(bool(row["correct"]) for row in rows))
                total = len(rows)
                accuracy = 100.0 * correct / total if total else float("nan")
                dataset_accuracies.append(accuracy)
                run_metric_rows.append(
                    {
                        "model_name": model_name,
                        "run": run_idx,
                        "dataset": dataset_name,
                        "correct": correct,
                        "total": total,
                        "accuracy": accuracy,
                    }
                )

            macro_accuracy = float(np.mean(dataset_accuracies)) if dataset_accuracies else float("nan")
            run_metric_rows.append(
                {
                    "model_name": model_name,
                    "run": run_idx,
                    "dataset": "MacroAverage",
                    "correct": np.nan,
                    "total": np.nan,
                    "accuracy": macro_accuracy,
                }
            )
    finally:
        # Delete the caller's reference before emptying the CUDA cache.  Calling a
        # helper with llm as an argument would leave this local reference alive.
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    predictions_df = pd.DataFrame(all_rows)
    run_metrics_df = pd.DataFrame(run_metric_rows)

    safe_name = sanitize_name(model_name)
    predictions_df.to_csv(
        out_dir / f"{safe_name}_all_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    save_jsonl(
        predictions_df.to_dict(orient="records"),
        out_dir / f"{safe_name}_all_predictions.jsonl",
    )
    run_metrics_df.to_csv(
        out_dir / f"{safe_name}_run_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    return predictions_df, run_metrics_df


# -----------------------------------------------------------------------------
# Descriptive statistics
# -----------------------------------------------------------------------------


def t_critical_975(df: int) -> float:
    if df <= 0:
        return float("nan")
    if scipy_stats is not None:
        return float(scipy_stats.t.ppf(0.975, df=df))
    return 1.959963984540054


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    clean = [float(x) for x in values if not pd.isna(x)]
    n = len(clean)
    if n == 0:
        return {
            "num_runs": 0,
            "mean_accuracy": float("nan"),
            "sample_variance": float("nan"),
            "std_dev": float("nan"),
            "std_error": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "min_accuracy": float("nan"),
            "max_accuracy": float("nan"),
        }

    mean = statistics.fmean(clean)
    if n >= 2:
        variance = statistics.variance(clean)
        std_dev = math.sqrt(variance)
        std_error = std_dev / math.sqrt(n)
        half_width = t_critical_975(n - 1) * std_error
        ci_low = mean - half_width
        ci_high = mean + half_width
    else:
        variance = float("nan")
        std_dev = float("nan")
        std_error = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")

    return {
        "num_runs": n,
        "mean_accuracy": mean,
        "sample_variance": variance,
        "std_dev": std_dev,
        "std_error": std_error,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "min_accuracy": min(clean),
        "max_accuracy": max(clean),
    }


def make_model_summary(run_metrics: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dataset_order = DATASETS + ["MacroAverage"]
    for dataset_name in dataset_order:
        values = run_metrics.loc[
            (run_metrics["model_name"] == model_name)
            & (run_metrics["dataset"] == dataset_name),
            "accuracy",
        ].tolist()
        row: Dict[str, Any] = {"model_name": model_name, "dataset": dataset_name}
        row.update(summarize_values(values))
        rows.append(row)
    return pd.DataFrame(rows)


def format_summary_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.copy()
    numeric_cols = [
        "mean_accuracy",
        "sample_variance",
        "std_dev",
        "std_error",
        "ci95_low",
        "ci95_high",
        "min_accuracy",
        "max_accuracy",
    ]
    for col in numeric_cols:
        display[col] = display[col].map(
            lambda x: "NA" if pd.isna(x) else f"{float(x):.4f}"
        )
    return display


# -----------------------------------------------------------------------------
# Paired significance testing
# -----------------------------------------------------------------------------


def align_predictions(
    primary: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["run", "dataset", "example_id"]
    left = primary[keys + ["correct"]].rename(columns={"correct": "primary_correct"})
    right = baseline[keys + ["correct"]].rename(columns={"correct": "baseline_correct"})
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")

    expected = min(len(left), len(right))
    if len(merged) != expected:
        warnings.warn(
            f"Only {len(merged)} paired predictions were aligned out of "
            f"primary={len(left)}, baseline={len(right)}."
        )
    merged["primary_correct"] = merged["primary_correct"].astype(bool)
    merged["baseline_correct"] = merged["baseline_correct"].astype(bool)
    return merged


def paired_bootstrap_difference(
    primary_correct: np.ndarray,
    baseline_correct: np.ndarray,
    group_ids: Sequence[str],
    num_samples: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Cluster bootstrap by example ID, preserving dependence across runs."""
    if len(primary_correct) == 0:
        return float("nan"), float("nan"), float("nan")

    differences = primary_correct.astype(float) - baseline_correct.astype(float)
    observed = 100.0 * float(np.mean(differences))

    groups = pd.DataFrame(
        {"group_id": list(group_ids), "difference": differences}
    ).groupby("group_id", sort=False)["difference"].agg(["sum", "count"])
    group_sums = groups["sum"].to_numpy(dtype=float)
    group_counts = groups["count"].to_numpy(dtype=float)
    num_groups = len(groups)

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(num_samples, dtype=float)
    for idx in range(num_samples):
        sampled_groups = rng.integers(0, num_groups, size=num_groups)
        sampled_sum = float(np.sum(group_sums[sampled_groups]))
        sampled_count = float(np.sum(group_counts[sampled_groups]))
        bootstrap_means[idx] = 100.0 * sampled_sum / sampled_count

    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return observed, float(low), float(high)


def exact_mcnemar(primary_correct: np.ndarray, baseline_correct: np.ndarray) -> Dict[str, Any]:
    primary_only = int(np.sum(primary_correct & ~baseline_correct))
    baseline_only = int(np.sum(~primary_correct & baseline_correct))
    discordant = primary_only + baseline_only

    if discordant == 0:
        p_value = 1.0
    elif scipy_stats is not None:
        p_value = float(
            scipy_stats.binomtest(
                primary_only,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    else:
        p_value = float("nan")

    return {
        "primary_only_correct": primary_only,
        "baseline_only_correct": baseline_only,
        "discordant_pairs": discordant,
        "mcnemar_exact_p": p_value,
    }


def paired_run_tests(
    primary_run_acc: Sequence[float],
    baseline_run_acc: Sequence[float],
) -> Dict[str, float]:
    primary = np.asarray(primary_run_acc, dtype=float)
    baseline = np.asarray(baseline_run_acc, dtype=float)
    valid = ~(np.isnan(primary) | np.isnan(baseline))
    primary = primary[valid]
    baseline = baseline[valid]

    result = {
        "paired_t_stat": float("nan"),
        "paired_t_p": float("nan"),
        "wilcoxon_stat": float("nan"),
        "wilcoxon_p": float("nan"),
    }

    if len(primary) < 2 or scipy_stats is None:
        return result

    differences = primary - baseline
    if np.allclose(differences, differences[0]):
        # scipy may report precision-loss warnings for a constant vector.
        if np.isclose(differences[0], 0.0):
            result["paired_t_stat"] = 0.0
            result["paired_t_p"] = 1.0
        else:
            result["paired_t_stat"] = math.copysign(float("inf"), differences[0])
            result["paired_t_p"] = 0.0
    else:
        t_result = scipy_stats.ttest_rel(primary, baseline, nan_policy="omit")
        result["paired_t_stat"] = float(t_result.statistic)
        result["paired_t_p"] = float(t_result.pvalue)

    if np.allclose(differences, 0.0):
        result["wilcoxon_stat"] = 0.0
        result["wilcoxon_p"] = 1.0
    else:
        try:
            w_result = scipy_stats.wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
            )
            result["wilcoxon_stat"] = float(w_result.statistic)
            result["wilcoxon_p"] = float(w_result.pvalue)
        except ValueError:
            pass

    return result


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan, dtype=float)
    valid_indices = np.where(~np.isnan(p))[0]
    if len(valid_indices) == 0:
        return adjusted.tolist()

    valid_p = p[valid_indices]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    m = len(ranked)

    corrected = np.empty(m, dtype=float)
    running_min = 1.0
    for reverse_idx in range(m - 1, -1, -1):
        rank = reverse_idx + 1
        candidate = ranked[reverse_idx] * m / rank
        running_min = min(running_min, candidate)
        corrected[reverse_idx] = min(1.0, running_min)

    unsorted = np.empty(m, dtype=float)
    unsorted[order] = corrected
    adjusted[valid_indices] = unsorted
    return adjusted.tolist()


def make_significance_table(
    primary_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    primary_run_metrics: pd.DataFrame,
    baseline_run_metrics: pd.DataFrame,
    primary_name: str,
    baseline_name: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    aligned = align_predictions(primary_predictions, baseline_predictions)
    rows: List[Dict[str, Any]] = []

    for dataset_name in DATASETS:
        dataset_pairs = aligned[aligned["dataset"] == dataset_name]
        primary_correct = dataset_pairs["primary_correct"].to_numpy(dtype=bool)
        baseline_correct = dataset_pairs["baseline_correct"].to_numpy(dtype=bool)

        primary_run = primary_run_metrics[
            (primary_run_metrics["dataset"] == dataset_name)
            & (primary_run_metrics["model_name"] == primary_name)
        ].sort_values("run")
        baseline_run = baseline_run_metrics[
            (baseline_run_metrics["dataset"] == dataset_name)
            & (baseline_run_metrics["model_name"] == baseline_name)
        ].sort_values("run")
        paired_runs = primary_run[["run", "accuracy"]].merge(
            baseline_run[["run", "accuracy"]],
            on="run",
            suffixes=("_primary", "_baseline"),
            validate="one_to_one",
        )

        primary_accuracy = 100.0 * float(np.mean(primary_correct)) if len(primary_correct) else float("nan")
        baseline_accuracy = 100.0 * float(np.mean(baseline_correct)) if len(baseline_correct) else float("nan")

        diff, boot_low, boot_high = paired_bootstrap_difference(
            primary_correct,
            baseline_correct,
            group_ids=dataset_pairs["example_id"].astype(str).tolist(),
            num_samples=bootstrap_samples,
            seed=seed + stable_hash_int(dataset_name),
        )
        row: Dict[str, Any] = {
            "dataset": dataset_name,
            "primary_model": primary_name,
            "baseline_model": baseline_name,
            "paired_predictions": len(dataset_pairs),
            "primary_accuracy": primary_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "difference_points": diff,
            "bootstrap_ci95_low": boot_low,
            "bootstrap_ci95_high": boot_high,
        }
        row.update(exact_mcnemar(primary_correct, baseline_correct))
        row.update(
            paired_run_tests(
                paired_runs["accuracy_primary"].tolist(),
                paired_runs["accuracy_baseline"].tolist(),
            )
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    result["mcnemar_bh_q"] = benjamini_hochberg(result["mcnemar_exact_p"].tolist())
    result["significant_at_0.05"] = result["mcnemar_bh_q"].map(
        lambda x: bool(x < 0.05) if not pd.isna(x) else False
    )
    result["direction"] = result["difference_points"].map(
        lambda x: "primary_better" if x > 0 else ("baseline_better" if x < 0 else "tie")
    )
    return result


# -----------------------------------------------------------------------------
# Output tables
# -----------------------------------------------------------------------------


def dataframe_to_markdown_safe(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def save_tables(
    out_dir: Path,
    primary_summary: pd.DataFrame,
    primary_run_metrics: pd.DataFrame,
    baseline_summary: Optional[pd.DataFrame],
    baseline_run_metrics: Optional[pd.DataFrame],
    significance: Optional[pd.DataFrame],
) -> None:
    primary_summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")
    primary_run_metrics.to_csv(out_dir / "run_metrics.csv", index=False, encoding="utf-8")

    display_sections: List[Tuple[str, pd.DataFrame]] = [
        ("Primary model summary", format_summary_for_display(primary_summary)),
        ("Primary run metrics", primary_run_metrics),
    ]

    if baseline_summary is not None and baseline_run_metrics is not None:
        baseline_summary.to_csv(
            out_dir / "baseline_summary.csv", index=False, encoding="utf-8"
        )
        baseline_run_metrics.to_csv(
            out_dir / "baseline_run_metrics.csv", index=False, encoding="utf-8"
        )
        display_sections.extend(
            [
                ("Baseline model summary", format_summary_for_display(baseline_summary)),
                ("Baseline run metrics", baseline_run_metrics),
            ]
        )

    if significance is not None:
        significance.to_csv(
            out_dir / "significance.csv", index=False, encoding="utf-8"
        )
        display_sections.append(("Paired significance tests", significance))
    else:
        status = pd.DataFrame(
            [
                {
                    "status": "not_computed",
                    "reason": "Set --baseline_model to run paired significance tests.",
                }
            ]
        )
        status.to_csv(out_dir / "significance.csv", index=False, encoding="utf-8")
        display_sections.append(("Paired significance tests", status))

    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        for title, table in display_sections:
            f.write(f"## {title}\n\n")
            f.write(dataframe_to_markdown_safe(table))
            f.write("\n\n")

    with (out_dir / "summary.tex").open("w", encoding="utf-8") as f:
        f.write("% Primary model summary\n")
        f.write(primary_summary.to_latex(index=False, float_format="%.4f"))
        if baseline_summary is not None:
            f.write("\n% Baseline model summary\n")
            f.write(baseline_summary.to_latex(index=False, float_format="%.4f"))
        if significance is not None:
            f.write("\n% Paired significance tests\n")
            f.write(significance.to_latex(index=False, float_format="%.6f"))


def save_config(args: argparse.Namespace, out_dir: Path) -> None:
    config = vars(args).copy()
    config["datasets"] = DATASETS
    config["dataset_config"] = DATASET_CONFIG
    # config["system_prompt"] = SYSTEM_PROMPT
    config["user_prompt_template"] = USER_PROMPT_TEMPLATE
    save_json(config, out_dir / "config.json")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct mathematical benchmark evaluation with vLLM and statistics."
    )

    # Models
    parser.add_argument("--model", type=str, required=True, help="Primary model path or Hugging Face ID.")
    parser.add_argument("--model_name", type=str, default="Model", help="Display name for the primary model.")
    parser.add_argument("--baseline_model", type=str, default=None, help="Optional baseline model path or HF ID.")
    parser.add_argument("--baseline_name", type=str, default="Baseline", help="Display name for the baseline model.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer path for the primary model.")
    parser.add_argument(
        "--baseline_tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer path for the baseline model.",
    )

    # Repeated evaluation
    parser.add_argument("--num_runs", type=int, default=5, help="Number of repeated generation runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Fraction of each filtered dataset to evaluate.")
    parser.add_argument(
        "--max_examples_per_dataset",
        type=int,
        default=None,
        help="Optional cap after applying sample_ratio; useful for debugging.",
    )
    bool_flag(
        parser,
        "resample_each_run",
        default=False,
        help_text="Draw a different subset in every run when sample_ratio < 1.",
    )
    bool_flag(parser, "resume", default=True, help_text="Resume from existing per-example JSONL files.")

    # Sampling
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=0, help="0 submits the entire dataset to vLLM at once.")
    bool_flag(parser, "use_tqdm", default=True, help_text="Show the vLLM generation progress bar.")

    # vLLM engine
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--cpu_offload_gb", type=float, default=0.0)
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument(
        "--generation_config",
        type=str,
        default="vllm",
        help="Use 'vllm' for explicit script settings or 'auto' for model generation_config.json.",
    )
    bool_flag(parser, "trust_remote_code", default=False, help_text="Allow remote model/tokenizer code.")
    bool_flag(parser, "enforce_eager", default=False, help_text="Disable CUDA graphs and use eager execution.")
    bool_flag(parser, "enable_prefix_caching", default=False, help_text="Enable vLLM prefix caching.")

    # Statistics and output
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="runs_vllm_accuracy_stats")

    args = parser.parse_args()

    if args.num_runs < 1:
        parser.error("--num_runs must be at least 1.")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap_samples should be at least 100.")
    if not 0 < args.sample_ratio <= 1:
        parser.error("--sample_ratio must be in (0, 1].")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative.")

    return args


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, out_dir)

    if args.num_runs > 1 and args.temperature == 0 and not args.resample_each_run:
        warnings.warn(
            "temperature=0 with a fixed test set is normally deterministic; "
            "run-to-run variance may be zero. Use temperature>0 to measure "
            "sampling variation, or --resample_each_run with sample_ratio<1 "
            "to include test-set sampling variation."
        )

    if scipy_stats is None:
        warnings.warn(
            "SciPy is not installed. Descriptive statistics will be produced, "
            "but significance-test p-values will be unavailable. Install scipy "
            "to enable paired t, Wilcoxon, and exact McNemar tests."
        )

    all_examples = load_all_datasets(args.hf_token)
    testsets = build_run_testsets(
        all_examples=all_examples,
        num_runs=args.num_runs,
        sample_ratio=args.sample_ratio,
        max_examples_per_dataset=args.max_examples_per_dataset,
        base_seed=args.seed,
        resample_each_run=args.resample_each_run,
        out_dir=out_dir,
    )

    primary_predictions, primary_run_metrics = evaluate_model(
        model_path=args.model,
        model_name=args.model_name,
        tokenizer_path=args.tokenizer,
        testsets=testsets,
        args=args,
        out_dir=out_dir,
    )
    primary_summary = make_model_summary(primary_run_metrics, args.model_name)

    baseline_predictions: Optional[pd.DataFrame] = None
    baseline_run_metrics: Optional[pd.DataFrame] = None
    baseline_summary: Optional[pd.DataFrame] = None
    significance: Optional[pd.DataFrame] = None

    if args.baseline_model:
        baseline_predictions, baseline_run_metrics = evaluate_model(
            model_path=args.baseline_model,
            model_name=args.baseline_name,
            tokenizer_path=args.baseline_tokenizer,
            testsets=testsets,
            args=args,
            out_dir=out_dir,
        )
        baseline_summary = make_model_summary(baseline_run_metrics, args.baseline_name)
        significance = make_significance_table(
            primary_predictions=primary_predictions,
            baseline_predictions=baseline_predictions,
            primary_run_metrics=primary_run_metrics,
            baseline_run_metrics=baseline_run_metrics,
            primary_name=args.model_name,
            baseline_name=args.baseline_name,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )

    save_tables(
        out_dir=out_dir,
        primary_summary=primary_summary,
        primary_run_metrics=primary_run_metrics,
        baseline_summary=baseline_summary,
        baseline_run_metrics=baseline_run_metrics,
        significance=significance,
    )

    print("\n[Primary summary]")
    print(dataframe_to_markdown_safe(format_summary_for_display(primary_summary)))
    if baseline_summary is not None:
        print("\n[Baseline summary]")
        print(dataframe_to_markdown_safe(format_summary_for_display(baseline_summary)))
    if significance is not None:
        print("\n[Paired significance tests]")
        print(dataframe_to_markdown_safe(significance))
    else:
        print("\n[Significance] Not computed; set --baseline_model for paired tests.")

    print(f"\n[Saved] {out_dir.resolve()}")


if __name__ == "__main__":
    main()
