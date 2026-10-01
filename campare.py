from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from safetensors import safe_open


def get_weight_map(root: Path) -> Dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(data["weight_map"])

    files = sorted(root.glob("*.safetensors"))
    if len(files) != 1:
        raise RuntimeError(
            f"Cannot determine weights in {root}: found {len(files)} safetensors files "
            "but no model.safetensors.index.json"
        )

    with safe_open(files[0], framework="pt", device="cpu") as f:
        return {key: files[0].name for key in f.keys()}


def tensor_stats(
    base: torch.Tensor,
    ft: torch.Tensor,
    chunk_elements: int,
) -> Dict[str, float | int | bool | str]:
    if base.shape != ft.shape:
        raise ValueError(f"Shape mismatch: {tuple(base.shape)} vs {tuple(ft.shape)}")

    base_flat = base.reshape(-1)
    ft_flat = ft.reshape(-1)
    n = base_flat.numel()

    diff_abs_sum = 0.0
    diff_sq_sum = 0.0
    base_sq_sum = 0.0
    changed = 0
    max_abs_diff = 0.0

    for start in range(0, n, chunk_elements):
        end = min(start + chunk_elements, n)

        # Convert only one chunk at a time to keep peak RAM low.
        b = base_flat[start:end].float()
        f = ft_flat[start:end].float()
        d = f - b
        abs_d = d.abs()

        diff_abs_sum += abs_d.double().sum().item()
        diff_sq_sum += d.double().square().sum().item()
        base_sq_sum += b.double().square().sum().item()
        changed += int((ft_flat[start:end] != base_flat[start:end]).sum().item())

        if abs_d.numel():
            max_abs_diff = max(max_abs_diff, abs_d.max().item())

        del b, f, d, abs_d

    mean_abs_diff = diff_abs_sum / n if n else 0.0
    rms_diff = math.sqrt(diff_sq_sum / n) if n else 0.0
    base_rms = math.sqrt(base_sq_sum / n) if n else 0.0
    relative_l2 = (
        math.sqrt(diff_sq_sum / base_sq_sum)
        if base_sq_sum > 0
        else (0.0 if diff_sq_sum == 0 else float("inf"))
    )

    return {
        "numel": n,
        "base_dtype": str(base.dtype),
        "ft_dtype": str(ft.dtype),
        "exactly_equal": changed == 0,
        "changed_count": changed,
        "changed_ratio": changed / n if n else 0.0,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "rms_diff": rms_diff,
        "base_rms": base_rms,
        "relative_l2": relative_l2,
        "_diff_sq_sum": diff_sq_sum,
        "_base_sq_sum": base_sq_sum,
        "_diff_abs_sum": diff_abs_sum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two Hugging Face safetensors model directories tensor by tensor."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--ft", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=Path("weight_diff_stats.csv"))
    parser.add_argument("--output_json", type=Path, default=Path("weight_diff_summary.json"))
    parser.add_argument(
        "--chunk_elements",
        type=int,
        default=4_000_000,
        help="Elements converted to float32 at once; lower this if RAM is tight.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of tensors to print for each ranking.",
    )
    args = parser.parse_args()

    if not args.base.exists():
        raise FileNotFoundError(f"Base directory does not exist: {args.base}")
    if not args.ft.exists():
        raise FileNotFoundError(f"Fine-tuned directory does not exist: {args.ft}")
    if args.chunk_elements < 1:
        raise ValueError("--chunk_elements must be positive")

    base_map = get_weight_map(args.base)
    ft_map = get_weight_map(args.ft)

    base_keys = set(base_map)
    ft_keys = set(ft_map)
    common_keys = sorted(base_keys & ft_keys)
    only_base = sorted(base_keys - ft_keys)
    only_ft = sorted(ft_keys - base_keys)

    print(f"Base tensors: {len(base_keys)}")
    print(f"FT tensors:   {len(ft_keys)}")
    print(f"Common:       {len(common_keys)}")
    print(f"Only base:    {len(only_base)}")
    print(f"Only FT:      {len(only_ft)}")

    # Open each pair of shards once instead of reopening them for every tensor.
    grouped: Dict[Tuple[str, str], list[str]] = defaultdict(list)
    for key in common_keys:
        grouped[(base_map[key], ft_map[key])].append(key)

    rows: list[dict] = []
    shape_mismatches: list[dict] = []

    total_numel = 0
    total_changed = 0
    total_diff_sq = 0.0
    total_base_sq = 0.0
    total_diff_abs = 0.0
    global_max_abs = 0.0
    equal_tensors = 0

    processed = 0
    for (base_file, ft_file), keys in grouped.items():
        base_path = args.base / base_file
        ft_path = args.ft / ft_file

        with safe_open(base_path, framework="pt", device="cpu") as base_handle, \
             safe_open(ft_path, framework="pt", device="cpu") as ft_handle:
            for key in keys:
                processed += 1
                base_tensor = base_handle.get_tensor(key)
                ft_tensor = ft_handle.get_tensor(key)

                if base_tensor.shape != ft_tensor.shape:
                    shape_mismatches.append(
                        {
                            "key": key,
                            "base_shape": list(base_tensor.shape),
                            "ft_shape": list(ft_tensor.shape),
                        }
                    )
                    print(
                        f"[{processed}/{len(common_keys)}] SHAPE MISMATCH {key}: "
                        f"{tuple(base_tensor.shape)} vs {tuple(ft_tensor.shape)}"
                    )
                    continue

                stats = tensor_stats(base_tensor, ft_tensor, args.chunk_elements)
                row = {"key": key, "shape": str(tuple(base_tensor.shape))}
                row.update({k: v for k, v in stats.items() if not k.startswith("_")})
                rows.append(row)

                total_numel += int(stats["numel"])
                total_changed += int(stats["changed_count"])
                total_diff_sq += float(stats["_diff_sq_sum"])
                total_base_sq += float(stats["_base_sq_sum"])
                total_diff_abs += float(stats["_diff_abs_sum"])
                global_max_abs = max(global_max_abs, float(stats["max_abs_diff"]))
                equal_tensors += int(bool(stats["exactly_equal"]))

                print(
                    f"[{processed}/{len(common_keys)}] {key} | "
                    f"changed={float(stats['changed_ratio']):.6%} | "
                    f"rel_l2={float(stats['relative_l2']):.6e} | "
                    f"max={float(stats['max_abs_diff']):.6e}"
                )

                del base_tensor, ft_tensor

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "key",
        "shape",
        "numel",
        "base_dtype",
        "ft_dtype",
        "exactly_equal",
        "changed_count",
        "changed_ratio",
        "max_abs_diff",
        "mean_abs_diff",
        "rms_diff",
        "base_rms",
        "relative_l2",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    global_relative_l2 = (
        math.sqrt(total_diff_sq / total_base_sq)
        if total_base_sq > 0
        else (0.0 if total_diff_sq == 0 else float("inf"))
    )
    global_rms_diff = math.sqrt(total_diff_sq / total_numel) if total_numel else 0.0
    global_base_rms = math.sqrt(total_base_sq / total_numel) if total_numel else 0.0

    summary = {
        "base_dir": str(args.base.resolve()),
        "ft_dir": str(args.ft.resolve()),
        "base_tensor_count": len(base_keys),
        "ft_tensor_count": len(ft_keys),
        "common_tensor_count": len(common_keys),
        "compared_tensor_count": len(rows),
        "exactly_equal_tensor_count": equal_tensors,
        "only_base_keys": only_base,
        "only_ft_keys": only_ft,
        "shape_mismatches": shape_mismatches,
        "total_numel": total_numel,
        "global_changed_count": total_changed,
        "global_changed_ratio": total_changed / total_numel if total_numel else 0.0,
        "global_max_abs_diff": global_max_abs,
        "global_mean_abs_diff": total_diff_abs / total_numel if total_numel else 0.0,
        "global_rms_diff": global_rms_diff,
        "global_base_rms": global_base_rms,
        "global_relative_l2": global_relative_l2,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    top_k = max(1, args.top_k)
    top_relative = sorted(rows, key=lambda r: float(r["relative_l2"]), reverse=True)[:top_k]
    top_max = sorted(rows, key=lambda r: float(r["max_abs_diff"]), reverse=True)[:top_k]
    top_changed = sorted(rows, key=lambda r: float(r["changed_ratio"]), reverse=True)[:top_k]

    print("\n=== WHOLE-MODEL SUMMARY ===")
    print(f"Compared tensors:       {len(rows)}")
    print(f"Exactly equal tensors: {equal_tensors}")
    print(f"Total parameters:       {total_numel:,}")
    print(f"Changed parameters:     {total_changed:,}")
    print(f"Global changed ratio:   {summary['global_changed_ratio']:.6%}")
    print(f"Global max abs diff:    {global_max_abs:.9g}")
    print(f"Global mean abs diff:   {summary['global_mean_abs_diff']:.9g}")
    print(f"Global RMS diff:        {global_rms_diff:.9g}")
    print(f"Global base RMS:        {global_base_rms:.9g}")
    print(f"Global relative L2:     {global_relative_l2:.9g}")

    def print_top(title: str, selected: Iterable[dict], metric: str) -> None:
        print(f"\n=== {title} ===")
        for row in selected:
            print(
                f"{float(row[metric]):.9g}\t"
                f"changed={float(row['changed_ratio']):.4%}\t"
                f"{row['key']}"
            )

    print_top("TOP RELATIVE L2", top_relative, "relative_l2")
    print_top("TOP MAX ABS DIFF", top_max, "max_abs_diff")
    print_top("TOP CHANGED RATIO", top_changed, "changed_ratio")

    print(f"\nPer-tensor CSV: {args.output_csv.resolve()}")
    print(f"Summary JSON:  {args.output_json.resolve()}")


if __name__ == "__main__":
    main()