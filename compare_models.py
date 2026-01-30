#!/usr/bin/env python3
"""
Run advanced benchmark tests across multiple models and write a CSV summary.
"""

import argparse
import csv
import os
import time
from typing import List, Tuple

from api import load_config, create_provider, LLMConfig
from benchmark import load_advanced_test_cases, run_single_test


def load_models(path: str) -> List[str]:
    models: List[str] = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            models.append(stripped)
    if not models:
        raise ValueError("No models found in model list file.")
    return models


def format_test_label(index: int, size: int, density: float) -> str:
    return f"{index}:{size}x{size}@{density}"


def run_model(
    model: str,
    test_cases: List[Tuple[int, float]],
    base_config: LLMConfig,
) -> dict:
    config = LLMConfig(
        api_key=base_config.api_key,
        model=model,
        temperature=base_config.temperature,
        max_tokens=base_config.max_tokens,
        reasoning_effort=base_config.reasoning_effort,
    )
    provider = create_provider(config)

    correct_tests: List[str] = []
    points_earned = 0
    max_points = 0
    total_cost = 0.0
    completion_tokens_total = 0
    total_tokens_total = 0

    start_time = time.time()
    for idx, (size, density) in enumerate(test_cases, 1):
        seed = 42 + idx
        result = run_single_test(
            size,
            size,
            "Advanced",
            seed,
            provider,
            density=density,
            test_type="Advanced",
        )
        label = format_test_label(idx, size, density)
        if result.perfect_match:
            correct_tests.append(label)
        points_earned += result.points_awarded
        max_points += result.max_points
        total_cost += result.cost
        completion_tokens_total += result.completion_tokens
        total_tokens_total += result.total_tokens

    elapsed = time.time() - start_time

    return {
        "model": model,
        "correct_tests": ";".join(correct_tests),
        "points_earned": points_earned,
        "max_points": max_points,
        "completion_tokens_total": completion_tokens_total,
        "total_tokens_total": total_tokens_total,
        "total_cost": total_cost,
        "time_seconds": elapsed,
    }


def generate_output_path(base_path: str, run_number: int, total_runs: int) -> str:
    """Generate output path for a specific run."""
    if total_runs == 1:
        return base_path
    # Split base path into name and extension
    if "." in base_path:
        name, ext = base_path.rsplit(".", 1)
        return f"{name}_run{run_number}.{ext}"
    return f"{base_path}_run{run_number}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models on advanced tests.")
    parser.add_argument("models_file", help="Path to a text file of model IDs (one per line).")
    parser.add_argument("tests_file", help="Path to advanced tests txt file.")
    parser.add_argument(
        "--out",
        default="model_comparison.csv",
        help="Output CSV base path (default: model_comparison.csv)",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run the full test suite (default: 1). Each run produces a separate CSV.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.models_file):
        raise SystemExit(f"Models file not found: {args.models_file}")
    if not os.path.exists(args.tests_file):
        raise SystemExit(f"Tests file not found: {args.tests_file}")
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    base_config = load_config(args.config)
    if not base_config.api_key:
        raise SystemExit("API key not set in config.json.")

    models = load_models(args.models_file)
    test_cases = load_advanced_test_cases(args.tests_file)

    for run_num in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n{'='*50}")
            print(f"RUN {run_num}/{args.runs}")
            print(f"{'='*50}")

        rows = []
        for model in models:
            print(f"Running model: {model}")
            rows.append(run_model(model, test_cases, base_config))

        output_path = generate_output_path(args.out, run_num, args.runs)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "correct_tests",
                    "completion_tokens_total",
                    "total_tokens_total",
                    "total_cost",
                    "points_earned",
                    "max_points",
                    "time_seconds",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved CSV to: {output_path}")

    if args.runs > 1:
        print(f"\nCompleted {args.runs} runs.")


if __name__ == "__main__":
    main()
