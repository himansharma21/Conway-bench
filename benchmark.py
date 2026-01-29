"""
LLM Benchmark runner for Conway's Game of Life.
Uses OpenRouter API to test LLM spatial reasoning capabilities.
"""

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List

import numpy as np

from conway import (
    next_state,
    board_to_ascii,
    ascii_to_board,
    generate_random_board,
    calculate_accuracy,
    is_perfect_match,
)
from api import LLMProvider, load_config, create_provider


@dataclass
class TestResult:
    """Result of a single test case."""
    difficulty: str
    grid_size: str
    seed: int
    initial_board: str
    expected_board: str
    predicted_board: str
    cell_accuracy: float
    perfect_match: bool
    response_time: float
    raw_response: str


@dataclass
class BenchmarkResult:
    """Overall benchmark results."""
    model: str
    timestamp: str
    results: List[TestResult]
    overall_accuracy: float
    perfect_matches: int
    total_tests: int


def build_prompt(board_ascii: str) -> str:
    """
    Build the prompt for the LLM.

    Args:
        board_ascii: ASCII representation of the initial board state

    Returns:
        The complete prompt string
    """
    return f"""You are playing Conway's Game of Life. Given the current board state below, compute the next generation.

Rules:
- Any live cell (#) with 2-3 live neighbors survives
- Any dead cell (.) with exactly 3 live neighbors becomes alive
- All other cells die or stay dead
- Neighbors are the 8 adjacent cells (horizontal, vertical, and diagonal)
- Cells outside the grid boundaries are considered dead

Current board state:
```
{board_ascii}
```

Think through this carefully. For each cell, count its live neighbors and apply the rules.

After your reasoning, output ONLY the final board in a code block like this:
```
<your board here>
```
Use '#' for live cells and '.' for dead cells. The board must be the same dimensions as the input."""


def extract_board_from_response(response: str) -> str:
    """
    Extract the ASCII board from the LLM response.

    Args:
        response: Raw LLM response

    Returns:
        Extracted ASCII board string
    """
    # Try to extract from code blocks
    code_block_pattern = r"```(?:\w*\n)?(.*?)```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try to find lines containing only . and #
    lines = response.split("\n")
    board_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and all(c in ".#" for c in stripped):
            board_lines.append(stripped)

    if board_lines:
        return "\n".join(board_lines)

    # Return the whole response as fallback
    return response.strip()


def run_single_test(
    rows: int,
    cols: int,
    difficulty: str,
    seed: int,
    provider: LLMProvider,
) -> TestResult:
    """
    Run a single test case.

    Args:
        rows: Number of rows in the board
        cols: Number of columns in the board
        difficulty: Difficulty label for this test
        seed: Random seed for reproducibility
        provider: LLM provider for querying

    Returns:
        TestResult with all metrics
    """
    # Generate board
    board = generate_random_board(rows, cols, seed=seed)
    board_ascii = board_to_ascii(board)

    # Compute expected next state
    expected = next_state(board)
    expected_ascii = board_to_ascii(expected)

    # Build prompt and query LLM
    prompt = build_prompt(board_ascii)
    response = provider.query(prompt)

    if response.error:
        raw_response = f"ERROR: {response.error}"
    else:
        raw_response = response.content
    response_time = response.response_time

    # Extract predicted board
    predicted_ascii = extract_board_from_response(raw_response)
    try:
        predicted = ascii_to_board(predicted_ascii)
    except Exception:
        predicted = np.zeros_like(expected)

    # Calculate metrics
    accuracy = calculate_accuracy(predicted, expected)
    perfect = is_perfect_match(predicted, expected)

    return TestResult(
        difficulty=difficulty,
        grid_size=f"{rows}x{cols}",
        seed=seed,
        initial_board=board_ascii,
        expected_board=expected_ascii,
        predicted_board=predicted_ascii,
        cell_accuracy=accuracy,
        perfect_match=perfect,
        response_time=response_time,
        raw_response=raw_response,
    )


def run_benchmark(
    config_path: str = "config.json",
    output_path: str = "results.json",
) -> BenchmarkResult:
    """
    Run the full benchmark suite.

    Args:
        config_path: Path to the configuration file
        output_path: Path to save results JSON

    Returns:
        BenchmarkResult with all test results
    """
    config = load_config(config_path)
    provider = create_provider(config)

    test_cases = [
        (3, 3, "Easy", 42),
        (3, 3, "Easy", 43),
        (5, 5, "Medium", 42),
        (5, 5, "Medium", 43),
        (5, 5, "Medium", 44),
        (8, 8, "Hard", 42),
        (8, 8, "Hard", 43),
        (10, 10, "Expert", 42),
        (10, 10, "Expert", 43),
    ]

    results = []

    print(f"Running benchmark with model: {config.model}")
    print(f"Test cases: {len(test_cases)}")
    print("-" * 50)

    for rows, cols, difficulty, seed in test_cases:
        print(f"Running {difficulty} test ({rows}x{cols}, seed={seed})...", end=" ")

        result = run_single_test(rows, cols, difficulty, seed, provider)
        results.append(result)

        status = "✓" if result.perfect_match else "✗"
        print(f"{status} accuracy={result.cell_accuracy:.2%}, time={result.response_time:.2f}s")

    # Calculate overall metrics
    overall_accuracy = sum(r.cell_accuracy for r in results) / len(results)
    perfect_matches = sum(1 for r in results if r.perfect_match)

    benchmark_result = BenchmarkResult(
        model=config.model,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        results=results,
        overall_accuracy=overall_accuracy,
        perfect_matches=perfect_matches,
        total_tests=len(results),
    )

    # Save results
    save_results(benchmark_result, output_path)

    print("-" * 50)
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    print(f"Perfect matches: {perfect_matches}/{len(results)}")
    print(f"Results saved to: {output_path}")

    return benchmark_result


def save_results(result: BenchmarkResult, output_path: str) -> None:
    """
    Save benchmark results to a JSON file.

    Args:
        result: BenchmarkResult to save
        output_path: Path to save the results
    """
    # Convert to dict for JSON serialization
    data = asdict(result)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def print_detailed_results(result: BenchmarkResult) -> None:
    """
    Print detailed results to the console.

    Args:
        result: BenchmarkResult to display
    """
    print("\n" + "=" * 70)
    print(f"DETAILED RESULTS - Model: {result.model}")
    print(f"Timestamp: {result.timestamp}")
    print("=" * 70)

    for i, r in enumerate(result.results, 1):
        print(f"\nTest {i}: {r.difficulty} ({r.grid_size}, seed={r.seed})")
        print(f"  Cell accuracy: {r.cell_accuracy:.2%}")
        print(f"  Perfect match: {r.perfect_match}")
        print(f"  Response time: {r.response_time:.2f}s")

        if not r.perfect_match:
            print(f"\n  Initial state:")
            for line in r.initial_board.split("\n"):
                print(f"    {line}")
            print(f"\n  Expected:")
            for line in r.expected_board.split("\n"):
                print(f"    {line}")
            print(f"\n  Predicted:")
            for line in r.predicted_board.split("\n"):
                print(f"    {line}")


if __name__ == "__main__":
    import sys

    # Check for API key
    if not os.path.exists("config.json"):
        print("Error: config.json not found. Please create it with your OpenRouter API key.")
        sys.exit(1)

    config = load_config()
    if not config.api_key:
        print("Error: API key not set in config.json. Please add your OpenRouter API key.")
        sys.exit(1)

    # Run benchmark
    result = run_benchmark()

    # Print detailed results
    print_detailed_results(result)
