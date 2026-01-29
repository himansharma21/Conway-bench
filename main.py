#!/usr/bin/env python3
"""
Interactive CLI for the Conway's Game of Life LLM Benchmark.
"""

import os
import sys

from api import load_config, create_provider
from benchmark import (
    run_benchmark,
    run_advanced_benchmark,
    load_advanced_test_cases,
    run_single_test,
    build_prompt,
    extract_board_from_response,
    print_detailed_results,
)
from conway import (
    generate_random_board,
    board_to_ascii,
    next_state,
)


# Difficulty configurations: (rows, cols)
DIFFICULTIES = {
    "1": ("Easy", 3, 3),
    "2": ("Medium", 5, 5),
    "3": ("Hard", 8, 8),
    "4": ("Expert", 10, 10),
}


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    """Print the application header."""
    print("=" * 60)
    print("  Conway's Game of Life - LLM Spatial Reasoning Benchmark")
    print("=" * 60)
    print()


def print_menu():
    """Print the main menu."""
    print("Main Menu:")
    print("-" * 40)
    print("  1. Run simple test")
    print("  2. Run simple benchmark")
    print("  3. Preview a test case (no LLM)")
    print("  4. Run advanced benchmark (from txt file)")
    print("  5. Show current configuration")
    print("  6. Exit")
    print("-" * 40)


def get_difficulty_choice() -> tuple[str, int, int]:
    """Get difficulty selection from user."""
    print("\nSelect difficulty:")
    print("  1. Easy (3x3)")
    print("  2. Medium (5x5)")
    print("  3. Hard (8x8)")
    print("  4. Expert (10x10)")

    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        if choice in DIFFICULTIES:
            return DIFFICULTIES[choice]
        print("Invalid choice. Please enter 1-4.")


def get_seed() -> int:
    """Get seed value from user."""
    while True:
        seed_input = input("Enter seed (default 42): ").strip()
        if not seed_input:
            return 42
        try:
            return int(seed_input)
        except ValueError:
            print("Invalid seed. Please enter an integer.")


def run_single_test_interactive():
    """Run a single test with user-selected parameters."""
    print("\n--- Simple Test ---\n")

    # Check config
    if not os.path.exists("config.json"):
        print("Error: config.json not found.")
        input("\nPress Enter to continue...")
        return

    try:
        config = load_config()
        if not config.api_key:
            print("Error: API key not set in config.json.")
            input("\nPress Enter to continue...")
            return
        provider = create_provider(config)
    except Exception as e:
        print(f"Error loading config: {e}")
        input("\nPress Enter to continue...")
        return

    # Get test parameters
    difficulty, rows, cols = get_difficulty_choice()
    seed = get_seed()

    print(f"\nRunning {difficulty} test ({rows}x{cols}) with seed {seed}...")
    print(f"Model: {config.model}")
    print("-" * 40)

    # Run the test
    result = run_single_test(rows, cols, difficulty, seed, provider)

    # Display results
    print(f"\nResults:")
    print(f"  Cell accuracy: {result.cell_accuracy:.2%}")
    print(f"  Perfect match: {'Yes' if result.perfect_match else 'No'}")
    print(f"  Response time: {result.response_time:.2f}s")

    print(f"\nInitial board:")
    for line in result.initial_board.split("\n"):
        print(f"  {line}")

    print(f"\nExpected next state:")
    for line in result.expected_board.split("\n"):
        print(f"  {line}")

    print(f"\nLLM predicted:")
    for line in result.predicted_board.split("\n"):
        print(f"  {line}")

    if not result.perfect_match:
        print("\nMismatches marked with *:")
        expected_lines = result.expected_board.split("\n")
        predicted_lines = result.predicted_board.split("\n")
        for i, (exp, pred) in enumerate(zip(expected_lines, predicted_lines)):
            diff_line = ""
            for j, (e, p) in enumerate(zip(exp, pred)):
                diff_line += "*" if e != p else "."
            print(f"  {diff_line}")

    input("\nPress Enter to continue...")


def run_full_benchmark_interactive():
    """Run the full benchmark suite."""
    print("\n--- Simple Benchmark ---\n")

    # Check config
    if not os.path.exists("config.json"):
        print("Error: config.json not found.")
        input("\nPress Enter to continue...")
        return

    try:
        config = load_config()
        if not config.api_key:
            print("Error: API key not set in config.json.")
            input("\nPress Enter to continue...")
            return
    except Exception as e:
        print(f"Error loading config: {e}")
        input("\nPress Enter to continue...")
        return

    print(f"This will run 9 test cases across all difficulty levels.")
    print(f"Model: {config.model}")
    confirm = input("\nProceed? (y/n): ").strip().lower()

    if confirm != "y":
        print("Cancelled.")
        input("\nPress Enter to continue...")
        return

    print()
    result = run_benchmark()
    print_detailed_results(result)

    input("\nPress Enter to continue...")


def run_advanced_benchmark_interactive():
    """Run the advanced benchmark suite from a text file."""
    print("\n--- Advanced Benchmark ---\n")

    # Check config
    if not os.path.exists("config.json"):
        print("Error: config.json not found.")
        input("\nPress Enter to continue...")
        return

    try:
        config = load_config()
        if not config.api_key:
            print("Error: API key not set in config.json.")
            input("\nPress Enter to continue...")
            return
    except Exception as e:
        print(f"Error loading config: {e}")
        input("\nPress Enter to continue...")
        return

    tests_path = input("Enter path to advanced tests txt file: ").strip()
    if not tests_path:
        print("No file provided.")
        input("\nPress Enter to continue...")
        return
    if not os.path.exists(tests_path):
        print("File not found.")
        input("\nPress Enter to continue...")
        return

    try:
        test_cases = load_advanced_test_cases(tests_path)
    except Exception as e:
        print(f"Error loading advanced tests: {e}")
        input("\nPress Enter to continue...")
        return

    print("\nTest summary:")
    total_points = 0
    for idx, (size, density) in enumerate(test_cases, 1):
        points = size * size
        total_points += points
        print(f"  {idx}. size={size}x{size}, density={density}, points={points}")
    print(f"Total possible points: {total_points}")

    print(f"Model: {config.model}")
    confirm = input("\nProceed? (y/n): ").strip().lower()

    if confirm != "y":
        print("Cancelled.")
        input("\nPress Enter to continue...")
        return

    print()
    try:
        result = run_advanced_benchmark(tests_path=tests_path, show_summary=False)
    except Exception as e:
        print(f"Error running advanced benchmark: {e}")
        input("\nPress Enter to continue...")
        return

    print_detailed_results(result)

    input("\nPress Enter to continue...")


def preview_test_case():
    """Preview a test case without calling the LLM."""
    print("\n--- Preview Test Case ---\n")

    difficulty, rows, cols = get_difficulty_choice()
    seed = get_seed()

    # Generate board
    board = generate_random_board(rows, cols, seed=seed)
    board_ascii = board_to_ascii(board)

    # Compute next state
    expected = next_state(board)
    expected_ascii = board_to_ascii(expected)

    print(f"\n{difficulty} ({rows}x{cols}), seed={seed}")
    print("-" * 40)

    print(f"\nInitial board:")
    for line in board_ascii.split("\n"):
        print(f"  {line}")

    print(f"\nExpected next state:")
    for line in expected_ascii.split("\n"):
        print(f"  {line}")

    print(f"\nPrompt that would be sent to LLM:")
    print("-" * 40)
    prompt = build_prompt(board_ascii)
    print(prompt)

    input("\nPress Enter to continue...")


def show_configuration():
    """Show current configuration."""
    print("\n--- Current Configuration ---\n")

    if not os.path.exists("config.json"):
        print("Error: config.json not found.")
        print("\nCreate a config.json file with the following structure:")
        print('''
{
  "openrouter": {
    "api_key": "your-api-key-here",
    "model": "anthropic/claude-3.5-sonnet",
    "temperature": 0.0,
    "max_tokens": 1000
  }
}
''')
        input("\nPress Enter to continue...")
        return

    try:
        config = load_config()
        print(f"API Key: {'*' * 20 + config.api_key[-4:] if config.api_key else 'NOT SET'}")
        print(f"Model: {config.model}")
        print(f"Temperature: {config.temperature}")
        print(f"Max Tokens: {config.max_tokens}")

        if not config.api_key:
            print("\nWarning: API key is not set. Add your OpenRouter API key to config.json.")

    except Exception as e:
        print(f"Error loading config: {e}")

    input("\nPress Enter to continue...")


def main():
    """Main entry point for the interactive CLI."""
    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            run_single_test_interactive()
        elif choice == "2":
            run_full_benchmark_interactive()
        elif choice == "3":
            preview_test_case()
        elif choice == "4":
            run_advanced_benchmark_interactive()
        elif choice == "5":
            show_configuration()
        elif choice == "6":
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please enter 1-6.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
