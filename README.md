# Conway's Game of Life LLM Benchmark

A benchmark for testing LLM spatial reasoning capabilities using Conway's Game of Life. The benchmark presents grid states as ASCII text, asks the LLM to compute the next generation, and evaluates accuracy across multiple difficulty levels.

## Table of Contents

- [Overview](#overview)
- [Quick Example](#quick-example)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Methodology](#methodology)
- [Scoring Metrics](#scoring-metrics)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Extending the Benchmark](#extending-the-benchmark)

---

## Overview

This benchmark evaluates LLM reasoning capabilities through a classic computational puzzle. Given an ASCII representation of a Game of Life grid, the LLM must apply Conway's rules to compute the next generation.

**What it tests:**
- Spatial pattern recognition
- Consistent rule application across a grid
- Edge case handling (boundaries, neighbor counting)
- Output formatting accuracy

**Key features:**
- Reproducible test cases via fixed random seeds
- Multiple difficulty levels (3x3 to 10x10 grids)
- Comprehensive scoring (accuracy, F1-based correctness, perfect matches)
- Cost and token tracking via OpenRouter
- Multi-model comparison tooling

---

## Quick Example

**Input (given to LLM):**
```
.#.
##.
.#.
```

**Expected output:**
```
##.
###
##.
```

The LLM must recognize that each `#` is an alive cell and `.` is dead, then apply Conway's rules:
- A live cell survives with 2-3 neighbors
- A dead cell becomes alive with exactly 3 neighbors
- All other cells die or stay dead

---

## Getting Started

### Prerequisites

- Python 3.8+
- An [OpenRouter](https://openrouter.ai/) API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/himansharma21/Conway-bench.git
cd Conway-bench
```

2. Set up a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your API key by creating `config.json`:
```json
{
  "openrouter": {
    "api_key": "your-api-key-here",
    "model": "anthropic/claude-sonnet-4",
    "temperature": 0.0,
    "max_tokens": 1000
  }
}
```

### Verify Installation

Run the unit tests to ensure everything works:
```bash
pytest test_conway.py -v
```

---

## Usage

### Interactive CLI

The easiest way to explore the benchmark:

```bash
python main.py
```

Options:
1. **Run single test** - Choose difficulty and seed
2. **Run simple benchmark** - Full 9-test suite
3. **Preview test case** - See a test without calling the LLM
4. **Run advanced benchmark** - Custom tests from a file
5. **Show configuration** - Verify your settings

### Run the Standard Benchmark

```bash
python benchmark.py
```

This runs 9 tests across difficulty levels:
- 2 Easy (3x3)
- 3 Medium (5x5)
- 2 Hard (8x8)
- 2 Expert (10x10)

Results are saved to `results.json`.

### Compare Multiple Models

```bash
python compare_models.py models.txt advanced_tests.txt --out results.csv
```

Where:
- `models.txt` - One model identifier per line
- `advanced_tests.txt` - Test definitions (see Advanced Tests below)
- `--runs N` - Run each test N times for statistical variance (optional)

Output CSV includes: solved tests, token usage, cost, points, and execution time.

### Advanced Tests

Create a text file with custom test cases:

```
4 0.5
6 0.25
10 0.3
```

Format: `<grid_size> <density>` where:
- `grid_size` - Integer (e.g., `4` creates a 4x4 grid)
- `density` - Probability of a cell being alive (0.0 to 1.0)

Run with:
```bash
python benchmark.py --advanced advanced_tests.txt
```

Or use the interactive CLI option 4.

---

## Methodology

### Approach

The benchmark tests spatial reasoning through a well-defined, deterministic task:

1. **Generate** a random initial board using a fixed seed for reproducibility
2. **Compute** the expected next state using Conway's rules
3. **Prompt** the LLM with the ASCII board and clear rule instructions
4. **Extract** the predicted board from the LLM response
5. **Score** the prediction against the expected output

### Conway's Rules

For each cell, count the 8 neighboring cells (including diagonals):

| Current State | Neighbor Count | Next State |
|---------------|----------------|------------|
| Alive | 2-3 | Alive (survives) |
| Alive | <2 or >3 | Dead (dies) |
| Dead | 3 | Alive (birth) |
| Dead | Not 3 | Dead (stays dead) |

**Edge handling:** Cells outside the grid boundary are treated as dead.

### Prompt Design

The prompt:
- Presents the board in a code block for clarity
- States the rules explicitly
- Encourages step-by-step reasoning
- Requests the final answer in a code block

The response extractor takes the **last** code block found, allowing models to show their reasoning without interference.

### Difficulty Levels

| Difficulty | Grid Size | Cells | Test Seeds |
|------------|-----------|-------|------------|
| Easy | 3x3 | 9 | 42, 43 |
| Medium | 5x5 | 25 | 42, 43, 44 |
| Hard | 8x8 | 64 | 42, 43 |
| Expert | 10x10 | 100 | 42, 43 |

---

## Scoring Metrics

### Cell Accuracy

Simple percentage of correctly predicted cells:

```
accuracy = correct_cells / total_cells
```

Easy to understand but doesn't account for class imbalance (boards are typically ~30% alive).

### Correctness Score (Primary Metric)

Geometric mean of F1 scores for alive and dead cells:

```
correctness = sqrt(F1_alive × F1_dead)
```

**Why this metric?**
- Handles class imbalance fairly
- Prevents gaming (predicting all-dead gives F1_alive=0, thus correctness=0)
- Provides continuous scoring to reduce variance

### Perfect Match

Binary score: 1 if the predicted board exactly matches expected, 0 otherwise.

### Points

```
points = correctness × grid_size²
```

Example: A 5x5 grid with 80% correctness earns 20 points out of 25.

### Cost and Tokens

Tracked per test and summed for the run:
- Prompt tokens
- Completion tokens
- API cost (when reported by OpenRouter)
- Response time

---

## Configuration

The `config.json` file controls LLM settings:

```json
{
  "openrouter": {
    "api_key": "sk-or-v1-...",
    "model": "google/gemini-2.5-flash",
    "temperature": 0.0,
    "max_tokens": 6000,
    "reasoning_effort": "high"
  }
}
```

| Setting | Description | Recommended |
|---------|-------------|-------------|
| `api_key` | Your OpenRouter API key | Required |
| `model` | Model identifier | See OpenRouter models |
| `temperature` | Sampling temperature | 0.0 for determinism |
| `max_tokens` | Maximum response tokens | 1000-6000 |
| `reasoning_effort` | Extended thinking: "high", "medium", "low" | Optional |

**Note:** `config.json` is gitignored to protect API keys.

### Available Models

Any model on [OpenRouter](https://openrouter.ai/models) works. Examples:
- `anthropic/claude-sonnet-4`
- `anthropic/claude-opus-4`
- `google/gemini-2.5-flash`
- `openai/gpt-4o`

---

## Project Structure

```
Conway-bench/
├── conway.py            # Core Game of Life logic
├── api.py               # LLM provider abstraction (OpenRouter)
├── benchmark.py         # Test runner and scoring engine
├── compare_models.py    # Multi-model comparison tool
├── main.py              # Interactive CLI
├── test_conway.py       # Unit tests (25+ tests)
├── requirements.txt     # Python dependencies
├── config.json          # API configuration (gitignored)
├── results.json         # Benchmark output (generated)
├── AGENTS.md            # Implementation documentation
└── README.md            # This file
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `conway.py` | Game of Life implementation: `next_state()`, board conversions, accuracy calculations |
| `api.py` | LLM abstraction: `OpenRouterProvider`, config loading, response handling |
| `benchmark.py` | Benchmark execution: prompt building, response parsing, test orchestration |
| `compare_models.py` | Multi-model evaluation with CSV output |
| `main.py` | User-friendly interactive menu |

---

## Extending the Benchmark

### Adding a New LLM Provider

1. Create a new class in `api.py` inheriting from `LLMProvider`
2. Implement `query(prompt)` and `list_models()` methods
3. Add to the `providers` dict in `create_provider()`

### Modifying the Prompt

Edit `build_prompt()` in `benchmark.py`. The prompt must:
- Present the board clearly (code block recommended)
- State the rules
- Request output in a code block (extractor depends on this)

### Changing Test Cases

Modify the `test_cases` list in `run_benchmark()` in `benchmark.py`.

Format: `(rows, cols, difficulty_label, seed)`

### Adjusting Extended Thinking

Set `reasoning_effort` in `config.json`:
- `"high"` - Maximum reasoning (180s timeout)
- `"medium"` - Moderate reasoning
- `"low"` - Light reasoning
- Remove the field for no extended thinking

---

## License

See repository for license information.
