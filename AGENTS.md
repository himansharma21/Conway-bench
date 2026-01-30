# AGENTS.md - Conway's Game of Life LLM Benchmark

## Project Overview

This is a benchmark for testing LLM spatial reasoning capabilities using Conway's Game of Life. The benchmark presents grid states as ASCII text, asks the LLM to compute the next generation, and scores the accuracy of the response.

## File Structure

```
Conway-bench/
├── conway.py          # Core game logic (next_state, board conversions)
├── api.py             # LLM API abstraction layer (OpenRouter)
├── benchmark.py       # Test runner, prompts, and scoring
├── compare_models.py  # Multi-model comparison runner
├── main.py            # Interactive CLI
├── config.json        # API keys and model settings (gitignored)
├── test_conway.py     # Unit tests for game logic
└── results.json       # Benchmark output (generated)
```

## Key Components

### conway.py
Core Conway's Game of Life implementation:
- `next_state(board)` - Computes next generation given a 2D numpy array
- `board_to_ascii(board)` - Converts numpy array to ASCII string (`#` = alive, `.` = dead)
- `ascii_to_board(ascii_str)` - Parses ASCII string back to numpy array
- `generate_random_board(rows, cols, density, seed)` - Creates random initial states
- `calculate_accuracy(predicted, expected)` - Returns float 0.0-1.0 (simple cell accuracy)
- `is_perfect_match(predicted, expected)` - Returns bool
- `calculate_correctness(predicted, expected)` - Returns float 0.0-1.0 (geometric mean of F1_alive and F1_dead)

### api.py
LLM provider abstraction:
- `LLMConfig` - Dataclass with: api_key, model, temperature, max_tokens, reasoning_effort
- `LLMResponse` - Dataclass with: content, model, response_time, cost, error
- `OpenRouterProvider` - Implements API calls to OpenRouter
- `load_config(path)` - Loads config.json
- `create_provider(config)` - Factory function for providers

**Important**: The `reasoning_effort` parameter ("high", "medium", "low") enables extended thinking for supported models. When set, timeout increases to 180s.

### benchmark.py
Test execution and scoring:
- `build_prompt(board_ascii)` - Constructs the LLM prompt with rules and instructions
- `extract_board_from_response(response)` - Parses LLM output, extracts from code blocks
- `run_single_test(rows, cols, difficulty, seed, provider)` - Runs one test case
- `run_benchmark(config_path, output_path)` - Runs full simple test suite (9 cases)
- `run_advanced_benchmark(tests_path, config_path, output_path)` - Runs advanced tests from a txt file
- `print_detailed_results(result)` - Displays results with diff on failures

### compare_models.py
Multi-model comparison:
- Reads a model list file (one model per line)
- Reads an advanced tests file (`<grid_size> <density>`)
- Runs all models against all tests
- Writes a CSV with solved tests, token totals, cost total, points, and time

**Prompt design**: The prompt encourages step-by-step reasoning and requests the final board in a code block. The extractor takes the LAST code block found, so reasoning output doesn't interfere.

### main.py
Interactive menu-driven CLI:
1. Run simple test (choose difficulty + seed)
2. Run simple benchmark
3. Preview test case (no LLM call)
4. Run advanced benchmark (from txt file)
5. Show configuration
6. Exit

## Configuration

`config.json` structure:
```json
{
  "openrouter": {
    "api_key": "sk-or-v1-...",
    "model": "google/gemini-3-flash-preview",
    "temperature": 0.7,
    "max_tokens": 6000,
    "reasoning_effort": "high"
  }
}
```

**Note**: config.json is gitignored to protect API keys.

## Test Cases

| Difficulty | Grid Size | Seeds |
|------------|-----------|-------|
| Easy       | 3x3       | 42, 43 |
| Medium     | 5x5       | 42, 43, 44 |
| Hard       | 8x8       | 42, 43 |
| Expert     | 10x10     | 42, 43 |

### Advanced Tests

Advanced tests are loaded from a text file, one test per line:

```
<grid_size> <density>
```

Example:
```
4 0.5
6 0.25
10 0.3
```

## Running the Benchmark

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add API key to config.json

# Interactive mode
python main.py

# Direct benchmark run
python benchmark.py
```

## Common Tasks

### Adding a new LLM provider
1. Create a new class in `api.py` inheriting from `LLMProvider`
2. Implement `query(prompt)` and `list_models()` methods
3. Add to the `providers` dict in `create_provider()`

### Modifying the prompt
Edit `build_prompt()` in `benchmark.py`. Key considerations:
- Board is wrapped in code blocks for clarity
- Prompt encourages reasoning before final answer
- Final answer must be in a code block (extraction depends on this)

### Changing test cases
Modify the `test_cases` list in `run_benchmark()` in `benchmark.py`. Format: `(rows, cols, difficulty_label, seed)`

### Adjusting reasoning/thinking
Set `reasoning_effort` in config.json to "high", "medium", "low", or remove for no reasoning. This uses OpenRouter's reasoning tokens feature.

## Key Implementation Details

1. **Board representation**: Numpy 2D arrays where 1=alive, 0=dead
2. **Edge handling**: Cells outside grid boundaries are treated as dead
3. **Response parsing**: `extract_board_from_response()` tries:
   - Code blocks (regex, takes LAST match)
   - Lines containing only `.` and `#`
   - Falls back to full response
4. **Scoring**:
   - `cell_accuracy`: Simple percentage of cells correctly predicted
   - `correctness`: Geometric mean of F1_alive and F1_dead (see below)
   - `perfect_match`: Boolean for exact match
5. **Points**: `correctness * grid_size²` (continuous scoring, not binary)
6. **Cost**: Per-test cost from OpenRouter response (when available), summed across the run
7. **Seeds**: Fixed seeds ensure reproducible test cases across runs

## Correctness Scoring

The `correctness` metric uses the geometric mean of F1 scores for alive and dead cells:

```
correctness = sqrt(F1_alive * F1_dead)
```

This approach:
- Handles class imbalance (typically ~30% alive, ~70% dead)
- Penalizes models that exploit imbalance (e.g., predicting all-dead gives F1_alive=0, thus correctness=0)
- Provides continuous scoring to reduce run-to-run variance compared to binary perfect-match scoring

Edge cases:
- If expected has no alive cells: F1_alive = 1.0 if predicted also has none, else 0.0
- If expected has no dead cells: F1_dead = 1.0 if predicted also has none, else 0.0
