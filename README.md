# Conway's Game of Life LLM Benchmark

A benchmark for testing LLM spatial reasoning using Conway's Game of Life.

## What it does

Presents grid states as ASCII text and asks LLMs to compute the next generation. Tests pattern recognition, rule application, and spatial reasoning capabilities.

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
##.
##.
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your OpenRouter API key:
   - Copy `config.json` and add your API key:
   ```json
   {
     "openrouter": {
       "api_key": "your-api-key-here",
       "model": "anthropic/claude-3.5-sonnet",
       "temperature": 0.0,
       "max_tokens": 1000
     }
   }
   ```

## Usage

### Run the Benchmark

```bash
python benchmark.py
```

This will:
1. Load LLM configuration from `config.json`
2. Run test cases across multiple difficulty levels (Easy, Medium, Hard, Expert)
3. Display results with cell accuracy and response times
4. Save detailed results to `results.json`

### Run Unit Tests

```bash
pytest test_conway.py -v
```

## Configuration

The `config.json` file contains all LLM settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `api_key` | Your OpenRouter API key | (empty) |
| `model` | Model identifier | `anthropic/claude-3.5-sonnet` |
| `temperature` | Sampling temperature | `0.0` |
| `max_tokens` | Maximum response tokens | `1000` |

### Available Models

You can use any model available on OpenRouter. Some options:
- `anthropic/claude-3.5-sonnet`
- `anthropic/claude-3-opus`
- `openai/gpt-4o`
- `openai/gpt-4-turbo`
- `google/gemini-pro`
- `meta-llama/llama-3.1-70b-instruct`

See [OpenRouter models](https://openrouter.ai/models) for the full list.

## Test Cases

The benchmark includes test cases of varying difficulty:

| Difficulty | Grid Size | Description |
|------------|-----------|-------------|
| Easy | 3x3 | Minimal grid, few interactions |
| Medium | 5x5 | Standard complexity |
| Hard | 8x8 | Many cell interactions |
| Expert | 10x10 | Large-scale pattern tracking |

## Scoring Metrics

- **Cell Accuracy**: Percentage of cells correctly predicted
- **Perfect Match**: Binary score for exact board match
- **Response Time**: How long the LLM took to respond

## File Structure

```
conway-bench/
├── config.json          # LLM configuration (API key, model, etc.)
├── conway.py            # Core Game of Life implementation
├── benchmark.py         # LLM benchmark runner
├── test_conway.py       # Unit tests
├── requirements.txt     # Python dependencies
├── PROPOSAL.md          # Design specification
└── README.md            # This file
```

## Core Functions

### `conway.py`

- `next_state(board)` - Compute the next generation
- `board_to_ascii(board)` - Convert board to ASCII string
- `ascii_to_board(ascii_str)` - Parse ASCII back to board array
- `generate_random_board(rows, cols, density, seed)` - Generate random initial state
- `calculate_accuracy(predicted, expected)` - Calculate cell accuracy
- `is_perfect_match(predicted, expected)` - Check for exact match

### `benchmark.py`

- `run_benchmark()` - Run the full benchmark suite
- `load_config()` - Load configuration from `config.json`
- `query_llm(prompt)` - Send query to LLM via OpenRouter

## See Also

- [PROPOSAL.md](PROPOSAL.md) - Detailed design and API specification
- [OpenRouter Documentation](https://openrouter.ai/docs)
