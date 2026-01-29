# Conway's Game of Life LLM Spatial Reasoning Benchmark

## Overview

A benchmark for testing LLM spatial reasoning capabilities using Conway's Game of Life. The benchmark presents grid states as ASCII text and asks the LLM to compute the next generation, testing pattern recognition and rule application.

## Core Components

### 1. Game Logic (`next_state`)

```python
def next_state(board: np.ndarray) -> np.ndarray:
    """
    Compute the next generation of a Conway's Game of Life board.

    Args:
        board: 2D numpy array where 1 = alive, 0 = dead

    Returns:
        2D numpy array representing the next state
    """
```

**Rules applied:**
- Any live cell with 2-3 neighbors survives
- Any dead cell with exactly 3 neighbors becomes alive
- All other cells die or stay dead

### 2. ASCII Conversion

```python
def board_to_ascii(board: np.ndarray, alive: str = "#", dead: str = ".") -> str:
    """
    Convert a board array to ASCII string representation.

    Args:
        board: 2D numpy array
        alive: Character for live cells (default "#")
        dead: Character for dead cells (default ".")

    Returns:
        Multi-line ASCII string
    """

def ascii_to_board(ascii_str: str, alive: str = "#", dead: str = ".") -> np.ndarray:
    """
    Parse ASCII string back into a board array.

    Args:
        ascii_str: Multi-line ASCII representation
        alive: Character representing live cells
        dead: Character representing dead cells

    Returns:
        2D numpy array
    """
```

### 3. Random State Generation

```python
def generate_random_board(
    rows: int,
    cols: int,
    density: float = 0.3,
    seed: int = 42
) -> np.ndarray:
    """
    Generate a random initial board state.

    Args:
        rows: Number of rows
        cols: Number of columns
        density: Probability of a cell being alive (0.0-1.0)
        seed: Random seed for reproducibility (default 42)

    Returns:
        2D numpy array with random initial state
    """
```

## Example Usage

```python
import numpy as np

# Generate a random 5x5 board
board = generate_random_board(rows=5, cols=5, seed=42)

# Convert to ASCII for LLM prompt
ascii_repr = board_to_ascii(board)
# Output:
# .#..#
# #.#..
# ..#.#
# .#...
# #..#.

# Compute actual next state
expected = next_state(board)

# Parse LLM response back to array
llm_response = """
.#...
.##..
.##..
.....
.....
"""
predicted = ascii_to_board(llm_response)

# Compare for scoring
accuracy = np.mean(predicted == expected)
```

## Benchmark Design

### Test Cases

| Difficulty | Grid Size | Description |
|------------|-----------|-------------|
| Easy       | 3x3       | Minimal grid, few interactions |
| Medium     | 5x5       | Standard complexity |
| Hard       | 8x8       | Many cell interactions |
| Expert     | 10x10+    | Large-scale pattern tracking |

### Scoring Metrics

- **Cell accuracy**: Percentage of cells correctly predicted
- **Perfect match**: Binary score for exact board match
- **Edge handling**: Correct behavior at boundaries (cells outside grid are dead)

## File Structure

```
conway_benchmark/
├── conway.py          # Core implementation
├── benchmark.py       # Test runner and scoring
└── test_conway.py     # Unit tests
```

## Dependencies

- `numpy` - Array operations and efficient neighbor counting
