# Benchmark Summary

## Purpose

This benchmark tests LLM **spatial reasoning** and **rule-following** capabilities using Conway's Game of Life. It evaluates whether models can correctly simulate one generation step given a grid state.

## What It Tests

The benchmark presents LLMs with ASCII grids (using `#` for alive, `.` for dead cells) and asks them to compute the next generation according to Conway's rules:

- Live cells with 2-3 neighbors survive
- Dead cells with exactly 3 neighbors become alive
- All other cells die or stay dead

## Test Suite

| Difficulty | Grid Size | Test Cases |
|------------|-----------|------------|
| Easy       | 3x3       | 2          |
| Medium     | 5x5       | 3          |
| Hard       | 8x8       | 2          |
| Expert     | 10x10     | 2          |

Total: **9 test cases** with deterministic seeds for reproducibility.

## Metrics Collected

1. **Cell Accuracy** - Percentage of cells correctly predicted
2. **Perfect Match** - Whether the entire board was predicted correctly
3. **Response Time** - Latency of the LLM response

## How It Works

1. Generate a random board using a fixed seed
2. Compute the correct next state using the reference implementation
3. Prompt the LLM with the current board state
4. Parse the LLM's ASCII response
5. Compare predicted vs expected boards
6. Record accuracy metrics

## Why This Is Interesting

Conway's Game of Life requires:

- **Counting** - Accurately counting neighbors for each cell
- **Spatial awareness** - Tracking positions in a 2D grid from text
- **Rule application** - Applying different rules based on cell state and neighbor count
- **Precision** - Every cell must be correct for a perfect match

Larger grids exponentially increase difficulty as the model must track more cells and their relationships simultaneously.
