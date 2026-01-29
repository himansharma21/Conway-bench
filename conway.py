"""
Core implementation of Conway's Game of Life for the LLM benchmark.
"""

import numpy as np


def next_state(board: np.ndarray) -> np.ndarray:
    """
    Compute the next generation of a Conway's Game of Life board.

    Rules applied:
    - Any live cell with 2-3 neighbors survives
    - Any dead cell with exactly 3 neighbors becomes alive
    - All other cells die or stay dead

    Args:
        board: 2D numpy array where 1 = alive, 0 = dead

    Returns:
        2D numpy array representing the next state
    """
    rows, cols = board.shape
    new_board = np.zeros((rows, cols), dtype=int)

    for i in range(rows):
        for j in range(cols):
            # Count neighbors (including diagonals)
            # Cells outside the grid are considered dead (edge handling)
            neighbors = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        neighbors += board[ni, nj]

            # Apply Conway's rules
            if board[i, j] == 1:
                # Live cell: survives if 2 or 3 neighbors
                if neighbors == 2 or neighbors == 3:
                    new_board[i, j] = 1
            else:
                # Dead cell: becomes alive if exactly 3 neighbors
                if neighbors == 3:
                    new_board[i, j] = 1

    return new_board


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
    lines = []
    for row in board:
        line = "".join(alive if cell == 1 else dead for cell in row)
        lines.append(line)
    return "\n".join(lines)


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
    stripped = ascii_str.strip()
    if not stripped:
        return np.zeros((0, 0), dtype=int)
    lines = stripped.split("\n")
    rows = len(lines)
    cols = max(len(line) for line in lines)

    board = np.zeros((rows, cols), dtype=int)

    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == alive:
                board[i, j] = 1
            # Dead cells stay 0

    return board


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
    rng = np.random.default_rng(seed)
    return (rng.random((rows, cols)) < density).astype(int)


def calculate_accuracy(predicted: np.ndarray, expected: np.ndarray) -> float:
    """
    Calculate the percentage of cells correctly predicted.

    Args:
        predicted: The predicted board state
        expected: The expected/correct board state

    Returns:
        Float between 0.0 and 1.0 representing accuracy
    """
    if predicted.shape != expected.shape:
        return 0.0
    return float(np.mean(predicted == expected))


def is_perfect_match(predicted: np.ndarray, expected: np.ndarray) -> bool:
    """
    Check if the predicted board exactly matches the expected board.

    Args:
        predicted: The predicted board state
        expected: The expected/correct board state

    Returns:
        True if boards are identical, False otherwise
    """
    if predicted.shape != expected.shape:
        return False
    return np.array_equal(predicted, expected)
