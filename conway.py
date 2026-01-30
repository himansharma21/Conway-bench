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


def calculate_correctness(predicted: np.ndarray, expected: np.ndarray) -> float:
    """
    Calculate correctness score using geometric mean of F1 scores for alive and dead cells.

    This metric handles class imbalance by computing F1 for both classes separately,
    then taking the geometric mean. The geometric mean ensures that poor performance
    on either class (e.g., predicting all-dead) results in a low score.

    Args:
        predicted: The predicted board state (2D numpy array, 1=alive, 0=dead)
        expected: The expected/correct board state

    Returns:
        Float between 0.0 and 1.0 representing correctness
    """
    if predicted.shape != expected.shape:
        return 0.0

    pred_flat = predicted.flatten()
    exp_flat = expected.flatten()

    # F1 for alive cells (class 1)
    tp_alive = np.sum((pred_flat == 1) & (exp_flat == 1))
    fp_alive = np.sum((pred_flat == 1) & (exp_flat == 0))
    fn_alive = np.sum((pred_flat == 0) & (exp_flat == 1))

    # F1 for dead cells (class 0)
    tp_dead = np.sum((pred_flat == 0) & (exp_flat == 0))
    fp_dead = np.sum((pred_flat == 0) & (exp_flat == 1))
    fn_dead = np.sum((pred_flat == 1) & (exp_flat == 0))

    # Calculate F1 for alive cells with edge case handling
    total_expected_alive = tp_alive + fn_alive
    total_predicted_alive = tp_alive + fp_alive

    if total_expected_alive == 0:
        # No alive cells in expected: perfect if none predicted, else 0
        f1_alive = 1.0 if total_predicted_alive == 0 else 0.0
    elif total_predicted_alive == 0:
        # Expected some alive but predicted none
        f1_alive = 0.0
    else:
        precision_alive = tp_alive / total_predicted_alive
        recall_alive = tp_alive / total_expected_alive
        if precision_alive + recall_alive > 0:
            f1_alive = 2 * precision_alive * recall_alive / (precision_alive + recall_alive)
        else:
            f1_alive = 0.0

    # Calculate F1 for dead cells with edge case handling
    total_expected_dead = tp_dead + fn_dead
    total_predicted_dead = tp_dead + fp_dead

    if total_expected_dead == 0:
        # No dead cells in expected: perfect if none predicted, else 0
        f1_dead = 1.0 if total_predicted_dead == 0 else 0.0
    elif total_predicted_dead == 0:
        # Expected some dead but predicted none
        f1_dead = 0.0
    else:
        precision_dead = tp_dead / total_predicted_dead
        recall_dead = tp_dead / total_expected_dead
        if precision_dead + recall_dead > 0:
            f1_dead = 2 * precision_dead * recall_dead / (precision_dead + recall_dead)
        else:
            f1_dead = 0.0

    # Geometric mean of both F1 scores
    return float(np.sqrt(f1_alive * f1_dead))
