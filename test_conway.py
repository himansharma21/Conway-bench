"""
Unit tests for Conway's Game of Life implementation.
"""

import numpy as np
import pytest

from conway import (
    next_state,
    board_to_ascii,
    ascii_to_board,
    generate_random_board,
    calculate_accuracy,
    is_perfect_match,
)


class TestNextState:
    """Tests for the next_state function."""

    def test_empty_board_stays_empty(self):
        """An empty board should remain empty."""
        board = np.zeros((5, 5), dtype=int)
        result = next_state(board)
        assert np.array_equal(result, board)

    def test_lonely_cell_dies(self):
        """A single cell with no neighbors should die."""
        board = np.zeros((5, 5), dtype=int)
        board[2, 2] = 1
        result = next_state(board)
        assert result[2, 2] == 0

    def test_cell_with_two_neighbors_survives(self):
        """A live cell with exactly 2 neighbors should survive."""
        board = np.zeros((5, 5), dtype=int)
        # Horizontal line of 3
        board[2, 1] = 1
        board[2, 2] = 1
        board[2, 3] = 1
        result = next_state(board)
        # Middle cell has 2 neighbors, should survive
        assert result[2, 2] == 1

    def test_cell_with_three_neighbors_survives(self):
        """A live cell with exactly 3 neighbors should survive."""
        board = np.zeros((5, 5), dtype=int)
        # Create a 2x2 block
        board[1, 1] = 1
        board[1, 2] = 1
        board[2, 1] = 1
        board[2, 2] = 1
        result = next_state(board)
        # All cells in 2x2 block have 3 neighbors, should survive
        assert result[1, 1] == 1
        assert result[1, 2] == 1
        assert result[2, 1] == 1
        assert result[2, 2] == 1

    def test_cell_with_four_neighbors_dies(self):
        """A live cell with 4+ neighbors should die (overpopulation)."""
        board = np.zeros((5, 5), dtype=int)
        # Cross shape - center has 4 neighbors
        board[1, 2] = 1
        board[2, 1] = 1
        board[2, 2] = 1
        board[2, 3] = 1
        board[3, 2] = 1
        result = next_state(board)
        # Center cell has 4 neighbors, should die
        assert result[2, 2] == 0

    def test_dead_cell_with_three_neighbors_becomes_alive(self):
        """A dead cell with exactly 3 neighbors should become alive (reproduction)."""
        board = np.zeros((5, 5), dtype=int)
        # L-shape
        board[1, 1] = 1
        board[1, 2] = 1
        board[2, 1] = 1
        result = next_state(board)
        # Cell at (2, 2) has 3 neighbors, should become alive
        assert result[2, 2] == 1

    def test_horizontal_becomes_vertical(self):
        """A horizontal blinker should become vertical."""
        board = np.zeros((5, 5), dtype=int)
        # Horizontal line of 3
        board[2, 1] = 1
        board[2, 2] = 1
        board[2, 3] = 1
        result = next_state(board)
        # Should become vertical line
        assert result[1, 2] == 1
        assert result[2, 2] == 1
        assert result[3, 2] == 1
        # Other cells should be dead
        assert result[2, 1] == 0
        assert result[2, 3] == 0

    def test_edge_handling(self):
        """Cells at edges should treat outside as dead."""
        board = np.zeros((3, 3), dtype=int)
        # Corner cell with only 1 neighbor
        board[0, 0] = 1
        board[0, 1] = 1
        result = next_state(board)
        # Both should die
        assert result[0, 0] == 0
        assert result[0, 1] == 0


class TestBoardToAscii:
    """Tests for the board_to_ascii function."""

    def test_empty_board(self):
        """Empty board should be all dead characters."""
        board = np.zeros((3, 3), dtype=int)
        result = board_to_ascii(board)
        assert result == "...\n...\n..."

    def test_full_board(self):
        """Full board should be all alive characters."""
        board = np.ones((3, 3), dtype=int)
        result = board_to_ascii(board)
        assert result == "###\n###\n###"

    def test_mixed_board(self):
        """Mixed board should show correct pattern."""
        board = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        result = board_to_ascii(board)
        assert result == "#.#\n.#.\n#.#"

    def test_custom_characters(self):
        """Custom characters should be used."""
        board = np.array([[1, 0], [0, 1]])
        result = board_to_ascii(board, alive="O", dead="X")
        assert result == "OX\nXO"


class TestAsciiToBoard:
    """Tests for the ascii_to_board function."""

    def test_empty_ascii(self):
        """Empty ASCII should return empty board."""
        result = ascii_to_board("")
        assert result.shape == (0, 0)

    def test_all_dead(self):
        """All dead cells should return zeros."""
        result = ascii_to_board("...\n...\n...")
        expected = np.zeros((3, 3), dtype=int)
        assert np.array_equal(result, expected)

    def test_all_alive(self):
        """All alive cells should return ones."""
        result = ascii_to_board("###\n###\n###")
        expected = np.ones((3, 3), dtype=int)
        assert np.array_equal(result, expected)

    def test_mixed_pattern(self):
        """Mixed pattern should be correctly parsed."""
        result = ascii_to_board("#.#\n.#.\n#.#")
        expected = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        assert np.array_equal(result, expected)

    def test_custom_characters(self):
        """Custom characters should be correctly parsed."""
        result = ascii_to_board("OX\nXO", alive="O", dead="X")
        expected = np.array([[1, 0], [0, 1]])
        assert np.array_equal(result, expected)

    def test_roundtrip(self):
        """board_to_ascii and ascii_to_board should be inverses."""
        original = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        ascii_repr = board_to_ascii(original)
        result = ascii_to_board(ascii_repr)
        assert np.array_equal(result, original)


class TestGenerateRandomBoard:
    """Tests for the generate_random_board function."""

    def test_shape(self):
        """Generated board should have correct shape."""
        result = generate_random_board(5, 7, seed=42)
        assert result.shape == (5, 7)

    def test_reproducibility(self):
        """Same seed should produce same board."""
        result1 = generate_random_board(5, 5, seed=42)
        result2 = generate_random_board(5, 5, seed=42)
        assert np.array_equal(result1, result2)

    def test_different_seeds(self):
        """Different seeds should likely produce different boards."""
        result1 = generate_random_board(10, 10, seed=42)
        result2 = generate_random_board(10, 10, seed=43)
        # Very unlikely to be identical with large boards
        assert not np.array_equal(result1, result2)

    def test_density_low(self):
        """Low density should produce mostly dead cells."""
        result = generate_random_board(20, 20, density=0.1, seed=42)
        alive_ratio = np.mean(result)
        assert alive_ratio < 0.2  # Should be around 0.1

    def test_density_high(self):
        """High density should produce mostly alive cells."""
        result = generate_random_board(20, 20, density=0.9, seed=42)
        alive_ratio = np.mean(result)
        assert alive_ratio > 0.8  # Should be around 0.9


class TestCalculateAccuracy:
    """Tests for the calculate_accuracy function."""

    def test_perfect_match(self):
        """Identical boards should have 100% accuracy."""
        board = np.array([[1, 0], [0, 1]])
        assert calculate_accuracy(board, board) == 1.0

    def test_no_match(self):
        """Opposite boards should have 0% accuracy."""
        predicted = np.ones((2, 2), dtype=int)
        expected = np.zeros((2, 2), dtype=int)
        assert calculate_accuracy(predicted, expected) == 0.0

    def test_partial_match(self):
        """Partial match should have proportional accuracy."""
        predicted = np.array([[1, 0], [1, 0]])
        expected = np.array([[1, 0], [0, 1]])
        # 2 out of 4 correct = 50%
        assert calculate_accuracy(predicted, expected) == 0.5

    def test_different_shapes(self):
        """Different shapes should return 0 accuracy."""
        predicted = np.ones((3, 3))
        expected = np.ones((2, 2))
        assert calculate_accuracy(predicted, expected) == 0.0


class TestIsPerfectMatch:
    """Tests for the is_perfect_match function."""

    def test_identical_boards(self):
        """Identical boards should be a perfect match."""
        board = np.array([[1, 0], [0, 1]])
        assert is_perfect_match(board, board) is True

    def test_different_boards(self):
        """Different boards should not be a perfect match."""
        board1 = np.array([[1, 0], [0, 1]])
        board2 = np.array([[1, 1], [0, 1]])
        assert is_perfect_match(board1, board2) is False

    def test_different_shapes(self):
        """Different shapes should not be a perfect match."""
        board1 = np.ones((3, 3))
        board2 = np.ones((2, 2))
        assert is_perfect_match(board1, board2) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
