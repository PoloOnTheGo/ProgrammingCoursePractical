from typing import Optional, Tuple
from agents.common import BoardPiece, SavedState, PlayerAction

import numpy as np


def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
        PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    action, _ = minimax_with_alpha_beta_pruning(board, 4, -np.inf, np.inf, player)
    return action, saved_state