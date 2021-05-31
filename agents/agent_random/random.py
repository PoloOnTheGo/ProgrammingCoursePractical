from agents.common import PlayerAction, SavedState, BoardPiece, is_action_valid
from typing import Optional, Tuple

import numpy as np


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    action = PlayerAction(-1)
    rows, columns = board.shape
    is_valid = False
    while not is_valid:
        action = np.random.choice(columns, 1)[0]
        is_valid = is_action_valid(board, action)
    return action, saved_state