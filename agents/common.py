from enum import Enum
from typing import Optional
import numpy as np
from typing import Callable, Tuple

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiece_Print = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = str(' ')
PLAYER1_PRINT = str('X')
PLAYER2_PRINT = str('O')

PlayerAction = np.int8  # The column to be played
CONNECT_N = 4


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.zeros(shape=(6, 7), dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    board_str = '|==============|\n'
    end_two_lines = board_str + '|0 1 2 3 4 5 6 |'

    str_board_array = board.copy().astype('object')
    """ 
    using ternary operator here - where we are checking 
    if (the board position is equal to Player1)
       the use the print version of player1 
    else if (the board position is equal to Player2) 
       the use the print version of player2
    else
       the use the print version of no_player
    """
    str_board_array = np.where(str_board_array == PLAYER1, PLAYER1_PRINT,
                               np.where(str_board_array == PLAYER2, PLAYER2_PRINT, NO_PLAYER_PRINT))

    for i in range(1, str_board_array.shape[0] + 1):
        # reversing and joining the space and the pipes
        board_str += '|' + str(' '.join(str_board_array[-i, :])) + ' |' + '\n'

    return board_str + end_two_lines


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    :rtype: board
    """
    game_rows_string = pp_board.splitlines()
    # ignoring the boundary styles of the board
    game_rows_string_actual = game_rows_string[1: len(game_rows_string) - 2]
    game_rows_string_actual_reversed = game_rows_string_actual[::-1]  # reversing the string board
    each_row_length = len(game_rows_string_actual_reversed[1]) - 2
    board = np.empty((0, int(each_row_length / 2)), np.int8)
    for game_row in game_rows_string_actual_reversed:
        # removing the '|'s from start and end and taking only the player print characters
        game_row_actual = game_row[1:each_row_length:2]
        # replacing player print character with player numbers
        game_row_actual = game_row_actual.replace(PLAYER1_PRINT, str(PLAYER1)).replace(PLAYER2_PRINT, str(PLAYER2)).replace(' ', str(NO_PLAYER))
        # creating board only is board is undefined so as to generalize the size of the board
        board = np.vstack((board, list(map(np.int8, game_row_actual))))
    return board


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    updated_board = board
    if copy:
        updated_board = board.copy()
    i = np.min(np.where(updated_board[:, action] == NO_PLAYER))
    updated_board[i, action] = player
    return updated_board


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """

    rows, cols = board.shape
    rows_edge = rows - CONNECT_N
    cols_edge = cols - CONNECT_N
    if last_action is not None:
        start_action_idx = max(0, last_action - 4)
        end_action_idx = min(last_action + 4, cols)
        small_board = board[:, start_action_idx:end_action_idx]
        # Remark: you could do the same for rows
        #  - Student comment: added row filter
        last_row_action = np.max(np.argwhere(small_board[:, last_action] != NO_PLAYER))
        start_action_idx = last_row_action - 4
        if start_action_idx > 0:
            small_board = small_board[start_action_idx:last_row_action, :]
        return connected_four(small_board, player)
    else:
        for i in range(rows):
            for j in range(cols):
                # horizontal connected 4
                if j <= cols_edge and np.all(board[i, j:(j + CONNECT_N)] == player):
                    return True
                # vertical connected 4
                if i <= rows_edge and np.all(board[i:(i + CONNECT_N), j] == player):
                    return True
                # positively sloped diagonal connected 4
                # Remark: diagonals could be expressed more concisely
                #  - Student comment: tried to concise it for both positively and negatively sloped diagonal,
                #                     by using np.diag(block) but it is somehow not working and taking diagonal of
                #                     length 3 sometimes also, I tried to debug it through, but failed
                if i <= rows_edge and j <= cols_edge \
                        and board[i][j] == player and board[i + 1][j + 1] == player \
                        and board[i + 2][j + 2] == player and board[i + 3][j + 3] == player:
                    # block = board[i:i + CONNECT_N, j:j + CONNECT_N]
                    # if np.all(np.diag(block) == player):
                    return True
                # negatively sloped diagonal connected 4
                if i >= CONNECT_N - 1 and j <= cols_edge \
                        and board[i][j] == player and board[i - 1][j + 1] == player \
                        and board[i - 2][j + 2] == player and board[i - 3][j + 3] == player:
                    # block = board[i:i + CONNECT_N, j:j + CONNECT_N]
                    # if np.all(np.diag(block[::-1, :]) == player):
                    return True
        return False


def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player, last_action):
        return GameState.IS_WIN
    if not (board == NO_PLAYER).any():
        return GameState.IS_DRAW
    return GameState.STILL_PLAYING


def is_action_valid(board: np.ndarray, action: PlayerAction):
    """
    Returns True if the current action is a valid action
    """
    return board[len(board) - 1, action] == NO_PLAYER


def get_valid_actions(board: np.ndarray):
    """
    Returns all the valid next actions
    """
    valid_actions = []
    for action in range(len(board[0])):
        if is_action_valid(board, action):
            valid_actions.append(action)
    return valid_actions


class SavedState:
    def __init__(self, computational_result):
        self.computational_result = computational_result


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]
