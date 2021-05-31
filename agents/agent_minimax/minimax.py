from typing import Optional, Tuple
from agents.common import BoardPiece, SavedState, PlayerAction, is_action_valid, get_valid_actions, check_end_state, \
    GameState, PLAYER1, PLAYER2, NO_PLAYER, apply_player_action, CONNECT_N, connected_four
import numpy as np


def checkForScoreForNoOfFilledPosition(board: np.ndarray, player: BoardPiece, noOfFilledPosition: int):
    """
    :param board:               np.ndarray
                                Current board represented by array for game state
    :param player:              BoardPiece
                                Current player taking the turn
    :param noOfFilledPosition:  int
                                number of current player positions
    :return:                    int
                                returns the heuristic score of current board  given the number of current player positions
    """
    count = 0
    rows, cols = board.shape
    rows_edge = rows - CONNECT_N
    cols_edge = cols - CONNECT_N
    # for each piece in the board...
    for i in range(rows):
        for j in range(cols):
            # ...that is of the player we're looking for...
            if board[i][j] == player:

                # horizontal connected noOfFilledPosition
                if j <= cols_edge and np.all(board[i, j:j + noOfFilledPosition] == player) \
                        and np.all(board[i, j + noOfFilledPosition: CONNECT_N] == NO_PLAYER):
                    count += 1

                # vertical connected noOfFilledPosition
                if i <= rows_edge and np.all(board[i:i + noOfFilledPosition, j] == player) \
                        and np.all(board[i + noOfFilledPosition: CONNECT_N, j] == NO_PLAYER):
                    count += 1

                # positively sloped diagonal connected noOfFilledPosition
                if i <= rows_edge and j <= cols_edge:
                    count += positiveDiagonalCheck(i, j, board, player, noOfFilledPosition)

                # negatively sloped diagonal connected noOfFilledPosition
                if i >= CONNECT_N - 1 and j <= cols_edge:
                    count += negativeDiagonalCheck(i, j, board, player, noOfFilledPosition)
    return count


def positiveDiagonalCheck(row: int, col: int, board: np.ndarray, player:BoardPiece, noOfFilledPosition: int):
    """
        :param row:                 int
                                    Current row
        :param col:                 int
                                    Current column
        :param board:               np.ndarray
                                    Current board represented by array for game state
        :param player:              BoardPiece
                                    Current player taking the turn
        :param noOfFilledPosition:  int
                                    number of current player positions
        :return:                    int
                                    returns the positive diagonal heuristic score of current board  given the number of
                                    current player positions
        """
    is_eligible_for_diag_score = False
    if noOfFilledPosition == 4:
        is_eligible_for_diag_score = board[row + 1][col + 1] == player and board[row + 2][col + 2] == player \
                                     and board[row + 3][col + 3] == player
    elif noOfFilledPosition == 3:
        is_eligible_for_diag_score = board[row + 1][col + 1] == player and board[row + 2][col + 2] == player \
                                     and board[row + 3][col + 3] == NO_PLAYER
    elif noOfFilledPosition == 2:
        is_eligible_for_diag_score = board[row + 1][col + 1] == player and board[row + 2][col + 2] == NO_PLAYER \
                                     and board[row + 3][col + 3] == NO_PLAYER

    if is_eligible_for_diag_score:
        return 1
    return 0


def negativeDiagonalCheck(row: int, col: int, board: np.ndarray, player: BoardPiece, noOfFilledPosition: int):
    """
    :param row:                 int
                                Current row
    :param col:                 int
                                Current column
    :param board:               np.ndarray
                                Current board represented by array for game state
    :param player:              BoardPiece
                                Current player taking the turn
    :param noOfFilledPosition:  int
                                number of current player positions
    :return:                    int
                                returns the negative diagonal heuristic score of current board  given the number of
                                current player positions
    """
    is_eligible_for_diag_score = False
    if noOfFilledPosition == 4:
        is_eligible_for_diag_score = board[row - 1][col + 1] == player and board[row - 2][col + 2] == player \
                                     and board[row - 3][col + 3] == player
    elif noOfFilledPosition == 3:
        is_eligible_for_diag_score = board[row - 1][col + 1] == player and board[row - 2][col + 2] == player \
                                     and board[row - 3][col + 3] == NO_PLAYER
    elif noOfFilledPosition == 2:
        is_eligible_for_diag_score = board[row - 1][col + 1] == player and board[row - 2][col + 2] == NO_PLAYER \
                                     and board[row - 3][col + 3] == NO_PLAYER
    if is_eligible_for_diag_score:
        return 1
    return 0


def score_action(board: np.ndarray, player: BoardPiece):
    """
    :param board:               np.ndarray
                                Current board represented by array for game state
    :param player:              BoardPiece
                                Current player taking the turn
    :return:                    int
                                returns the heuristic score of current board. Simple heuristic to evaluate board
                                configurations Heuristic is (num of 4-in-a-rows)*99999 + (num of 3-in-a-rows)*100 +
                                (num of 2-in-a-rows)*10 - (num of opponent 4-in-a-rows)*99999 - (num of opponent
                                3-in-a-rows)*100 - (num of opponent 2-in-a-rows)*10
    """
    opp_player = PLAYER1
    if player == PLAYER1:
        opp_player = PLAYER2

    my_fours = checkForScoreForNoOfFilledPosition(board, player, 4)
    my_threes = checkForScoreForNoOfFilledPosition(board, player, 3)
    my_twos = checkForScoreForNoOfFilledPosition(board, player, 2)
    opp_fours = checkForScoreForNoOfFilledPosition(board, opp_player, 4)
    opp_threes = checkForScoreForNoOfFilledPosition(board, opp_player, 3)
    opp_twos = checkForScoreForNoOfFilledPosition(board, opp_player, 2)
    if opp_fours > 0:
        return -100000
    else:
        return my_fours * 100000 + my_threes * 100 + my_twos * 10 - opp_fours * 100000 - opp_threes * 100 - opp_twos * 10


def minimax_with_alpha_beta_pruning(board: np.ndarray, depth: int, alpha: int, beta: int, player: BoardPiece) -> (
        PlayerAction, int):
    """
    Apply minimax with alpha beta pruning and generate move for current player and returns player action
    :param board:           np.ndarray
                            Current state of the board
    :param depth:           int
                            integer representing how deep into the game tree is search by minimax agent for evaluating move
    :param alpha:           float
                            alpha parameter to prune away computing node values whenever alpha > beta
    :param beta:            float
                            beta parameter to prune away computing node values whenever alpha > beta
    :param player:          BoardPiece
                            Player for whom move is being generated
    :param saved_state:     Optional[SavedState]
                            optional saved state config of board
    :return:                tuple
                            tuple containing player action (move) and saved state
    """
    if depth == 0 or check_end_state(board, player) != GameState.STILL_PLAYING:
        return -1, score_action(board, player)

    valid_locations = get_valid_actions(board)
    best_column = np.random.choice(valid_locations)

    if player == PLAYER1:
        max_score = -np.inf
        for col in valid_locations:
            updated_copy_board = apply_player_action(board, col, player, True)
            _, new_score = minimax_with_alpha_beta_pruning(updated_copy_board, depth - 1, alpha, beta, PLAYER2)
            if new_score > max_score:
                max_score = new_score
                best_column = col
            alpha = max(alpha, new_score)
            if beta <= alpha:
                break
        return best_column, max_score
    else:
        min_score = np.inf
        for col in valid_locations:
            updated_copy_board = apply_player_action(board, col, player, True)
            _, new_score = minimax_with_alpha_beta_pruning(updated_copy_board, depth - 1, alpha, beta, PLAYER1)
            if new_score < min_score:
                min_score = new_score
                best_column = col
            alpha = min(alpha, new_score)
            if beta <= alpha:
                break
        return best_column, min_score


def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    action, _ = minimax_with_alpha_beta_pruning(board, 4, -np.inf, np.inf, player)
    return action, saved_state
