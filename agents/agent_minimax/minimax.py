from enum import Enum
from typing import Optional, Tuple
from agents.common import BoardPiece, SavedState, PlayerAction, get_valid_actions, check_end_state, \
    GameState, PLAYER1, PLAYER2, NO_PLAYER, apply_player_action, CONNECT_N
import numpy as np


class Count(Enum):
    IS_COUNTABLE = 1
    IS_NOT_COUNTABLE = 0


def score_action(board: np.ndarray, player: BoardPiece):
    """
    Calculates the heuristic score of current board. Simple heuristic to evaluate board configurations Heuristic is
    (num of 4-in-a-rows)*99999 + (num of 3-in-a-rows)*100 + (num of 2-in-a-rows)*10
    - (num of opponent 4-in-a-rows)*99999 - (num of opponent 3-in-a-rows)*100 - (num of opponent 2-in-a-rows)*10
    :param board:               np.ndarray
                                Current board represented by array for game state
    :param player:              BoardPiece
                                Current player taking the turn
    :return:                    int
                                returns the heuristic score of current board
    """

    # Remark: cleaner/shorter is opp_player = PLAYER2 if player == PLAYER1 else PLAYER1
    #  - Student comment : fixed it
    # opp_player = PLAYER1
    # if player == PLAYER1:
    #     opp_player = PLAYER2
    opp_player = PLAYER2 if player == PLAYER1 else PLAYER1

    my_fours = check_for_score_for_no_of_filled_position(board, player, 4)
    my_threes = check_for_score_for_no_of_filled_position(board, player, 3)
    my_twos = check_for_score_for_no_of_filled_position(board, player, 2)
    opp_fours = check_for_score_for_no_of_filled_position(board, opp_player, 4)
    opp_threes = check_for_score_for_no_of_filled_position(board, opp_player, 3)
    opp_twos = check_for_score_for_no_of_filled_position(board, opp_player, 2)
    if opp_fours > 0:
        return -100000
    else:
        # Remark: easier:
        #         return 100000 (my_fours - opp_fours) + 100 (my_threes - opp_threes) + 10 (my_twos - opp_twos)
        #         - Student comment : fixed it
        # return my_fours*100000 + my_threes*100 + my_twos * 10 - opp_fours * 100000 - opp_threes * 100 - opp_twos * 10
        return (my_fours - opp_fours) * 100000 + (my_threes - opp_threes) * 100 + (my_twos - opp_twos) * 10


def check_for_score_for_no_of_filled_position(board: np.ndarray, player: BoardPiece, no_of_filled_position: int):
    """
    # Remark: explain briefly what the function is doing - Student comment : added
    Calculate the heuristic score of current board  given the number of current player positions
    :param board:               np.ndarray
                                Current board represented by array for game state
    :param player:              BoardPiece
                                Current player taking the turn
    :param no_of_filled_position:  int
                                number of current player positions
    :return:                    int
                                returns the heuristic score
    """
    count = 0
    rows, cols = board.shape
    rows_edge = rows - CONNECT_N
    cols_edge = cols - CONNECT_N
    # for each piece in the board...
    # Remark: you should restrict the loops instead of checking whether i,j are smaller than cols_edge, rows_edge
    #  - Student comment : as discussed, avoided multiple loops to improve performance
    for i in range(rows):
        for j in range(cols):
            # ...that is of the player we're looking for...
            if board[i][j] == player:
                # Remark: there's a mistake below:
                #         first you check whether you have a number of noOfFilledPosition connected pieces from player,
                #         but the next check should be on board[i, j + noOfFilledPosition : j + CONNECT_N]
                #         However, I'm not sure if this will detect wins if noOfFilledPosition == CONNECT_N
                #  - Student comment : fixed (the erroneous code is commented out and is to be removed after correction)
                # horizontal connected noOfFilledPosition
                if j <= cols_edge and np.all(board[i, j:j + no_of_filled_position] == player) \
                        and np.all(board[i, j + no_of_filled_position: j + CONNECT_N] == NO_PLAYER):
                    # and np.all(board[i, j + noOfFilledPosition: CONNECT_N] == NO_PLAYER):
                    count += 1

                # vertical connected noOfFilledPosition
                if i <= rows_edge and np.all(board[i:i + no_of_filled_position, j] == player) \
                        and np.all(board[i + no_of_filled_position: i + CONNECT_N, j] == NO_PLAYER):
                    # and np.all(board[i + noOfFilledPosition: CONNECT_N, j] == NO_PLAYER):
                    count += 1

                # positively sloped diagonal connected noOfFilledPosition
                if i <= rows_edge and j <= cols_edge:
                    count += positive_diagonal_check(i, j, board, player, no_of_filled_position)

                # negatively sloped diagonal connected noOfFilledPosition
                if i >= CONNECT_N - 1 and j <= cols_edge:
                    count += negative_diagonal_check(i, j, board, player, no_of_filled_position)
    return count


def positive_diagonal_check(row: int, col: int, board: np.ndarray, player: BoardPiece, no_of_filled_position: int):
    """
    Calculate the positive diagonal heuristic score of current board  given the number of current player positions
        :param row:                 int
                                    Current row
        :param col:                 int
                                    Current column
        :param board:               np.ndarray
                                    Current board represented by array for game state
        :param player:              BoardPiece
                                    Current player taking the turn
        :param no_of_filled_position:  int
                                    number of current player positions
        :return:                    int
                                    returns positive diagonal heuristic score
        """
    is_eligible_for_diag_score = False
    if no_of_filled_position == 4:
        is_eligible_for_diag_score = board[row + 1][col + 1] == player and board[row + 2][col + 2] == player \
                                     and board[row + 3][col + 3] == player
    elif no_of_filled_position == 3:
        is_eligible_for_diag_score = board[row + 1][col + 1] == player and board[row + 2][col + 2] == player \
                                     and board[row + 3][col + 3] == NO_PLAYER
    elif no_of_filled_position == 2:
        is_eligible_for_diag_score = board[row + 1][col + 1] == player and board[row + 2][col + 2] == NO_PLAYER \
                                     and board[row + 3][col + 3] == NO_PLAYER
    # Remark: you should return booleans here, or enumerations. Numerical values are easy to confuse
    #  - Student comment : fixed it
    if is_eligible_for_diag_score:
        return Count.IS_COUNTABLE.value
    return Count.IS_NOT_COUNTABLE.value


def negative_diagonal_check(row: int, col: int, board: np.ndarray, player: BoardPiece, no_of_filled_position: int):
    """
    Calculate the negative diagonal heuristic score of current board  given the number of current player positions
                                current player positions
    :param row:                 int
                                Current row
    :param col:                 int
                                Current column
    :param board:               np.ndarray
                                Current board represented by array for game state
    :param player:              BoardPiece
                                Current player taking the turn
    :param no_of_filled_position:  int
                                number of current player positions
    :return:                    int
                                returns the negative diagonal heuristic score
    """
    is_eligible_for_diag_score = False
    if no_of_filled_position == 4:
        is_eligible_for_diag_score = board[row - 1][col + 1] == player and board[row - 2][col + 2] == player \
                                     and board[row - 3][col + 3] == player
    elif no_of_filled_position == 3:
        is_eligible_for_diag_score = board[row - 1][col + 1] == player and board[row - 2][col + 2] == player \
                                     and board[row - 3][col + 3] == NO_PLAYER
    elif no_of_filled_position == 2:
        is_eligible_for_diag_score = board[row - 1][col + 1] == player and board[row - 2][col + 2] == NO_PLAYER \
                                     and board[row - 3][col + 3] == NO_PLAYER
    if is_eligible_for_diag_score:
        return Count.IS_COUNTABLE.value
    return Count.IS_NOT_COUNTABLE.value


def minimax_with_alpha_beta_pruning(board: np.ndarray, depth: int, alpha: float, beta: float, player: BoardPiece) -> (
        PlayerAction, int):
    """
    Apply minimax with alpha beta pruning and generate move for current player and returns player action
    :param board:           np.ndarray
                            Current state of the board
    :param depth:           int
                            integer representing how deep into the game tree is search by minimax agent to evaluate move
    :param alpha:           float
                            alpha parameter to prune away computing node values whenever alpha > beta
    :param beta:            float
                            beta parameter to prune away computing node values whenever alpha > beta
    :param player:          BoardPiece
                            Player for whom move is being generated
    :return:                tuple
                            tuple containing player action (move) and saved state
    """

    valid_locations = get_valid_actions(board)

    if depth == 0 or len(valid_locations) == 0 or check_end_state(board, player) != GameState.STILL_PLAYING:
        return -1, score_action(board, player)


    # Remark: you should randomize among the moves with the highest score after evaluating
    #  - Student comment : fixed it by just shuffling and then taking the best_column
    # best_column = np.random.choice(valid_locations)
    np.random.shuffle(valid_locations)

    if player == PLAYER1:
        max_score = -np.inf
        best_column = valid_locations.__getitem__(0)
        for col in valid_locations:
            updated_copy_board = apply_player_action(board, col, player, True)
            _, new_score = minimax_with_alpha_beta_pruning(updated_copy_board, depth - 1, alpha, beta, PLAYER2)
            if new_score > max_score:
                max_score = new_score
                best_column = col
            alpha = max(alpha, new_score)
            if alpha >= beta:
                break
        return best_column, max_score
    else:
        min_score = np.inf
        best_column = valid_locations.__getitem__(0)
        for col in valid_locations:
            updated_copy_board = apply_player_action(board, col, player, True)
            _, new_score = minimax_with_alpha_beta_pruning(updated_copy_board, depth - 1, alpha, beta, PLAYER1)
            if new_score < min_score:
                min_score = new_score
                best_column = col
            # Remark: you have to reset beta here (beta = min(beta, new_score))
            #  - Student Comment : fixed it, erroneous code commented out and will be removed after checking
            # alpha = min(alpha, new_score)
            beta = min(new_score, beta)
            if beta <= alpha:
                break
        return best_column, min_score


def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
        PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    action, _ = minimax_with_alpha_beta_pruning(board, 4, -np.inf, np.inf, player)
    return action, saved_state
