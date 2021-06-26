import numpy as np


def test_generate_minimax_move():
    """
    First checking whether the agent can return an action.
    Then it asserts the agent will producing valid move.
    Next, it will test if the agent can produce a winning move
    given a board state
    """
    from agents.agent_minimax import generate_move
    from agents.common import NO_PLAYER, PLAYER1, PLAYER2, BoardPiece

    # blank board test
    test_board = np.full((6, 7), NO_PLAYER)
    action, _ = generate_move(test_board, PLAYER1, None)
    assert action != -1

    # negative test - invalid situations
    test_board = np.full((6, 7), PLAYER2)
    test_board[0, 0] = NO_PLAYER
    action, _ = generate_move(test_board, PLAYER1, None)
    assert (action == -1)

    test_board = np.zeros((6, 7))
    test_board[3:, 0] = PLAYER1
    test_board[5, 1:3] = PLAYER2
    action, _ = generate_move(test_board, PLAYER1, None)
    assert (action == -1)

    action, _ = generate_move(test_board, PLAYER2, None)
    assert (action == -1)

    # positive test - winning move
    from agents.common import string_to_board
    board_str = """|==============| 
|              |
|              |
|              |
|        X O   |
|    O X X O   |
|X O X O O X   |
|==============|
|0 1 2 3 4 5 6 |"""

    board = string_to_board(board_str)
    action, _ = generate_move(board, PLAYER1, None)
    assert (action == 5)


    board_str = """|==============| 
|              |
|              |
|              |
|        X O   |
|    O O X O   |
|X O X O X X   |
|==============|
|0 1 2 3 4 5 6 |"""

    board = string_to_board(board_str)
    action, _ = generate_move(board, PLAYER1, None)
    assert (action == 4)