import numpy as np


def test_generate_mcts_move():
    """
    First checking whether the agent can return an action.
    Then it asserts the agent will producing valid move.
    Next, it will test if the agent can produce a winning move
    given a board state
    """
    from agents.agent_mcts import generate_move
    from agents.common import NO_PLAYER, PLAYER1, PLAYER2, BoardPiece
    print("\n 1st Test : ---------------------")
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
    print(board_str)
    board = string_to_board(board_str)
    action, _ = generate_move(board, PLAYER1, None)
    assert (action == 5)

    print("\n 2nd Test : ---------------------")

    board_str = """|==============| 
|              |
|              |
|              |
|        X O   |
|    O O X O   |
|X O X O X X   |
|==============|
|0 1 2 3 4 5 6 |"""

    print(board_str)
    board = string_to_board(board_str)
    action, _ = generate_move(board, PLAYER1, None)
    assert (action == 4)

    print("\n 3rd Test : ---------------------")

    board_str = """|==============|
|              |
|              |
|              |
|              |
|    X X       |
|    X O O O   |
|==============|
|0 1 2 3 4 5 6 |"""

    print(board_str)
    board = string_to_board(board_str)
    action, _ = generate_move(board, PLAYER1, None)
    assert (action == 4)