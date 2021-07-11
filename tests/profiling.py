import cProfile
from agents.agent_mcts import generate_move
from main import human_vs_agent

cProfile.run(
    "human_vs_agent(generate_move, generate_move)", "mmab.dat"
)

import pstats
# from pstats import SortKey

with open("output_time.txt", "w") as f:
    p = pstats.Stats("mmab.dat", stream=f)
    p.sort_stats("time").print_stats()

with open("output_calls.txt", "w") as f:
    p = pstats.Stats("mmab.dat", stream=f)
    p.sort_stats("calls").print_stats()
