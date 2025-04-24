from enum import Enum
import json
import random

import numpy as np
from collections import defaultdict
from tqdm import tqdm
from dummy_agents import RandomAgent

with open("preflop_evs.json", "r") as file:
    preflop_evs = json.load(file)


def generate_starting_hands():
    ranks = list("23456789TJQKA")
    starting_hands = []

    # Pairs
    for rank in ranks:
        hand_str = rank + rank + "O"  # Pairs are offsuit
        for _ in range(6):
            starting_hands.append(hand_str)

    # Non-pair combinations
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            rank1 = ranks[j]  # Higher rank
            rank2 = ranks[i]  # Lower rank
            hand_str_suited = rank1 + rank2 + "S"
            hand_str_offsuit = rank1 + rank2 + "O"
            for _ in range(4):
                starting_hands.append(hand_str_suited)
            for _ in range(12):
                starting_hands.append(hand_str_offsuit)

    return starting_hands


all_starting_hands = generate_starting_hands()


class Actions(Enum):
    CHECK = 0
    FOLD = 1
    CALL = 2
    RAISE = 3

