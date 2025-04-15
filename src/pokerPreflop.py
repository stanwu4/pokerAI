from enum import Enum
import json
import random

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import copy

with open("src/starting_hands.json", "r") as file:
    preflopEVs = json.load(file)

def generateStartingHands():
    """
    Generates all possible starting hands in poker.
    """
    ranks = list("23456789TJQKA")
    starting_hands = []
    
    for rank in ranks:
        handStr = rank + rank + "O"
        for _ in range(6):
            starting_hands.append(handStr)
    
    for i in range(len(ranks)):
        for j in range(i, len(ranks)):
            rank1 = ranks[i]
            rank2 = ranks[j]
            suited = rank1 + rank2 + "S"
            offsuit = rank1 + rank2 + "O"
            for _ in range(4):
                starting_hands.append(suited)
            for _ in range(6):
                starting_hands.append(offsuit)
    return starting_hands

class actions(Enum):
    """
    Enum for poker actions.
    """
    CHECK = 0
    Fold = 1
    Call = 2
    Raise = 3
    All_In = 4




