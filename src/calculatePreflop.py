import random
import numpy as np
from phevaluator import evaluate_cards
from collections import defaultdict

def allCards():
    # """Returns a list of all 52 cards in a standard deck."""
    suits = ['H', 'D', 'C', 'S']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    cards = []
    for suit in suits:
        for rank in ranks:
            cards.append(rank + suit)
    return cards

def genStartingHands():
    ranks = '23456789TJQKA'
    standingHands = []

    for i in range(len(ranks)):
        for j in range(i, len(ranks)):
            handSuited = ranks[i] + ranks[j] + 's'
            handOff = ranks[i] + ranks[j] + 'o'
            standingHands.append(handSuited)
            standingHands.append(handOff)
    return standingHands

def drawCardsForHandType(handType,cards):
    card1 = handType[0]
    card2 = handType[1]
    suited = handType[2] == 's'
    paired = card1 == card2

    hand = []
    
