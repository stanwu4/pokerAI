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
    for card in cards:
        if len(hand) == 0:
            if card[0] == card1 or card[0] == card2:
                hand.append(card)
        elif card[0] != hand[0][0]:
            if card[0] == card1 or card[0] == card2:
                if (card[1] == hand[0][1]) == suited:
                    hand.append(card)
                    break
        elif paired:
            hand.append(card)
            break
    for card in hand:
        cards.remove(card)
    return hand

#this function calculates the estimated value of a hand against another hand - given 2 starting cards and 5 cards on the table
def calcEVForMatchup(hand_type1, hand_type2):
    hand1_wins, hand1ties = 0.0, 0.0
    num_iters = 10000
    for _ in range(num_iters):
        deck = allCards()
        np.random.shuffle(deck)
        hand1 = drawCardsForHandType(hand_type1, deck)
        hand2 = drawCardsForHandType(hand_type2, deck)

        board = random.sample(deck,5)
        hand1_score = evaluate_cards(board + hand1)
        hand2_score = evaluate_cards(board + hand2)
        if hand1_score > hand2_score:
            hand1_wins += 1
        elif hand1_score == hand2_score:
            hand1ties += 1
    
    hand1_equity = hand1_wins / num_iters + hand1ties / (2 * num_iters)
    return hand1_equity

if __name__ == "__main__":
    starting_hands = genStartingHands()
    ev_dict = defaultdict(lambda:defaultdict(float))
    for i in range(len(starting_hands)):
        for j in range(len(starting_hands)):
            hand1 = starting_hands[i]
            hand2 = starting_hands[j]
            if hand1 == hand2:
                continue
            if hand2 in ev_dict:
                if hand1 in ev_dict[hand2]:
                    continue
            ev_dict[hand1][hand2] = calcEVForMatchup(hand1,hand2)
            print(hand1,hand2,ev_dict[hand1][hand2])
    import json

    with open("preflop_ev.json", "w") as f:
        json.dump(ev_dict, f)
        