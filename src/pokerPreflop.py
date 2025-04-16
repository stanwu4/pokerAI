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

all_starting_hands = generateStartingHands()

class actions(Enum):
    """
    Enum for poker actions.
    """
    CHECK = 0
    Fold = 1
    Call = 2
    Raise = 3
    All_In = 4

class Game:
    def __init__(self):
        
        self.state = []
        idx1 = np.random.randint(0, len(all_starting_hands) - 1)
        idx2 = np.random.randint(0, len(all_starting_hands) - 1)
        self.player1_hand, self.player2_hand = (all_starting_hands[idx1], all_starting_hands[idx2])
        self.winner = None
        self.player1_turn = True

        self.starting_stack = 20
        self.pot = 1.5 #small blind + big blind
        self.cur_raise = 0.5

        self.player1_stack = self.starting_stack - 0.5
        self.player2_stack = self.starting_stack - 1.0

        self.game_over = False
        self.reward = 0
    
    @staticmethod
    def get_actions(state):
        if len(state) == 0:
            return [actions.Fold, actions.Call, actions.Raise, actions.All_In]
        elif state[-1] == actions.Call:
            return [actions.Fold, actions.Raise, actions.All_In]
        elif state[-1] == actions.Raise:
            return [actions.Fold, actions.Call, actions.Raise, actions.All_In]
        elif state[-1] == actions.All_In:
            return [actions.Fold, actions.Call]
        else:
            return []

    def make_action(self,action):
        legal_actions = self.get_actions(self.state)
        if action not in legal_actions:
            raise Exception("Illegal Action")
        self.state.append(action)
        if action == actions.Fold:
            self.reward = (self.player1_stack - self.starting_stack
            if self.player1_turn
            else self.starting_stack - self.player2_stack
            )
            self.game_over = True
        if action == actions.CHECK:
            self.evaluate_winner()
        if action == actions.Call:
            if self.player1_turn:
                self.player1_stack -= self.cur_raise
            else:
                self.player2_stack -= self.cur_raise
            self.pot += self.cur_raise
            self.cur_raise = 0
            if (len(self.state) >= 2) and (self.state[-2] == actions.Raise or self.state[-2] == actions.All_In):
                self.evaluate_winner()

        if action == actions.Raise:
            
            raise_amount = self.cur_raise + (self.pot + self.cur_raise)
            self.pot += raise_amount
            self.cur_raise = raise_amount - self.cur_raise

            if self.player1_turn:
                self.player1_stack -= raise_amount
            else:
                self.player2_stack -= raise_amount
        
        if action == actions.All_In:
            raise_amount = self.player1_stack if self.player1_turn else self.player2_stack
            self.pot += raise_amount
            self.cur_raise = raise_amount - self.cur_raise
            if self.player1_turn:
                self.player1_stack = 0
            else:
                self.player2_stack = 0
        
        self.player1_turn = not self.player1_turn

    def evaluate_winner(self):
        hand1 = self.player1_hand
        hand2 = self.player2_hand
        
        hand1_equity = preflopEVs[hand1]

        self.reward = self.pot * hand1_equity + self.player1_stack - self.starting_stack
        self.game_over = True

    def get_game_over(self):
        return self.game_over
    
    def get_reward(self):
        return self.reward
    
    def hand_to_string(self,hand):
        rank = list("23456789TJQKA")
        rank_int_0 = Card.rank_int(hand[0])
        suit_int_0 = Card.get_suit_int(hand[0])
        rank_int_1 = Card.rank_int(hand[1])
        suit_int_1 = Card.get_suit_int(hand[1])

        hand_str = ""
        if rank_int_0 > rank_int_1:
            hand_str += rank[rank_int_0] + rank[rank_int_1]
        else:
            hand_str += rank[rank_int_1] + rank[rank_int_0]
        hand_str += "S" if suit_int_0 == suit_int_1 else "O"
        return hand_str
    
    def get_state(self,player_one):
        state_str = []
        if player_one:
            state_str.append(self.player1_hand)
        else:
            state_str.append(self.player2_hand)

        for action in self.state:
            if action == actions.CHECK:
                state_str.append("X")
            elif action == actions.Fold:
                state_str.append("F")
            elif action == actions.Call:
                state_str.append("C")
            elif action == actions.Raise:
                state_str.append("R")
            elif action == actions.All_In:
                state_str.append("RAI")
        return ",".join(state_str)
    
class QLearningAgent:
    def __init__(self, alpha = 0.25, gamma = 0.75, epsilon = 0.1, batch_size = 4096):
        self.q_table = defaultdict(float) #q-values initialized to 0
        self.alpha = alpha #learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.history = []  # To store the (state, action) pairs of each game
        self.experience_buffer = []  # To store all experiences for batch updates
        self.buffer_cnt = (
            0  # count for how many games are stored in our experience buffer
        )
        self.batch_size = batch_size  # Number of experiences to collect before updating
    
    def choose_action(self, state, legal_actions):
        #Epsilon-greedy action selection
        if np.random.uniform(0,1) < self.epsilon:
            return random.choice(legal_actions)
        else:
            q_values = [self.q_table[(state, action)] for action in legal_actions]
            max_q = max(q_values)
            max_q_actions = [
                action
                for action in legal_actions
                if self.q_table[(state, action)] == max_q
                ]
            return random.choice(max_q_actions)
        
    def load_experiences(self, final_reward):
        for t in reversed(range(len(self.history))):
            state,action = self.history[t]
            if t == len(self.history) - 1:
                reward = final_reward
                next_state = None
            else:
                next_state, _ = self.history[t + 1]
                reward = 0

            self.add_experience(state, action, reward, next_state)
        
        self.history = []
        if self.buffer_cnt >= self.batch_size:
            self.update()
        
    def add_experience(self, state, action, reward, next_state):
        self.experience_buffer.append((state, action, reward, next_state))
        
    def update(self):
        reward_sums = defaultdict(float)
        reward_counts = defaultdict(int)

        for state, action, reward, next_state in self.experience_buffer:
            reward_sums[(state, action)] += reward
            reward_counts[(state, action)] += 1

            if next_state:
                future_qs = list(self.q_table[entry]
                                 for entry in self.q_table
                                 if entry[0] == next_state
                )
                future_q = 0
                if len(future_qs) > 0:
                    future_q = max(future_qs)
                reward_sums[(state, action)] += future_q
        
        for (state,action), total_reward in reward_sums.items():
            current_q = self.q_table[(state, action)]
            average_reward = total_reward / reward_counts[(state, action)]
            if action == actions.FOLD:
                self.q_table[(state,action)] = average_reward
            else:
                self.q_table[(state,action)] = (
                    current_q + self.alpha * (average_reward - current_q)
                )
        self.experience_buffer = []
        self.buffer_cnt = 0

    def print_table(slef, save_file = None):
        for state_action in self.q_table.keys():
            state, action = state_action
            print(f"State: {state}", file = save_file)
            print(f"Action: {action}", file = save_file)
            print(f"Q-Value: {self.q_table[state_action]}", file = save_file)
            print("", file = save_file)
    
    def main():


        if __name__ == "__main__":
            main()