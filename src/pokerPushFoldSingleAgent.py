from enum import Enum
import random
import copy
import json

import numpy as np
from collections import defaultdict
from tqdm import tqdm


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


class Game:
    def __init__(self):
        self.state = []
        idx1 = np.random.randint(0, len(all_starting_hands) - 1)
        idx2 = np.random.randint(0, len(all_starting_hands) - 1)
        self.player1_hand, self.player2_hand = (
            all_starting_hands[idx1],
            all_starting_hands[idx2],
        )

        self.winner = None
        self.player1_turn = True

        self.starting_stack = 20.0
        self.pot = 1.5  # SB + BB
        self.cur_raise = 0.5

        self.player1_stack = self.starting_stack - 0.5
        self.player2_stack = self.starting_stack - 1.0

        self.game_over = False
        self.reward = 0

    @staticmethod
    def get_actions(state):
        if len(state) == 0:
            return [Actions.FOLD, Actions.CALL, Actions.RAISE]
        elif state[-1] == Actions.CALL:
            return [Actions.CHECK, Actions.RAISE]
        elif state[-1] == Actions.RAISE:
            return [Actions.FOLD, Actions.CALL]
        else:
            return []

    def make_action(self, action):
        legal_actions = self.get_actions(self.state)
        if action not in legal_actions:
            raise Exception("Illegal Action")
        self.state.append(action)
        if action == Actions.FOLD:
            self.reward = (
                self.player1_stack - self.starting_stack
                if self.player1_turn
                else self.starting_stack - self.player2_stack
            )
            self.game_over = True
        if action == Actions.CHECK:
            self.evaluate_winner()
        if action == Actions.CALL:
            if self.player1_turn:
                self.player1_stack -= self.cur_raise
            else:
                self.player2_stack -= self.cur_raise

            self.pot += self.cur_raise
            self.cur_raise = 0
            if (len(self.state) >= 2) and self.state[-2] == Actions.RAISE:
                self.evaluate_winner()
        if action == Actions.RAISE:
            raise_amount = (
                self.player1_stack if self.player1_turn else self.player2_stack
            )
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

        hand1_equity = preflop_evs[hand1][hand2]

        self.reward = self.pot * hand1_equity + self.player1_stack - self.starting_stack
        self.game_over = True

    def get_game_over(self):
        return self.game_over

    def get_reward(self):
        return self.reward

    def get_state(self, player_one):
        state_str = []
        if player_one:
            state_str.append(self.player1_hand)
        else:
            state_str.append(self.player2_hand)

        for action in self.state:
            if action == Actions.CHECK:
                state_str.append("X")
            if action == Actions.FOLD:
                state_str.append("F")
            if action == Actions.CALL:
                state_str.append("C")
            if action == Actions.RAISE:
                state_str.append("R")

        return ",".join(state_str)


class QLearningAgent:
    def __init__(self, alpha=0.05, gamma=0.75, epsilon=0.15, batch_size=10240):
        self.q_table = defaultdict(float)  # Q-values initialized to 0
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.history = []  # To store the (state, action) pairs of each game
        self.experience_buffer = []  # To store all experiences for batch updates
        self.buffer_cnt = (
            0  # count for how many games are stored in our experience buffer
        )
        self.batch_size = batch_size  # Number of experiences to collect before updating

    def choose_action(self, state, legal_actions):
        if np.random.uniform(0, 1) < self.epsilon:
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
            state, action = self.history[t]
            if t >= len(self.history) - 1:
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
        reward_counts = defaultdict(
            int
        ) 

        for state, action, reward, next_state in self.experience_buffer:
            reward_sums[(state, action)] += reward
            reward_counts[(state, action)] += 1

            if next_state:
                future_qs = list(
                    self.q_table[entry]
                    for entry in self.q_table
                    if entry[0] == next_state
                )
                future_q = 0
                if len(future_qs) > 0:
                    future_q = max(future_qs)
                reward_sums[(state, action)] += future_q
                

        for (state, action), total_reward in reward_sums.items():
            current_q = self.q_table[(state, action)]
            average_reward = total_reward / reward_counts[(state, action)]
            
            if action == Actions.FOLD:
                self.q_table[(state, action)] = average_reward
            else:
                self.q_table[(state, action)] = current_q + self.alpha * (
                    average_reward - current_q
                )

        self.experience_buffer = []
        self.buffer_cnt = 0

    def print_table(self, save_file=None):
        for state_action in self.q_table.keys():
            state, action = state_action
            print(f"State: {state}", file=save_file)
            print(f"Action: {action}", file=save_file)
            print(f"Q-value: {self.q_table[state_action]}", file=save_file)
            print("", file=save_file)

    def load_table(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            state = None
            action_str = None
            for line in f:
                line = line.strip()
                if line.startswith("State: "):
                    state = line[len("State: ") :]
                elif line.startswith("Action: "):
                    action_str = line[len("Action: ") :]
                elif line.startswith("Q-value: "):
                    q_value_str = line[len("Q-value: ") :]
                    q_value = float(q_value_str)
                    action = Actions[action_str.split(".")[-1]]
                    self.q_table[(state, action)] = q_value

        print(f"Q-table successfully loaded from '{filename}'.")


def play_game(agent, copy_agent):
    game = Game()
    player1_agent = np.random.uniform() > 0.5

    while not game.get_game_over():
        if game.player1_turn:
            state = game.get_state(player_one=True)
            legal_actions = Game.get_actions(game.state)
            if player1_agent:
                action = agent.choose_action(state, legal_actions)
                agent.history.append((state, action))
            else:
                action = copy_agent.choose_action(state, legal_actions)
            game.make_action(action)
        else:
            state = game.get_state(player_one=False)
            legal_actions = Game.get_actions(game.state)
            if not player1_agent:
                action = agent.choose_action(state, legal_actions)
                agent.history.append((state, action))
            else:
                action = copy_agent.choose_action(state, legal_actions)
            game.make_action(action)

    reward = game.get_reward()

    agent.buffer_cnt += 1
    agent.load_experiences(final_reward=(reward if player1_agent else -reward))

    return reward


def simulate_game(agent1, agent2):
    game = Game()

    while not game.get_game_over():
        if game.player1_turn:
            state = game.get_state(player_one=True)
            legal_actions = Game.get_actions(game.state)

            action = agent1.choose_action(state, legal_actions)
            game.make_action(action)
        else:
            state = game.get_state(player_one=False)
            legal_actions = Game.get_actions(game.state)
            action = agent2.choose_action(state, legal_actions)
            game.make_action(action)

            print(str(action))

    return game.get_reward()


def main():
    hands_in_epoch = 100000
    agent = QLearningAgent(batch_size=hands_in_epoch, alpha=0.005)
    copy_agent = copy.deepcopy(agent)

    total_reward = 0
    total_hands = 0

    for epoch in range(0, 10000):
        epoch_reward = 0
        for _ in tqdm(range(hands_in_epoch)):
            reward = play_game(agent, copy_agent)
            epoch_reward += reward

        total_reward += epoch_reward
        total_hands += hands_in_epoch
        print(
            f"Epoch {epoch},  Epoch BB per hand: {epoch_reward / hands_in_epoch}, Overall BB per hand: {total_reward / total_hands}"
        )

        if epoch % 5 == 0 and epoch > 0:
          copy_agent = copy.deepcopy(agent)

        if epoch % 10 == 0 and epoch > 0:
            with open(
                f"self_play/q_table_agent_{epoch}.txt", "w", encoding="utf-8"
            ) as file:
                agent.print_table(save_file=file)


if __name__ == "__main__":
    main()
   