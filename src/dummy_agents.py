import numpy as np

class Agent:
    def choose_action(self,state,legal_actions):
        pass

class RandomAgent(Agent):
    def choose_action(self,state,legal_actions):
        return np.random.choice(legal_actions)