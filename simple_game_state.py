import numpy as np
import random

class CardGameState(object):
    def __init__(self, rand_seed):
        random.seed(rand_seed)
        self.reset()

    def reset(self):
        self.terminal = False
        self.action_counter = 0
        self.reward = 0

        self.my_cards = np.array([[[1], [1], [1]]])
        self.target_cards = np.array([[[1], [1], [1]]])

        index = random.randrange(3)
        self.target_cards[0][index][0] = 0

        self.s_t = np.concatenate((self.my_cards, self.target_cards), axis=2)

    def process(self, action):
        self.my_cards[0][action][0] = 0
        
        if self.target_cards[0][action][0] == 1:
            self.target_cards[0][action][0] = 0
        
        self.s_t1 = np.concatenate((self.my_cards, self.target_cards), axis=2)

        if self.action_counter == 1:
            self.terminal = True

            if np.sum(self.target_cards, axis=1)[0][0] == 0:
                self.reward = 1
            else:
                self.reward = -1

        self.action_counter += 1

    def update(self):
        self.s_t = self.s_t1

    def print_state(self):
        print('============================================')
        print('my cards: {}'.format(self.my_cards))
        print('target cards: {}'.format(self.target_cards))
        print('reward: {}'.format(self.reward))
        print('terminal: {}'.format(self.terminal))
        print('\n')
        
        

# test = CardGameState(2)

# print(test.target_cards)

# test.process(0)
# print(test.terminal)
# print(test.reward)
# print('===========')

# test.process(1)
# print(test.terminal)
# print(test.reward)
# print('===========')