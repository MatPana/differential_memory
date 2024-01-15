import numpy as np
from gym import Env, spaces


class Concentration(Env):
    def __init__(self, n_pairs: int = 4) -> None:
        self.n_pairs = n_pairs

        self.action_space = spaces.Discrete(2 * self.n_pairs)
        self.observation_space = spaces.Box(low=0, high=self.n_pairs - 1, shape=(1,))

        self.reset()

    def step(self, action: np.array) -> tuple:
        reward = 0
        done = False

        action = action.reshape(-1)

        # If we are uncovering the card that was already matched.
        if self.matched_board[action]:
            reward = -1
        # If we are uncovering the second in pair.
        elif self.board[action] == self.last_card:
            self.matched_board[action] = True
            self.matched_board[self.last_card] = True
            self.matched += 1

            if self.matched == self.n_pairs:
                done = True
                reward = 10
            else:
                reward = 1

        self.last_card = self.board[action]

        return self.board[action], reward, done, None, None

    def get_board(self) -> np.array:
        cards = np.array([i for i in range(self.n_pairs)] * 2)
        np.random.shuffle(cards)
        return cards

    def reset(self) -> tuple:
        self.board = self.get_board()
        self.matched_board = np.zeros_like(self.get_board(), dtype=bool)
        self.matched = 0
        self.last_card = self.board[[0]]

        return self.last_card, None
