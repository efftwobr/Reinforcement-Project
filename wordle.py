import os
from typing import Optional, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from . import state
from .const import WORDLE_N, REWARD

dirname = os.path.dirname(__file__)
VALID_WORDS_PATH = f'{dirname}/../../data/wordle_words.txt'


def _load_words(limit: Optional[int] = None) -> List[str]:
    with open(VALID_WORDS_PATH, 'r') as f:
        lines = [x.strip().upper() for x in f.readlines()]
        if limit is None:
            return lines
        return lines[:limit]


class WordleEnvBase(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, words: List[str],
                 max_turns: int,
                 allowable_words: Optional[int] = None,
                 frequencies: Optional[List[float]] = None,
                 mask_based_state_updates: bool = False):

        assert all(len(w) == WORDLE_N for w in words), "All words must be length WORDLE_N"

        self.words = words
        self.max_turns = max_turns
        self.allowable_words = allowable_words or len(self.words)
        self.mask_based_state_updates = mask_based_state_updates

        self.frequencies = None
        if frequencies:
            assert len(words) == len(frequencies)
            self.frequencies = np.array(frequencies, dtype=np.float32) / sum(frequencies)

        self.action_space = spaces.Discrete(len(self.words))
        dummy_state = state.new(max_turns)
        obs_shape = dummy_state.shape 

        self.observation_space = spaces.Box(
            low=0,
            high=6,
            shape=obs_shape,
            dtype=np.int32
        )

        self.done = True
        self.goal_word = -1

        self.state: state.WordleState = None
        self.state_updater = state.update
        if self.mask_based_state_updates:
            self.state_updater = state.update_mask

    def step(self, action: int):
        if self.done:
            raise ValueError("step() called after environment is done. Call reset().")

        self.state = self.state_updater(
            state=self.state,
            word=self.words[action],
            goal_word=self.words[self.goal_word]
        )

        reward = 0
        if action == self.goal_word:
            self.done = True
            if state.remaining_steps(self.state) == self.max_turns - 1:
                reward = 0
            else:
                reward = REWARD
        elif state.remaining_steps(self.state) == 0:
            self.done = True
            reward = -REWARD

        terminated = self.done
        truncated = False

        return self.state.astype(np.int32).copy(), reward, terminated, truncated, {"goal_id": self.goal_word}

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.state = state.new(self.max_turns)
        self.done = False
        self.goal_word = np.random.randint(0, self.allowable_words)
        return self.state.astype(np.int32).copy(), {}

    def set_goal_word(self, goal_word: str):
        self.goal_word = self.words.index(goal_word)

    def set_goal_id(self, goal_id: int):
        self.goal_word = goal_id


class WordleEnv10(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(10), max_turns=6)


class WordleEnv100(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), max_turns=6)


class WordleEnv100OneAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), allowable_words=1, max_turns=6)


class WordleEnv100WithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), max_turns=6, mask_based_state_updates=True)


class WordleEnv100TwoAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), allowable_words=2, max_turns=6)


class WordleEnv100FullAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=100, max_turns=6)


class WordleEnv1000(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(1000), max_turns=6)


class WordleEnv1000WithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(1000), max_turns=6, mask_based_state_updates=True)


class WordleEnv1000FullAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=1000, max_turns=6)


class WordleEnvFull(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), max_turns=6)


class WordleEnvReal(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=2315, max_turns=6)


class WordleEnvRealWithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(
            words=_load_words(),
            allowable_words=2315,
            max_turns=6,
            mask_based_state_updates=True,
        )