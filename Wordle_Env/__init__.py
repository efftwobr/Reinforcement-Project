import os
import random
import collections
from typing import Optional, List
import json
import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.registration import register

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'Wordle_Env_Config.json')
with open(config_path, 'r', encoding='utf-8') as file:
    dic_config = json.load(file)

Word_Len = dic_config['Word_Len']
Chars = dic_config['Chars']
Reward = dic_config['Reward']
Full_Words_Path = os.path.join(current_dir, dic_config['Full_Words_Path'])
Wordle_Words_Path = os.path.join(current_dir, dic_config['Wordle_Words_Path'])
WordleState = np.ndarray
NO = 0
SOMEWHERE = 1
YES = 2


def _load_words(words_path: str, limit: Optional[int] = None) -> List[str]:
    with open(words_path, 'r') as f:
        lines = [x.strip().upper() for x in f.readlines()]
        if limit is None:
            return lines
        return random.sample(lines, limit)
def new_wordle_state(max_turns: int) -> WordleState:
    return np.array(
        [max_turns] + [0] * len(Chars) + [0, 1, 0] * Word_Len * len(Chars),
        dtype=np.int32)

def remaining_steps(state: WordleState) -> int:
    return state[0]

def update_from_mask(state: WordleState, word: str, mask: List[int]) -> WordleState:
    state = state.copy()

    prior_yes = []
    prior_maybe = []
    # We need two passes because first pass sets definitely yesses
    # second pass sets the no's for those who aren't already yes
    state[0] -= 1
    for i, c in enumerate(word):
        cint = ord(c) - ord(Chars[0])
        offset = 1 + len(Chars) + cint * Word_Len * 3
        state[1 + cint] = 1
        if mask[i] == YES:
            prior_yes.append(c)
            # char at position i = yes, all other chars at position i == no
            state[offset + 3 * i:offset + 3 * i + 3] = [0, 0, 1]
            for ocint in range(len(Chars)):
                if ocint != cint:
                    oc_offset = 1 + len(Chars) + ocint * Word_Len * 3
                    state[oc_offset + 3 * i:oc_offset + 3 * i + 3] = [1, 0, 0]

    for i, c in enumerate(word):
        cint = ord(c) - ord(Chars[0])
        offset = 1 + len(Chars) + cint * Word_Len * 3
        if mask[i] == SOMEWHERE:
            prior_maybe.append(c)
            # Char at position i = no, other chars stay as they are
            state[offset + 3 * i:offset + 3 * i + 3] = [1, 0, 0]
        elif mask[i] == NO:
            # Need to check this first in case there's prior maybe + yes
            if c in prior_maybe:
                # Then the maybe could be anywhere except here
                state[offset+3*i:offset+3*i+3] = [1, 0, 0]
            elif c in prior_yes:
                # No maybe, definitely a yes, so it's zero everywhere except the yesses
                for j in range(Word_Len):
                    # Only flip no if previously was maybe
                    if state[offset + 3 * j:offset + 3 * j + 3][1] == 1:
                        state[offset + 3 * j:offset + 3 * j + 3] = [1, 0, 0]
            else:
                # Just straight up no
                state[offset:offset+3*Word_Len] = [1, 0, 0]*Word_Len

    return state

def get_mask(word: str, goal_word: str) -> List[int]:
    # Definite yesses first
    mask = [0, 0, 0, 0, 0]
    counts = collections.Counter(goal_word)
    for i, c in enumerate(word):
        if goal_word[i] == c:
            mask[i] = 2
            counts[c] -= 1

    for i, c in enumerate(word):
        if mask[i] == 2:
            continue
        elif c in counts:
            if counts[c] > 0:
                mask[i] = 1
                counts[c] -= 1
            else:
                for j in range(i+1, len(mask)):
                    if mask[j] == 2:
                        continue
                    mask[j] = 0

    return mask

def update_mask(state: WordleState, word: str, goal_word: str) -> WordleState:
    mask = get_mask(word, goal_word)
    return update_from_mask(state, word, mask)

def update(state: WordleState, word: str, goal_word: str) -> WordleState:
    state = state.copy()
    state[0] -= 1
    for i, c in enumerate(word):
        cint = ord(c) - ord(Chars[0])
        offset = 1 + len(Chars) + cint * Word_Len * 3
        state[1 + cint] = 1
        if goal_word[i] == c:
            # char at position i = yes, all other chars at position i == no
            state[offset + 3 * i:offset + 3 * i + 3] = [0, 0, 1]
            for ocint in range(len(Chars)):
                if ocint != cint:
                    oc_offset = 1 + len(Chars) + ocint * Word_Len * 3
                    state[oc_offset + 3 * i:oc_offset + 3 * i + 3] = [1, 0, 0]
        elif c in goal_word:
            # Char at position i = no, other chars stay as they are
            state[offset + 3 * i:offset + 3 * i + 3] = [1, 0, 0]
        else:
            state[offset:offset + 3 * Word_Len] = [1, 0, 0] * Word_Len
    return state


class WordleEnvBase(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, words: List[str],
                 max_turns: int,
                 allowable_words: Optional[int] = None,
                 frequencies: Optional[List[float]] = None,
                 mask_based_state_updates: bool = False):

        self.words = words
        self.max_turns = max_turns
        self.allowable_words = allowable_words or len(self.words)
        self.mask_based_state_updates = mask_based_state_updates

        self.frequencies = None
        if frequencies:
            assert len(words) == len(frequencies)
            self.frequencies = np.array(frequencies, dtype=np.float32) / sum(frequencies)

        self.action_space = gym.spaces.Discrete(len(self.words))
        dummy_state = new_wordle_state(max_turns)
        obs_shape = dummy_state.shape

        self.observation_space = gym.spaces.Box(
            low=0,
            high=6,
            shape=obs_shape,
            dtype=np.int32
        )

        self.done = True
        self.goal_word = -1

        self.state: WordleState = None
        self.state_updater = update
        if self.mask_based_state_updates:
            self.state_updater = update_mask

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
            if remaining_steps(self.state) == self.max_turns - 1:
                reward = Reward
            else:
                reward = Reward
        elif remaining_steps(self.state) == 0:
            self.done = True
            reward = -Reward

        terminated = self.done
        truncated = False

        return self.state.astype(np.int32).copy(), reward, terminated, truncated, {"goal_id": self.goal_word}

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.state = new_wordle_state(self.max_turns)
        self.done = False
        self.goal_word = np.random.randint(0, self.allowable_words)
        return self.state.astype(np.int32).copy(), {}

    def set_goal_word(self, goal_word: str):
        self.goal_word = self.words.index(goal_word)

    def set_goal_id(self, goal_id: int):
        self.goal_word = goal_id


def create_env(env_name):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    return env, input_dim, output_dim

def test_agent(env: WordleEnvBase, Agent):
    state, _ = env.reset()
    done = False
    print(f"Target ID: {env.unwrapped.goal_word}, Word: {env.unwrapped.words[env.unwrapped.goal_word]}")

    step = 1
    while not done:
        action, _ = Agent.act(state, eval_mode=True)
        guess_word = env.unwrapped.words[action]
        next_state, reward, terminated, truncated, _ = env.step(action)
        print(f"Step {step}: Guessed '{guess_word}', Reward: {reward}")

        state = next_state
        done = terminated or truncated
        step += 1

    if reward > 0:
        print("Result: WON!")
    else:
        print("Result: LOST")

def print_agent_train_log(recent_rewards, recent_wins, episode, epsilon = None):
    avg_reward = np.mean(recent_rewards)
    win_rate = np.mean(recent_wins) * 100

    print(f"Episode: {episode:7d} | "
          f"Avg Reward: {avg_reward:2.2f} | "
          f"Win Rate: {win_rate:3.2f}% "
          + (f"｜Epsilon: {epsilon:.4f}" if epsilon else ""))


class WordleEnv10(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(Wordle_Words_Path, 10), max_turns=6)


class WordleEnv100(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(Wordle_Words_Path, 100), max_turns=6)


class WordleEnv100WithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(Wordle_Words_Path, 100), max_turns=6, mask_based_state_updates=True)


class WordleEnv1000(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(Wordle_Words_Path, 1000), max_turns=6)


class WordleEnv1000WithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(Wordle_Words_Path, 1000), max_turns=6, mask_based_state_updates=True)


class WordleEnvReal(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(Wordle_Words_Path), max_turns=6)


class WordleEnvRealWithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(Wordle_Words_Path), max_turns=6, mask_based_state_updates=True)


class WordleEnvFull(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(Full_Words_Path), max_turns=6)


class WordleEnvFullWithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(Full_Words_Path), max_turns=6, mask_based_state_updates=True)


register(id="WordleEnv10", entry_point="Wordle_Env:WordleEnv10", max_episode_steps=6)
register(id="WordleEnv100", entry_point="Wordle_Env:WordleEnv100", max_episode_steps=6)
register(id="WordleEnv100WithMask", entry_point="Wordle_Env:WordleEnv100WithMask", max_episode_steps=6)
register(id="WordleEnv1000", entry_point="Wordle_Env:WordleEnv1000", max_episode_steps=6)
register(id="WordleEnv1000WithMask", entry_point="Wordle_Env:WordleEnv1000WithMask", max_episode_steps=6)
register(id="WordleEnvReal", entry_point="Wordle_Env:WordleEnvReal", max_episode_steps=6)
register(id="WordleEnvRealWithMask", entry_point="Wordle_Env:WordleEnvRealWithMask", max_episode_steps=6)
register(id="WordleEnvFull", entry_point="Wordle_Env:WordleEnvFull", max_episode_steps=6)
register(id="WordleEnvFullWithMask", entry_point="Wordle_Env:WordleEnvFullWithMask", max_episode_steps=6)