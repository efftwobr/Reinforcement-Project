import os, sys
import random
import collections
from typing import Optional, List
import json
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QPushButton, QMessageBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

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
        return lines[:limit]
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

def candidate_mask(state: WordleState, words: List[str]) -> np.ndarray:

    n = len(words)
    mask = np.ones(n, dtype=np.float32)

    if state is None:
        return mask

    # For each position i and each character, the triple lives at:
    # offset = 1 + len(Chars) + cint * Word_Len * 3
    for idx, w in enumerate(words):
        w = w.upper()
        if len(w) != Word_Len:
            mask[idx] = 0.0
            continue

        feasible = True
        for i, ch in enumerate(w):
            cint = ord(ch) - ord(Chars[0])
            if not (0 <= cint < len(Chars)):
                feasible = False
                break
            offset = 1 + len(Chars) + cint * Word_Len * 3
            triple = state[offset + 3 * i: offset + 3 * i + 3]
            # triple[0] == 1 => explicitly excluded at this position
            if int(triple[0]) == 1:
                feasible = False
                break
        mask[idx] = 1.0 if feasible else 0.0

    return mask


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
            reward = Reward
        if remaining_steps(self.state) == 0:
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


COLOR_CORRECT = "#6aaa64"
COLOR_PRESENT = "#c9b458"
COLOR_ABSENT = "#787c7e"
COLOR_DEFAULT = "#d3d6da"
COLOR_FILLED_BORDER = "#878a8c"


class WordleGame(QMainWindow):
    def __init__(self, env):
        super().__init__()
        self.setWindowTitle("Wordle")
        self.setFixedSize(600, 850)

        self.env = env

        self.cells = []
        self.keys = {}

        self.init_ui()
        self.apply_styles()
        self.reset_game()
        self.msg = QMessageBox(self)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(20, 30, 20, 30)
        central_widget.setLayout(main_layout)

        top_widget = QWidget()
        top_layout = QGridLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_widget.setLayout(top_layout)

        self.title_label = QLabel()
        self.title_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(self.title_label, 0, 1)

        self.btn_restart = QPushButton("Restart")
        self.btn_restart.setFixedSize(60, 40)
        self.btn_restart.setCursor(Qt.PointingHandCursor)
        self.btn_restart.setObjectName("restartBtn")
        self.btn_restart.clicked.connect(self.reset_game)

        top_layout.addWidget(self.btn_restart, 0, 2)

        top_layout.setColumnStretch(0, 0)
        top_layout.setColumnStretch(1, 1)  # 中间拉伸
        top_layout.setColumnStretch(2, 0)

        main_layout.addWidget(top_widget)

        grid_container = QWidget()
        grid_layout = QGridLayout()
        grid_layout.setSpacing(5)

        for row in range(6):
            row_cells = []
            for col in range(Word_Len):
                label = QLabel("")
                label.setFixedSize(62, 62)
                label.setAlignment(Qt.AlignCenter)
                label.setFont(QFont("Arial", 28, QFont.Bold))
                label.setProperty("status", "empty")
                grid_layout.addWidget(label, row, col)
                row_cells.append(label)
            self.cells.append(row_cells)

        grid_container.setLayout(grid_layout)
        main_layout.addWidget(grid_container, 0, Qt.AlignCenter)

        kb_widget = QWidget()
        kb_layout = QGridLayout()
        kb_layout.setSpacing(6)
        kb_layout.setContentsMargins(0, 20, 0, 0)
        kb_widget.setLayout(kb_layout)

        for i in range(20):
            kb_layout.setColumnStretch(i, 1)

        KEY_HEIGHT = 58

        row1_chars = "QWERTYUIOP"
        for i, char in enumerate(row1_chars):
            btn = self.create_key(char, height=KEY_HEIGHT)
            kb_layout.addWidget(btn, 0, i * 2, 1, 2)

        row2_chars = "ASDFGHJKL"
        for i, char in enumerate(row2_chars):
            btn = self.create_key(char, height=KEY_HEIGHT)
            kb_layout.addWidget(btn, 1, 1 + i * 2, 1, 2)

        btn_enter = QPushButton("ENTER")
        btn_enter.setFixedHeight(KEY_HEIGHT)
        btn_enter.setFont(QFont("Arial", 12, QFont.Bold))
        btn_enter.clicked.connect(self.submit)
        kb_layout.addWidget(btn_enter, 2, 0, 1, 3)

        row3_chars = "ZXCVBNM"
        for i, char in enumerate(row3_chars):
            btn = self.create_key(char, height=KEY_HEIGHT)
            kb_layout.addWidget(btn, 2, 3 + i * 2, 1, 2)

        btn_back = QPushButton("⌫")
        btn_back.setFixedHeight(KEY_HEIGHT)
        btn_back.setFont(QFont("Arial", 18, QFont.Bold))
        btn_back.clicked.connect(self.backspace)
        kb_layout.addWidget(btn_back, 2, 17, 1, 3)

        main_layout.addWidget(kb_widget)

    def create_key(self, char, height):
        btn = QPushButton(char)
        btn.setFixedHeight(height)
        btn.setFont(QFont("Arial", 14, QFont.Bold))
        btn.clicked.connect(lambda _, c=char: self.handle_input(c))
        self.keys[char] = btn
        return btn

    def apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: white; }}
            QLabel {{
                border: 2px solid #d3d6da;
                color: black;
                background-color: white;
            }}
            QLabel[status="filled"] {{
                border: 2px solid {COLOR_FILLED_BORDER};
                color: black;
                animation: pop 0.1s;
            }}
            QLabel[status="correct"] {{ background-color: {COLOR_CORRECT}; border: none; color: white; }}
            QLabel[status="present"] {{ background-color: {COLOR_PRESENT}; border: none; color: white; }}
            QLabel[status="absent"]  {{ background-color: {COLOR_ABSENT};  border: none; color: white; }}

            QPushButton {{
                background-color: {COLOR_DEFAULT};
                border: none;
                border-radius: 4px;
                color: black;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #c0c3c7; }}
            QPushButton:pressed {{ background-color: #a0a3a7; }}

            QPushButton#restartBtn {{
                background-color: transparent;
                border: 1px solid {COLOR_DEFAULT};
                border-radius: 4px;
                font-size: 14px;
                color: #555;
            }}
            QPushButton#restartBtn:hover {{
                background-color: #f0f0f0;
                color: black;
            }}
        """)

    def reset_game(self):
        self.env.reset()
        self.target_word_index = self.env.unwrapped.goal_word
        self.target_word = self.env.unwrapped.words[self.target_word_index]
        self.current_row = 0
        self.current_guess = ""
        self.game_over = False
        print(f"Target word is {self.target_word}")

        for row in range(6):
            for col in range(Word_Len):
                label = self.cells[row][col]
                label.setText("")
                label.setProperty("status", "empty")
                self.style().unpolish(label)
                self.style().polish(label)

        for btn in self.keys.values():
            btn.setStyleSheet("")
            btn.setEnabled(True)

    def keyPressEvent(self, event):
        if self.game_over: return
        key = event.text().upper()
        if key.isalpha() and len(key) == 1:
            self.handle_input(key)
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.submit()
        elif event.key() == Qt.Key_Backspace:
            self.backspace()

    def handle_input(self, char):
        if self.game_over or len(self.current_guess) >= Word_Len: return
        self.current_guess += char
        col = len(self.current_guess) - 1
        self.update(self.current_row, col, char, "filled")

    def backspace(self):
        if self.game_over or not self.current_guess: return
        col = len(self.current_guess) - 1
        self.current_guess = self.current_guess[:-1]
        self.update(self.current_row, col, "", "empty")

    def update(self, row, col, text, status):
        label = self.cells[row][col]
        label.setText(text)
        label.setProperty("status", status)
        self.style().unpolish(label)
        self.style().polish(label)

    def submit(self):
        if self.game_over: return
        guess = self.current_guess
        if len(guess) != Word_Len:
            self.show_message("Not enough letters")
            return
        if guess not in self.env.unwrapped.words:
            self.show_message("Invalid Word")
            return
        self.check_word(guess)
        if guess == self.target_word:
            self.show_message(f" You Won!\n\nThe word was: {self.target_word}", is_end=True)
            self.game_over = True
        elif self.current_row >= 6 - 1:
            self.show_message(f" You Lost!\n\nThe word was: {self.target_word}", is_end=True)
            self.game_over = True
        else:
            self.current_row += 1
            self.current_guess = ""

    def check_word(self, guess):
        target_chars = list(self.target_word)
        guess_chars = list(guess)
        result_colors = [None] * Word_Len

        for i in range(Word_Len):
            if guess_chars[i] == target_chars[i]:
                result_colors[i] = "correct"
                target_chars[i] = None
                guess_chars[i] = None

        for i in range(Word_Len):
            if result_colors[i] is None:
                char = guess_chars[i]
                if char in target_chars:
                    result_colors[i] = "present"
                    target_chars[target_chars.index(char)] = None
                else:
                    result_colors[i] = "absent"

        for i, status in enumerate(result_colors):
            self.update(self.current_row, i, self.current_guess[i], status)
            char = self.current_guess[i]
            if char in self.keys:
                btn = self.keys[char]
                current_style = btn.styleSheet()
                new_color = ""
                if status == "correct":
                    new_color = COLOR_CORRECT
                elif status == "present" and COLOR_CORRECT not in current_style:
                    new_color = COLOR_PRESENT
                elif status == "absent" and COLOR_CORRECT not in current_style and COLOR_PRESENT not in current_style:
                    new_color = COLOR_ABSENT

                if new_color:
                    btn.setStyleSheet(f"background-color: {new_color}; color: white; border: none; border-radius: 4px;")

    def show_message(self, text, is_end=False):
        self.msg = QMessageBox(self)
        self.msg.setWindowTitle("Message")
        self.msg.setText(text)
        self.msg.setStandardButtons(QMessageBox.Ok)
        QTimer.singleShot(1000, self.msg.accept)
        self.msg.exec_()



if __name__ == "__main__":
    env_name = "WordleEnv10"
    env, input_dim, output_dim = create_env(env_name)
    app = QApplication(sys.argv)
    game = WordleGame(env)
    game.show()
    state, _ = game.env.reset()
    sys.exit(app.exec_())
