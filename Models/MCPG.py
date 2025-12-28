from __future__ import annotations
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import Wordle_Env


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size, activation):
        super().__init__()
        sizes = [input_dim] + hidden_layer_size + [output_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(activation())
            else:
                layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)
    def forward(self,x):
        return self.model(x)


class Agent:
    def __init__(self,input_dim, output_dim, device, learning_rate,
                 hidden_layer_size, activation = nn.ReLU, optimizer = torch.optim.Adam, ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.memory = []
        self.build_model(learning_rate, optimizer, hidden_layer_size, activation)
    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_p = self.model(state_tensor)
        action = int(torch.argmax(action_p, dim=1).item())
        dist = torch.distributions.Categorical(action_p)
        action_sample = dist.sample()
        log_p = dist.log_prob(action_sample)
        return action, [action_sample.item(), log_p]
    def build_model(self, learning_rate, optimizer, hidden_layer_size, activation):
        self.model = PolicyNet(self.input_dim, self.output_dim, hidden_layer_size, activation)
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
    def update(self, rewards, log_ps, gamma):
        if len(rewards) == 1:
            pass
        else:
            gs = []
            g = 0
            for r in reversed(rewards):
                g = r + gamma * g
                gs.append(g)
            gs.reverse()

            deltas = torch.tensor(gs)
            log_probs = torch.stack(log_ps)
            loss = -torch.sum(log_probs * deltas)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def train_once(Agent, env, gamma, recent_rewards, recent_wins):
    state, _ = env.reset()
    log_ps = []
    rewards = []

    done = False
    while not done:
        _, [action, log_p] = Agent.act(state)
        log_ps.append(log_p)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards = rewards + [reward]
        state = next_state

    recent_rewards.append(sum(rewards))
    is_win = 1 if sum(rewards) > 0 else 0
    recent_wins.append(is_win)

    Agent.update(rewards, log_ps, gamma)


if __name__ == "__main__":
    hidden_layer_size = [256,512]
    learning_rate = 1e-4
    gamma = 0.99
    seed = 1
    episodes = 30000
    record = 1000
    env_name = "WordleEnv100"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recent_rewards = deque(maxlen=record)
    recent_wins = deque(maxlen=record)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env, input_dim, output_dim = Wordle_Env.create_env(env_name)
    Agent = Agent(input_dim, output_dim, device, learning_rate, hidden_layer_size)

    for episode in range(episodes):
        train_once(Agent, env, gamma, recent_rewards, recent_wins)
        if episode % record == 0:
            Wordle_Env.print_agent_train_log(recent_rewards, recent_wins, episode)
    print("Training finished.")
    print("\nRunning a test game with the trained agent...")
    Wordle_Env.test_agent(env, Agent)

    env.close()