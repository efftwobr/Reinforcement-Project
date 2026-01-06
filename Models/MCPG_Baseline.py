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
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)


class BaselineNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_size, activation):
        super().__init__()
        sizes = [input_dim] + hidden_layer_size + [1]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)


class Agent:
    def __init__(self, input_dim, output_dim, device, learning_rate, gamma, record,
                 hidden_layer_size, activation = nn.ReLU, optimizer = torch.optim.Adam):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.gamma = gamma
        self.recent_rewards = deque(maxlen=record)
        self.recent_wins = deque(maxlen=record)
        self.policy_net = PolicyNet(self.input_dim, self.output_dim, hidden_layer_size, activation).to(self.device)
        self.policy_optimizer = optimizer(self.policy_net.parameters(), lr=learning_rate)
        self.baseline_net = BaselineNet(self.input_dim, hidden_layer_size, activation).to(self.device)
        self.baseline_optimizer = optimizer(self.baseline_net.parameters(), lr=learning_rate)
    def act(self, state, eval_mode=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_p = self.policy_net(state_tensor)
        value = self.baseline_net(state_tensor)
        if eval_mode:
            action = int(torch.argmax(action_p, dim=1).item())
            return action, []
        else:
            dist = torch.distributions.Categorical(action_p)
            action = dist.sample()
            log_p = dist.log_prob(action)
            return action, [log_p, value]
    def update(self, rewards, log_ps, values, values_detach):
        if len(rewards) == 1:
            pass
        else:
            gs = []
            g = 0
            for r in reversed(rewards):
                g = r + self.gamma * g
                gs.append(g)
            gs.reverse()

            deltas = torch.tensor(gs)
            log_probs = torch.stack(log_ps)
            values_detach = torch.tensor(values_detach)
            values = torch.stack(values)
            loss = -torch.sum(log_probs * (deltas - values_detach))
            baseline_loss = torch.sum((deltas - values) ** 2)

            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()



def train_once(agent, env):
    state, _ = env.reset()
    log_ps = []
    rewards = []
    values = []
    values_detach = []

    done = False
    while not done:
        action, [log_p, value] = agent.act(state)
        log_ps.append(log_p)
        values = values + [value]
        values_detach = values_detach + [value.detach()]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards = rewards + [reward]
        state = next_state

    agent.recent_rewards.append(sum(rewards))
    is_win = 1 if sum(rewards) > 0 else 0
    agent.recent_wins.append(is_win)

    agent.update(rewards, log_ps, values, values_detach)


if __name__ == "__main__":
    hidden_layer_size = [16, 32]
    learning_rate = 1e-4
    gamma = 0.99
    seed = 1
    episodes = 30000
    record = 1000
    env_name = "WordleEnv10"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env, input_dim, output_dim = Wordle_Env.create_env(env_name)
    print(f"Starting optimized training on {env_name}")
    print(f"State Dim: {input_dim}, Action Dim: {output_dim}")
    print(f"Using device: {device}")
    agent = Agent(input_dim, output_dim, device, learning_rate, gamma, record, hidden_layer_size)

    for episode in range(1, episodes + 1):
        train_once(agent, env)
        if episode % record == 0:
            Wordle_Env.print_agent_train_log(agent.recent_rewards, agent.recent_wins, episode)
    print("Training finished.")
    print("\nRunning a test game with the trained agent...")
    Wordle_Env.test_agent(env, agent)

    env.close()