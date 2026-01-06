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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ACNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size, activation):
        super().__init__()
        sizes = [input_dim] + hidden_layer_size + [output_dim]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation())
        self.feature_layer = nn.Sequential(*layers)

        self.actor = nn.Sequential(nn.Linear(sizes[-2], sizes[-1]))
        self.critic = nn.Sequential(nn.Linear(sizes[-2], 1))

    def forward(self, x):
        features = self.feature_layer(x)
        action_logit = self.actor(features)
        state_value = self.critic(features)
        return action_logit, state_value

    def evaluate(self, state, action):
        action_logits, state_values = self.forward(state)

        dist = torch.distributions.Categorical(logits=action_logits)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_log_probs, state_values, dist_entropy


class Agent:
    def __init__(self, input_dim, output_dim, device, learning_rate, gamma, update_steps,
                 epochs, clip_ratio, value_coef, entropy_coef, record, hidden_layer_size,
                 activation = nn.ReLU, optimizer = torch.optim.Adam, loss_fn = nn.MSELoss):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.gamma = gamma
        self.update_steps = update_steps
        self.epochs = epochs
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.memory = []
        self.recent_rewards = deque(maxlen=record)
        self.recent_wins = deque(maxlen=record)
        self.net = ACNet(self.input_dim, self.output_dim, hidden_layer_size, activation).to(device)
        self.net_old = ACNet(self.input_dim, self.output_dim, hidden_layer_size, activation).to(device)
        self.net_old.load_state_dict(self.net.state_dict())
        self.optimizer = optimizer(self.net.parameters(), lr=learning_rate)
        self.loss = loss_fn()

    def act(self, state, eval_mode=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_logit, _ = self.net_old(state_tensor)
            if eval_mode:
                action = int(torch.argmax(action_logit, dim=1).item())
                return action, []
            else:
                dist = torch.distributions.Categorical(logits=action_logit)
                action = dist.sample()
                action_logprob = dist.log_prob(action)
                return action.item(), [action_logprob.item()]

    def update(self):
        states = torch.FloatTensor(np.array([i[0] for i in self.memory])).to(self.device)
        actions = torch.FloatTensor(np.array([i[1] for i in self.memory])).to(self.device)
        probs = torch.FloatTensor(np.array([i[2] for i in self.memory])).to(self.device)
        rewards = [i[3] for i in self.memory]
        dones = [i[4] for i in self.memory]

        rewards_renew = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_renew.insert(0, discounted_reward)

        rewards_renew = torch.FloatTensor(rewards_renew).to(device)
        rewards_norm = (rewards_renew - rewards_renew.mean()) / (rewards_renew.std() + 1e-7)

        for _ in range(self.epochs):
            logprobs, state_values, dist_entropy = self.net.evaluate(states, actions)

            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - probs.detach())
            advantages = rewards_norm - state_values.detach()

            temp1 = ratios * advantages
            temp2 = torch.clamp(ratios, (1 - self.clip_ratio), 1 + self.clip_ratio) * advantages

            loss = (-torch.min(temp1, temp2).mean() +
                    self.value_coef * self.loss(state_values, rewards_norm) -
                    self.entropy_coef * dist_entropy).mean()

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.net_old.load_state_dict(self.net.state_dict())
        self.memory = []


def train_once(agent, env, episode):
    state, _ = env.reset()
    rewards = []

    done = False
    while not done:
        action, [log_p] = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.memory.append((state, action, log_p, reward, done))
        rewards = rewards + [reward]
        state = next_state

        if len(agent.memory) >= agent.update_steps:
            agent.update()


    agent.recent_rewards.append(sum(rewards))
    is_win = 1 if sum(rewards) > 0 else 0
    agent.recent_wins.append(is_win)


if __name__ == "__main__":
    hidden_layer_size = [1024, 512]
    learning_rate = 3e-4
    gamma = 0.99
    seed = 1
    episodes = 200000
    epochs = 10
    clip_ratio = 0.2
    value_coef = 0.5
    entropy_coef = 0.03
    update_steps = 8192
    record = 100
    env_name = "WordleEnv100"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    best_win_rate = 0
    rewards_plot = []
    wins_plot = []

    env, input_dim, output_dim = Wordle_Env.create_env(env_name)
    print(f"Starting optimized training on {env_name}")
    print(f"State Dim: {input_dim}, Action Dim: {output_dim}")
    print(f"Using device: {device}")
    agent = Agent(input_dim, output_dim, device, learning_rate, gamma, update_steps,
                  epochs, clip_ratio, value_coef, entropy_coef, record, hidden_layer_size)
    for episode in range(1, episodes + 1):
        train_once(agent, env, episode)
        if episode % record == 0:
            Wordle_Env.print_agent_train_log(agent.recent_rewards, agent.recent_wins, episode)
            win_rate = np.mean(agent.recent_wins)
            rewards_plot.append(np.mean(agent.recent_rewards))
            if win_rate > best_win_rate:
                best_win_num = win_rate
                # torch.save(agent.net.state_dict(), "../trained_Models/PPO100.pth")
    print("Training finished.")
    print("\nRunning a test game with the trained agent...")
    Wordle_Env.test_agent(env, agent)

    env.close()
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(rewards_plot)
    plt.ylim([-10, 10])
    plt.grid(True)
    plt.xlabel('episode * 100')
    plt.title("Rewards in 100 words wordle")
    plt.show()

    env.close()