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

class QNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size, activation):
        super().__init__()
        sizes = [input_dim] + hidden_layer_size + [output_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(activation())
        self.q_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.q_net(x)


class Agent:
    def __init__(self, input_dim, output_dim, device, learning_rate, gamma, batch_size, memory_size, target_update,
                 epsilon, epsilon_min, epsilon_decay, record, hidden_layer_size,
                 activation = nn.ReLU, optimizer = torch.optim.Adam, loss_fn = nn.SmoothL1Loss):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.memory = deque(maxlen=memory_size)
        self.recent_rewards = deque(maxlen=record)
        self.recent_wins = deque(maxlen=record)
        self.policy_net = QNet(self.input_dim, self.output_dim, hidden_layer_size, activation).to(device)
        self.target_net = QNet(self.input_dim, self.output_dim, hidden_layer_size, activation).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optimizer(self.policy_net.parameters(), lr=learning_rate)
        self.loss = loss_fn()

    def act(self, state, eval_mode=False):
        if eval_mode:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item(), []
        else:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.output_dim), []
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item(), []

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        else:
            batch = random.sample(self.memory, self.batch_size)
            states = torch.FloatTensor(np.array([i[0] for i in batch])).to(self.device)
            actions = torch.LongTensor(np.array([i[1] for i in batch])).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(np.array([i[2] for i in batch])).to(self.device)
            next_states = torch.FloatTensor(np.array([i[3] for i in batch])).to(self.device)
            dones = torch.FloatTensor(np.array([i[4] for i in batch])).to(self.device)

            current_q = self.policy_net(states).gather(1, actions).squeeze(1)

            with torch.no_grad():
                max_next_q = self.target_net(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * max_next_q

            loss = self.loss(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def train_once(agent, env, episode):
    state, _ = env.reset()
    rewards = []

    done = False
    while not done:
        action, _ = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        rewards += [reward]
        agent.update()

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if episode % agent.target_update == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.recent_rewards.append(sum(rewards))
    is_win = 1 if sum(rewards) > 0 else 0
    agent.recent_wins.append(is_win)



if __name__ == "__main__":
    hidden_layer_size = [512, 512]
    learning_rate = 1e-4
    gamma = 0.99
    seed = 1
    episodes = 20000
    epsilon = 0.5
    epsilon_min = 0.01
    epsilon_decay = 0.9995
    batch_size = 64
    memory_size = 2000
    target_update = 50
    record = 100
    env_name = "WordleEnv100"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env, input_dim, output_dim = Wordle_Env.create_env(env_name)
    print(f"Starting optimized training on {env_name}")
    print(f"State Dim: {input_dim}, Action Dim: {output_dim}")
    print(f"Using device: {device}")
    agent = Agent(input_dim, output_dim, device, learning_rate, gamma, batch_size, memory_size, target_update,
                  epsilon, epsilon_min, epsilon_decay, record, hidden_layer_size)
    for episode in range(1, episodes + 1):
        train_once(agent, env, episode)
        if episode % record == 0:
            Wordle_Env.print_agent_train_log(agent.recent_rewards, agent.recent_wins, episode, agent.epsilon)
    print("Training finished.")
    print("\nRunning a test game with the trained agent...")
    Wordle_Env.test_agent(env, agent)

    env.close()