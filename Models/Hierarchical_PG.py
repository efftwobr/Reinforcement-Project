from __future__ import annotations
import sys, os
currentdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.dirname(currentdir)
sys.path.append(rootdir)

import random
import numpy as np
from collections import deque
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

import Wordle_Env  # same style as MCPG_Baseline.py


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size, activation):
        super().__init__()
        sizes = (input_dim,) + hidden_layer_size + (output_dim,)
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            # NOTE: no final Softmax — return raw logits
            if i < len(sizes) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BaselineNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_size, activation):
        super().__init__()
        sizes = (input_dim,) + hidden_layer_size + (1,)
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


NUM_OPTIONS = 3


class HighLevelPolicy(nn.Module):
    def __init__(self, input_dim, num_options=NUM_OPTIONS, hidden_layer_size=(32,)):
        super().__init__()
        sizes = (input_dim,) + hidden_layer_size + (num_options,)
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            # NOTE: no final Softmax — return logits
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_option_mask(state_np, option_idx, vocab_size):
    mask = np.ones(vocab_size, dtype=np.float32)

    if option_idx == 0:
        mask[vocab_size // 2 :] = 0.5
    elif option_idx == 1:
        mask[: vocab_size // 2] = 0.5
    elif option_idx == 2:
        remaining_steps = int(state_np[0])
        if remaining_steps <= 2:
            mask[:] = 0.1
            mask[:10] = 1.0

    return torch.from_numpy(mask)


class HierarchicalAgent:
    def __init__(
        self,
        input_dim,
        output_dim,
        device,
        learning_rate,
        gamma,
        record,
        hidden_layer_size=(16, 32),
        activation=nn.ReLU,
        optimizer=torch.optim.Adam,
        entropy_coeff: float = 0.02,
        grad_clip: float = 0.5,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip

        self.recent_rewards = deque(maxlen=record)
        self.recent_wins = deque(maxlen=record)

        self.policy_net = PolicyNet(
            self.input_dim, self.output_dim, hidden_layer_size, activation
        ).to(self.device)

        self.highlevel_net = HighLevelPolicy(
            self.input_dim, NUM_OPTIONS, hidden_layer_size=(32,)
        ).to(self.device)

        self.baseline_net = BaselineNet(
            self.input_dim, hidden_layer_size, activation
        ).to(self.device)

        self.policy_optimizer = optimizer(
            list(self.policy_net.parameters()) +
            list(self.highlevel_net.parameters()),
            lr=learning_rate,
        )
        self.baseline_optimizer = optimizer(
            self.baseline_net.parameters(), lr=learning_rate
        )

        # per-episode set to avoid repeating guesses
        self._tried_actions = set()

    def start_episode(self):
        """Call at the start of each episode to clear tried actions history."""
        self._tried_actions = set()

    def record_action(self, action_idx: int):
        """Record a chosen action so it can be masked out for the rest of the episode."""
        self._tried_actions.add(int(action_idx))

    def act(self, state, eval_mode: bool = False, candidate_mask: Optional[torch.Tensor] = None):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # high-level logits
        option_logits = self.highlevel_net(state_tensor)  # shape (1, num_options)
        option_dist = torch.distributions.Categorical(logits=option_logits)
        if eval_mode:
            option = torch.argmax(option_logits, dim=1)
            logp_option = None
            option_entropy = None
        else:
            option = option_dist.sample()
            logp_option = option_dist.log_prob(option)
            option_entropy = option_dist.entropy()

        # low-level action logits
        action_logits = self.policy_net(state_tensor).squeeze(0)  # shape (output_dim,)
        state_np = state_tensor.squeeze(0).detach().cpu().numpy()
        mask = build_option_mask(state_np, int(option.item()), self.output_dim).to(self.device)

        # apply candidate mask (if provided) which contains 1.0 for feasible words, 0.0 otherwise
        if candidate_mask is not None:
            # ensure it's a tensor on the right device
            if isinstance(candidate_mask, np.ndarray):
                cm = torch.from_numpy(candidate_mask).to(self.device).float()
            else:
                cm = candidate_mask.to(self.device).float()
            mask = mask * cm

        # mask out already tried actions (set them to very small probabilities)
        if len(self._tried_actions) > 0:
            tried_idx_list = list(self._tried_actions)
            mask = mask.clone()
            tiny = 1e-8
            for idx in tried_idx_list:
                if 0 <= idx < mask.shape[0]:
                    mask[idx] = tiny

        # combine logits with multiplicative mask by adding log(mask)
        eps = 1e-8
        masked_logits = action_logits + torch.log(mask + eps)

        action_dist = torch.distributions.Categorical(logits=masked_logits)

        if eval_mode:
            action = torch.argmax(masked_logits).unsqueeze(0)
            return int(action.item()), None
        else:
            action = action_dist.sample()
            logp_action = action_dist.log_prob(action)
            action_entropy = action_dist.entropy()
            return int(action.item()), logp_action, logp_option, action_entropy, option_entropy

    def update(
        self,
        rewards,
        logps_actions,
        logps_options,
        values,
        entropies,
        episode_lengths: Optional[list] = None,
    ):
        """
        Supports batched updates. If episode_lengths is provided, rewards/logps/... are concatenations
        of multiple episodes and episode_lengths lists the lengths of each episode in steps.
        """

        total_steps = len(rewards)
        if total_steps <= 1:
            return

        device = self.device

        # compute discounted returns per episode
        if episode_lengths is None:
            # single-episode behaviour
            Gs = []
            G = 0.0
            for r in reversed(rewards):
                G = r + self.gamma * G
                Gs.append(G)
            Gs.reverse()
            Gs = torch.tensor(Gs, dtype=torch.float32, device=device)
        else:
            # concatenated episodes: compute returns per episode and concat
            Gs_list = []
            idx = 0
            for length in episode_lengths:
                G = 0.0
                sub_rewards = rewards[idx: idx + length]
                sub_Gs = []
                for r in reversed(sub_rewards):
                    G = r + self.gamma * G
                    sub_Gs.append(G)
                sub_Gs.reverse()
                Gs_list.extend(sub_Gs)
                idx += length
            Gs = torch.tensor(Gs_list, dtype=torch.float32, device=device)

        # value estimates
        values_tensor = torch.stack(values).squeeze(1).to(device)

        # advantages
        advantages = Gs - values_tensor.detach()

        # normalize advantages to reduce variance
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logprobs_actions = torch.stack(logps_actions).to(device)
        logprobs_options = torch.stack(logps_options).to(device)

        loss_low = -(logprobs_actions * advantages).sum()
        loss_high = -(logprobs_options * advantages).sum()
        baseline_loss = torch.sum((Gs - values_tensor) ** 2)

        # entropy regularization: entropies is a list of scalars (action_entropy + option_entropy per step)
        if len(entropies) > 0:
            entropy_tensor = torch.stack(entropies).sum().to(device)
        else:
            entropy_tensor = torch.tensor(0.0, device=device)

        total_loss = loss_low + loss_high + baseline_loss - self.entropy_coeff * entropy_tensor

        self.policy_optimizer.zero_grad()
        self.baseline_optimizer.zero_grad()
        total_loss.backward()

        # gradient clipping
        params = list(self.policy_net.parameters()) + list(self.highlevel_net.parameters()) + list(self.baseline_net.parameters())
        nn_utils.clip_grad_norm_(params, max_norm=self.grad_clip)

        self.policy_optimizer.step()
        self.baseline_optimizer.step()

    def collect_episode(self, env):
        """Run one episode (no learning), return collected lists for training."""
        self.start_episode()
        state, _ = env.reset()
        logps_actions = []
        logps_options = []
        rewards = []
        values = []
        entropies = []

        done = False
        first_action = None

        while not done:
            # compute candidate mask using the environment helper
            cand_mask_np = None
            try:
                cand_mask_np = Wordle_Env.candidate_mask(state, env.unwrapped.words)
            except Exception:
                cand_mask_np = None

            action, logp_action, logp_option, action_entropy, option_entropy = self.act(state, eval_mode=False, candidate_mask=cand_mask_np)
            # capture first action
            if first_action is None:
                first_action = int(action)
            self.record_action(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.baseline_net(state_tensor)

            values.append(value)
            logps_actions.append(logp_action)
            logps_options.append(logp_option)
            rewards.append(reward)

            ent = action_entropy
            if option_entropy is not None:
                ent = ent + option_entropy
            entropies.append(ent)

            state = next_state

        sum_rewards = sum(rewards)
        is_win = 1 if sum_rewards > 0 else 0

        mean_entropy = float(np.mean([e.item() for e in entropies])) if len(entropies) > 0 else 0.0

        return {
            "rewards": rewards,
            "logps_actions": logps_actions,
            "logps_options": logps_options,
            "values": values,
            "entropies": entropies,
            "sum_rewards": sum_rewards,
            "is_win": is_win,
            "length": len(rewards),
            "first_action": first_action,
            "mean_entropy": mean_entropy,
        }


if __name__ == "__main__":
    hidden_layer_size = (128, 128)
    learning_rate = 3e-4
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
    print(f"Starting hierarchical training on {env_name}")
    print(f"State Dim {input_dim}, Action Dim {output_dim}")
    print(f"Using device {device}")

    agent = HierarchicalAgent(
        input_dim, output_dim, device,
        learning_rate, gamma, record,
        hidden_layer_size
    )

    for episode in range(1, episodes + 1):
        data = agent.collect_episode(env)
        agent.recent_rewards.append(data["sum_rewards"])
        agent.recent_wins.append(data["is_win"])
        agent.update(
            data["rewards"],
            data["logps_actions"],
            data["logps_options"],
            data["values"],
            data["entropies"],
        )
        if episode % record == 0:
            Wordle_Env.print_agent_train_log(
                agent.recent_rewards, agent.recent_wins, episode
            )