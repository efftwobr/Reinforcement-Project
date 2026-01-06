import torch
from collections import Counter, deque
from Wordle_Env import create_env, print_agent_train_log, test_agent

from Models.Hierarchical_PG import HierarchicalAgent

def print_first_action_stats(first_actions_deque, env, top_k=10):
    if len(first_actions_deque) == 0:
        return
    counts = Counter(first_actions_deque)
    most_common = counts.most_common(top_k)
    print("Top first-guess words (count):")
    for idx, cnt in most_common:
        # guard if env words shorter
        word = env.unwrapped.words[idx] if 0 <= idx < len(env.unwrapped.words) else str(idx)
        print(f"  {word}: {cnt}")


if __name__ == "__main__":
    # Use mask-based env for richer state information
    env_name = "WordleEnvReal"
    env, input_dim, output_dim = create_env(env_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters — adjust if needed
    learning_rate = 3e-4
    gamma = 0.99
    episodes = 5000
    record = 1000
    hidden_layer_size = (256, 256)

    # Batch update: collect this many episodes, then call update on concatenated trajectories
    batch_size = 32
    entropy_coeff = 0.03  # increase exploration slightly

    agent = HierarchicalAgent(
        input_dim, output_dim, device,
        learning_rate, gamma, record,
        hidden_layer_size,
        entropy_coeff=entropy_coeff,
    )

    # keep last-first-actions for diagnostics
    recent_first_actions = deque(maxlen=record)
    recent_entropies = deque(maxlen=record)

    episode = 0
    while episode < episodes:
        # collect a batch of episodes
        batch_rewards = []
        batch_logps_actions = []
        batch_logps_options = []
        batch_values = []
        batch_entropies = []
        episode_lengths = []
        for b in range(batch_size):
            if episode >= episodes:
                break
            data = agent.collect_episode(env)
            episode += 1

            # record scalar episode stats
            agent.recent_rewards.append(data["sum_rewards"])
            agent.recent_wins.append(data["is_win"])
            recent_first_actions.append(data["first_action"])
            recent_entropies.append(data["mean_entropy"])

            # accumulate trajectories
            batch_rewards.extend(data["rewards"])
            batch_logps_actions.extend(data["logps_actions"])
            batch_logps_options.extend(data["logps_options"])
            batch_values.extend(data["values"])
            batch_entropies.extend(data["entropies"])
            episode_lengths.append(data["length"])

            if episode % record == 0:
                print_agent_train_log(agent.recent_rewards, agent.recent_wins, episode)
                # diagnostics
                avg_entropy = sum(recent_entropies) / len(recent_entropies) if len(recent_entropies) > 0 else 0.0
                print(f"Avg episode entropy (recent {len(recent_entropies)}): {avg_entropy:.4f}")
                print_first_action_stats(recent_first_actions, env, top_k=10)

        # do a single update over the whole batch (update handles episode_lengths)
        if len(batch_rewards) > 0:
            agent.update(
                batch_rewards,
                batch_logps_actions,
                batch_logps_options,
                batch_values,
                batch_entropies,
                episode_lengths=episode_lengths,
            )

    print("\nTraining finished. Running a test game with the trained agent...")
    test_agent(env, agent)
    env.close()