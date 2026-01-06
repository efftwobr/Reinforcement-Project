"""
Evaluation script for HierarchicalAgent on Wordle environments.

Usage examples:
  python evaluate_agent.py --model_dir ./models --envs WordleEnv100WithMask WordleEnv100 --episodes 500 --out_dir ./eval_out

It looks for (in model_dir):
  policy_net.pt
  highlevel_net.pt
  baseline_net.pt

If the files are not present, the script will run the agent instance without loading weights (useful if you create the agent in-memory).
"""
import os
import argparse
import csv
from collections import Counter, defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import project modules
import importlib
import Wordle_Env
import Models.Hierarchical_PG as hp
importlib.reload(Wordle_Env)
importlib.reload(hp)

def candidate_mask_from_state(state, words):
    """
    Conservative per-position candidate mask computed from Wordle state encoding.
    Returns a numpy float32 array of shape (len(words),) with 1.0 feasible, 0.0 infeasible.
    This function mirrors the state encoding used in Wordle_Env.
    """
    if state is None:
        return np.ones(len(words), dtype=np.float32)

    Word_Len = Wordle_Env.Word_Len
    Chars = Wordle_Env.Chars
    mask = np.ones(len(words), dtype=np.float32)

    for idx, w in enumerate(words):
        w = w.upper()
        if len(w) != Word_Len:
            mask[idx] = 0.0
            continue
        feasible = True
        # Per-position check using state's triples for each character
        for i, ch in enumerate(w):
            if ord(ch) - ord(Chars[0]) < 0 or ord(ch) - ord(Chars[0]) >= len(Chars):
                feasible = False
                break
            cint = ord(ch) - ord(Chars[0])
            offset = 1 + len(Chars) + cint * Word_Len * 3
            triple = state[offset + 3 * i: offset + 3 * i + 3]
            # triple[0] == 1 means explicitly excluded at this position
            if int(triple[0]) == 1:
                feasible = False
                break
        mask[idx] = 1.0 if feasible else 0.0
    return mask

def load_agent_and_weights(env_name, model_dir, device, hidden_layer_size=(256,256), entropy_coeff=0.03):
    # create env to get input/output dims
    env, input_dim, output_dim = Wordle_Env.create_env(env_name)
    agent = hp.HierarchicalAgent(
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
        learning_rate=3e-4,
        gamma=0.99,
        record=1000,
        hidden_layer_size=hidden_layer_size,
        entropy_coeff=entropy_coeff,
    )
    # Attempt to load weights
    loaded_any = False
    policy_path = os.path.join(model_dir, "policy_net.pt")
    high_path = os.path.join(model_dir, "highlevel_net.pt")
    base_path = os.path.join(model_dir, "baseline_net.pt")
    # alternative: load a single file 'agent_all.pth' with dict of states
    all_path = os.path.join(model_dir, "agent_all.pth")

    try:
        if os.path.isfile(policy_path):
            agent.policy_net.load_state_dict(torch.load(policy_path, map_location=device))
            loaded_any = True
        if os.path.isfile(high_path):
            agent.highlevel_net.load_state_dict(torch.load(high_path, map_location=device))
            loaded_any = True
        if os.path.isfile(base_path):
            agent.baseline_net.load_state_dict(torch.load(base_path, map_location=device))
            loaded_any = True
        if not loaded_any and os.path.isfile(all_path):
            d = torch.load(all_path, map_location=device)
            if "policy" in d and "high" in d and "baseline" in d:
                agent.policy_net.load_state_dict(d["policy"])
                agent.highlevel_net.load_state_dict(d["high"])
                agent.baseline_net.load_state_dict(d["baseline"])
                loaded_any = True
    except Exception as e:
        print("Warning: error loading model weights:", e)
    if loaded_any:
        print(f"Loaded agent weights from {model_dir}")
    else:
        print("No weights loaded; using randomly initialized agent (use --model_dir to point to saved weights).")
    return agent

def evaluate(agent, env_name, episodes, out_dir, save_examples=10):
    env, _, _ = Wordle_Env.create_env(env_name)
    words = env.unwrapped.words
    results_csv = os.path.join(out_dir, f"{env_name}_results.csv")
    rows = []
    first_actions = []
    steps_list = []
    candidate_counts_by_step = defaultdict(list)
    examples = []

    for ep in range(1, episodes + 1):
        agent.start_episode()
        state, _ = env.reset()
        done = False
        ep_guesses = []
        ep_candidate_counts = []
        first_action_word = None
        steps = 0
        win = 0

        while not done:
            cand_mask_np = candidate_mask_from_state(state, words)
            # Agent.act signature supports candidate_mask kw
            try:
                action, _ = agent.act(state, eval_mode=True, candidate_mask=cand_mask_np)
            except TypeError:
                # fallback if act doesn't accept candidate_mask
                action, _ = agent.act(state, eval_mode=True)
            # record to prevent repeats
            if hasattr(agent, "record_action"):
                try:
                    agent.record_action(action)
                except Exception:
                    pass

            if first_action_word is None:
                first_action_word = words[action] if 0 <= action < len(words) else str(action)
                first_actions.append(first_action_word)

            ep_guesses.append(words[action] if 0 <= action < len(words) else str(action))
            ep_candidate_counts.append(int(cand_mask_np.sum()))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            steps += 1
            if done and reward > 0:
                win = 1

        steps_list.append(steps)
        for i, c in enumerate(ep_candidate_counts):
            candidate_counts_by_step[i].append(c)

        rows.append({
            "episode": ep,
            "win": win,
            "steps": steps,
            "first_action": first_action_word,
            "guesses": ";".join(ep_guesses),
            "candidate_counts": ";".join(map(str, ep_candidate_counts))
        })

        # save a few example episodes as text
        if len(examples) < save_examples:
            examples.append((ep, first_action_word, ep_guesses, ep_candidate_counts, win))

    # write CSV
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode","win","steps","first_action","guesses","candidate_counts"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved per-episode results to {results_csv}")

    # Summary stats
    wins = sum(r["win"] for r in rows)
    win_rate = wins / max(1, episodes)
    avg_steps = np.mean(steps_list) if len(steps_list) > 0 else 0.0
    print(f"Env {env_name} | Episodes: {episodes} | Win rate: {win_rate*100:.2f}% | Avg steps: {avg_steps:.2f}")

    # First-action stats
    fa_counts = Counter(first_actions)
    top_first = fa_counts.most_common(20)
    # Plot top-first bar chart
    labels = [w for w,c in top_first]
    values = [c for w,c in top_first]
    plt.figure(figsize=(10,5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title(f"Top first guesses ({env_name})")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{env_name}_top_first_guesses.png"))
    plt.close()

    # Steps histogram
    plt.figure(figsize=(6,4))
    plt.hist(steps_list, bins=np.arange(1, env.unwrapped.max_turns+2)-0.5, edgecolor='black')
    plt.xlabel("Steps to finish")
    plt.ylabel("Number of episodes")
    plt.title(f"Steps histogram ({env_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{env_name}_steps_hist.png"))
    plt.close()

    # Average candidate count by step (mean ± std)
    max_step = max(candidate_counts_by_step.keys()) if len(candidate_counts_by_step)>0 else 0
    means = []
    stds = []
    xs = []
    for i in range(max_step+1):
        vals = candidate_counts_by_step.get(i, [])
        if len(vals) == 0:
            means.append(np.nan)
            stds.append(0.0)
        else:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        xs.append(i+1)  # step number (1-based)

    plt.figure(figsize=(6,4))
    plt.errorbar(xs, means, yerr=stds, marker='o', capsize=3)
    plt.xlabel("Step number")
    plt.ylabel("Avg candidate words remaining")
    plt.title(f"Candidates remaining by step ({env_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{env_name}_candidates_by_step.png"))
    plt.close()

    # Save example episodes
    ex_file = os.path.join(out_dir, f"{env_name}_examples.txt")
    with open(ex_file, "w", encoding="utf-8") as f:
        for ep, fa, guesses, cand_counts, win in examples:
            f.write(f"Episode {ep} | win={win} | first={fa} | guesses={','.join(guesses)} | candidate_counts={','.join(map(str,cand_counts))}\n")
    print(f"Saved {len(examples)} example episodes to {ex_file}")

    summary = {
        "env": env_name,
        "episodes": episodes,
        "win_rate": win_rate,
        "avg_steps": avg_steps,
        "top_first": top_first[:10],
    }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory with saved net state_dicts")
    parser.add_argument("--envs", nargs="+", default=["WordleEnv100WithMask"], help="Env names to evaluate")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per env")
    parser.add_argument("--out_dir", type=str, default="./eval_out", help="Folder to write results/plots")
    parser.add_argument("--device", type=str, default="cpu", help="torch device")
    parser.add_argument("--top_k", type=int, default=10, help="How many top first guesses to plot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    summaries = []
    for env_name in args.envs:
        print(f"\nEvaluating env {env_name} for {args.episodes} episodes...")
        agent = load_agent_and_weights(env_name, args.model_dir, device)
        summary = evaluate(agent, env_name, args.episodes, args.out_dir, save_examples=10)
        summaries.append(summary)

    # Save overall summary CSV
    sum_csv = os.path.join(args.out_dir, "evaluation_summary.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["env","episodes","win_rate","avg_steps","top_first"])
        writer.writeheader()
        for s in summaries:
            writer.writerow({
                "env": s["env"],
                "episodes": s["episodes"],
                "win_rate": f"{s['win_rate']:.4f}",
                "avg_steps": f"{s['avg_steps']:.4f}",
                "top_first": ";".join([f"{w}:{c}" for w,c in s["top_first"]])
            })
    print(f"\nSaved summary to {sum_csv}")
    print("Done.")

if __name__ == "__main__":
    main()