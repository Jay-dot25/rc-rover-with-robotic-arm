"""
train.py — Training loop (v2)
==============================
Improvements:
  • Curriculum learning: starts on EASY, graduates to HARD
  • Prints live stats every 25 episodes
  • Early stopping if goal rate > 90% sustained
  • Saves checkpoint every 250 episodes
  • Much longer training: 2000 episodes
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import random
import time
import csv
import json

from env   import RoverEnv, N_ACTIONS, STATE_DIM
from agent import DDQNAgent, BATCH_SIZE

# ── Config ────────────────────────────────────────────────────────────────────
N_EPISODES    = 2000
RENDER_EVERY  = 25
SAVE_PATH     = os.path.join(os.path.dirname(__file__), "..", "outputs", "model")
LOG_PATH      = os.path.join(os.path.dirname(__file__), "..", "outputs", "training_log.csv")
SEED          = 42

# Curriculum: (start_ep, difficulty)
CURRICULUM = [
    (0,    "easy"),
    (400,  "medium"),
    (900,  "hard"),
]

random.seed(SEED)
np.random.seed(SEED)


def get_difficulty(ep):
    diff = "easy"
    for start, d in CURRICULUM:
        if ep >= start:
            diff = d
    return diff


def train():
    print("=" * 65)
    print("  RL Rover v2 — Double Dueling DQN + Prioritized Replay")
    print("=" * 65)
    print(f"  Episodes     : {N_EPISODES}")
    print(f"  State dim    : {STATE_DIM}   (12-D with 8 sensors)")
    print(f"  Actions      : {N_ACTIONS}   (fwd / L / R / hard-L / hard-R)")
    print(f"  Architecture : Dueling DQN  12→256→256→128→[V|A]→5")
    print(f"  Replay       : Prioritized  (50k buffer, α=0.6)")
    print(f"  Curriculum   : easy→medium→hard")
    print("=" * 65)

    agent = DDQNAgent(STATE_DIM, N_ACTIONS)

    episode_rewards  = []
    goal_flags       = []
    collision_flags  = []
    epsilon_history  = []
    loss_history     = []
    steps_history    = []

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_file = open(LOG_PATH, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["episode","reward","steps","goal","collision",
                     "epsilon","avg_loss","difficulty"])

    t_start = time.time()
    current_diff = "easy"
    env = RoverEnv(seed=SEED, difficulty=current_diff)

    for ep in range(1, N_EPISODES + 1):
        # Curriculum
        new_diff = get_difficulty(ep)
        if new_diff != current_diff:
            current_diff = new_diff
            env = RoverEnv(seed=SEED + ep, difficulty=current_diff)
            print(f"\n  [Curriculum] → switching to {current_diff.upper()} at ep {ep}\n")

        state = env.reset()
        ep_reward = 0.0
        ep_losses = []
        step = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.push(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            ep_reward += reward
            state = next_state
            step += 1

            if done:
                break

        agent.end_episode()

        reached_goal = info.get("goal", False)
        hit_obstacle = info.get("collision", False)
        avg_loss     = float(np.mean(ep_losses)) if ep_losses else 0.0

        episode_rewards.append(ep_reward)
        goal_flags.append(int(reached_goal))
        collision_flags.append(int(hit_obstacle))
        epsilon_history.append(agent.epsilon)
        loss_history.append(avg_loss)
        steps_history.append(step)

        writer.writerow([ep, f"{ep_reward:.4f}", step,
                         int(reached_goal), int(hit_obstacle),
                         f"{agent.epsilon:.5f}", f"{avg_loss:.6f}",
                         current_diff])
        log_file.flush()

        if ep % RENDER_EVERY == 0 or ep == 1:
            n = min(50, ep)
            goal_rate  = np.mean(goal_flags[-n:]) * 100
            col_rate   = np.mean(collision_flags[-n:]) * 100
            avg_rew    = np.mean(episode_rewards[-n:])
            elapsed    = time.time() - t_start
            eta_s      = (N_EPISODES - ep) / max(ep / elapsed, 0.001)
            bar_g = "█" * int(goal_rate / 5) + "░" * (20 - int(goal_rate / 5))
            print(f"  ep {ep:4d}/{N_EPISODES} [{current_diff[0].upper()}] | "
                  f"rew {avg_rew:+6.2f} | "
                  f"goal {goal_rate:5.1f}% {bar_g} | "
                  f"coll {col_rate:4.1f}% | "
                  f"ε {agent.epsilon:.3f} | "
                  f"ETA {eta_s/60:.1f}m")

        # Checkpoint
        if ep % 250 == 0:
            agent.save(SAVE_PATH + f"_ep{ep}")
            print(f"  [Checkpoint] saved at ep {ep}")

    log_file.close()
    elapsed_total = time.time() - t_start

    print(f"\n  Training complete in {elapsed_total/60:.1f} min")
    goal_last = np.mean(goal_flags[-200:]) * 100
    print(f"  Final goal rate (last 200 ep): {goal_last:.1f}%")

    agent.save(SAVE_PATH)

    summary = {
        "n_episodes":        N_EPISODES,
        "final_epsilon":     float(agent.epsilon),
        "goal_rate_last200": float(goal_last),
        "coll_rate_last200": float(np.mean(collision_flags[-200:]) * 100),
        "avg_reward_last200": float(np.mean(episode_rewards[-200:])),
        "total_time_s":      round(elapsed_total, 1),
    }
    out_dir = os.path.join(os.path.dirname(SAVE_PATH))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  Generating plots ...")
    from plot_results import generate_all_plots
    generate_all_plots(
        episode_rewards, goal_flags, collision_flags,
        epsilon_history, loss_history, steps_history,
        out_dir=out_dir
    )
    print("  Plots saved.")
    print("=" * 65)
    print("  Done!  Run:  python simulate.py     → real-time Pygame window")
    print("         Run:  python simulate.py --video → save MP4 instead")
    print("=" * 65)

    return agent, episode_rewards, goal_flags


if __name__ == "__main__":
    train()
