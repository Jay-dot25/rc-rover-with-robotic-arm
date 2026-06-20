"""
plot_results.py — Training visualisation (v2)
=============================================
Generates a 6-panel dashboard + individual plots.
"""

import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Dark theme
BG    = "#0d1117"
PANEL = "#161b22"
EDGE  = "#30363d"
GREEN = "#3fb950"
BLUE  = "#58a6ff"
RED   = "#f85149"
ORANGE= "#f5a623"
GRAY  = "#8b949e"
WHITE = "#e6edf3"
PURPLE= "#d2a8ff"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": EDGE, "axes.labelcolor": WHITE,
    "axes.titlecolor": WHITE, "xtick.color": GRAY, "ytick.color": GRAY,
    "grid.color": "#21262d", "grid.linewidth": 0.6,
    "text.color": WHITE, "font.family": "monospace", "font.size": 10,
    "lines.linewidth": 1.5,
})


def smooth(arr, w=30):
    if len(arr) < w:
        return np.array(arr, dtype=float)
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")


def generate_all_plots(rewards, goals, collisions, epsilons, losses,
                       steps, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    eps_x = np.arange(1, len(rewards) + 1)

    # ── 1. Summary dashboard (6 panels) ──────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle("RL Rover v2 — Double Dueling DQN Training Dashboard",
                 fontsize=16, color=WHITE, fontweight="bold", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    # Reward
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eps_x, rewards, color=BLUE, alpha=0.25, lw=0.8)
    if len(rewards) >= 30:
        sx = eps_x[29:]
        ax1.plot(sx, smooth(rewards, 30), color=BLUE, lw=2, label="smoothed")
    ax1.axhline(0, color=EDGE, lw=0.8, ls="--")
    ax1.set_title("Episode Reward"); ax1.set_xlabel("Episode"); ax1.set_ylabel("Total Reward")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Goal rate
    ax2 = fig.add_subplot(gs[0, 1])
    goal_rate = [np.mean(goals[max(0,i-49):i+1])*100 for i in range(len(goals))]
    ax2.fill_between(eps_x, goal_rate, alpha=0.3, color=GREEN)
    ax2.plot(eps_x, goal_rate, color=GREEN, lw=2)
    ax2.axhline(80, color=ORANGE, lw=1, ls="--", label="80% target")
    ax2.set_title("Goal Rate (50-ep window)")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Goal %")
    ax2.set_ylim(0, 105); ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Collision rate
    ax3 = fig.add_subplot(gs[0, 2])
    col_rate = [np.mean(collisions[max(0,i-49):i+1])*100 for i in range(len(collisions))]
    ax3.fill_between(eps_x, col_rate, alpha=0.3, color=RED)
    ax3.plot(eps_x, col_rate, color=RED, lw=2)
    ax3.set_title("Collision Rate (50-ep window)")
    ax3.set_xlabel("Episode"); ax3.set_ylabel("Collision %")
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)

    # Loss
    ax4 = fig.add_subplot(gs[1, 0])
    valid_loss = [(i, l) for i, l in zip(eps_x, losses) if l > 0]
    if valid_loss:
        lx, ly = zip(*valid_loss)
        ax4.plot(lx, ly, color=ORANGE, alpha=0.3, lw=0.8)
        if len(ly) >= 30:
            ax4.plot(list(lx)[29:], smooth(list(ly), 30), color=ORANGE, lw=2)
    ax4.set_title("Training Loss"); ax4.set_xlabel("Episode"); ax4.set_ylabel("MSE Loss")
    ax4.grid(True, alpha=0.3)

    # Epsilon
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(eps_x, epsilons, color=PURPLE, lw=2)
    ax5.fill_between(eps_x, epsilons, alpha=0.2, color=PURPLE)
    ax5.set_title("Exploration (ε)")
    ax5.set_xlabel("Episode"); ax5.set_ylabel("Epsilon")
    ax5.grid(True, alpha=0.3)

    # Steps per episode
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(eps_x, steps, color=GRAY, alpha=0.3, lw=0.8)
    if len(steps) >= 30:
        ax6.plot(eps_x[29:], smooth(steps, 30), color=WHITE, lw=2)
    ax6.set_title("Steps per Episode")
    ax6.set_xlabel("Episode"); ax6.set_ylabel("Steps")
    ax6.grid(True, alpha=0.3)

    # Final stats annotation
    n = min(200, len(rewards))
    stats_text = (f"Last {n} eps:  "
                  f"Goal {np.mean(goals[-n:])*100:.1f}%  |  "
                  f"Coll {np.mean(collisions[-n:])*100:.1f}%  |  "
                  f"Avg Reward {np.mean(rewards[-n:]):+.2f}")
    fig.text(0.5, 0.01, stats_text, ha="center", color=GRAY, fontsize=10)

    fig.savefig(os.path.join(out_dir, "summary_dashboard.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Individual plots ──────────────────────────────────────────────────────
    def single(data, color, title, ylabel, fname, ref=None):
        fig2, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(1, len(data) + 1)
        ax.plot(x, data, color=color, alpha=0.25, lw=0.8)
        if len(data) >= 30:
            ax.plot(x[29:], smooth(data, 30), color=color, lw=2)
        if ref is not None:
            ax.axhline(ref[0], color=ref[1], lw=1, ls="--", label=ref[2])
            ax.legend()
        ax.set_title(title); ax.set_xlabel("Episode"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig2)

    single(rewards,  BLUE,   "Episode Reward",    "Total Reward",  "reward_curve.png")
    single(goal_rate,GREEN,  "Goal Rate",          "Goal %",        "goal_rate.png",   (80, ORANGE, "80% target"))
    single(epsilons, PURPLE, "Epsilon Decay",      "Epsilon",       "epsilon_decay.png")
    single([l for l in losses if l > 0], ORANGE, "Training Loss", "MSE Loss", "loss_curve.png")

    print(f"  [Plots] saved to {out_dir}/")
