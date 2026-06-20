# RL Rover v2 — Double Dueling DQN
### Real-time Pygame Simulation · Prioritized Replay · Curriculum Learning

---

## What's new in v2

| Feature | v1 (old) | v2 (new) |
|---|---|---|
| **Simulation** | Saves MP4, no live view | **Live Pygame window** — see it in real time |
| **Agent** | Vanilla DQN | **Double Dueling DQN** |
| **Replay** | Uniform random | **Prioritized Experience Replay** |
| **Sensors** | 3 rays | **8 rays (270° arc)** |
| **Actions** | 3 (fwd/L/R) | **5 (+ hard-L/R)** |
| **State dim** | 5 | **12** (velocity, angular vel) |
| **Training** | 800 episodes | **2000 episodes + curriculum** |
| **Network** | 5→128→64→3 | **12→256→256→128→[V\|A]→5 (Dueling)** |
| **Reward** | Sparse shaping | **Dense: alignment, wall penalty, progress** |
| **Curriculum** | None | **Easy→Medium→Hard auto-progression** |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

On Linux you may need SDL for Pygame:
```bash
sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
```

### 2. Train the agent (≈ 10–20 min)

```bash
python run_all.py
```

Or separately:
```bash
python src/train.py        # train only
python src/simulate.py     # live window after training
```

### 3. Live simulation controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `R` | Reset episode |
| `S` | Toggle sensor rays |
| `T` | Toggle trail |
| `+` / `-` | Speed up / slow down |
| `Q` / `ESC` | Quit |

---

## Architecture

### Double DQN
Instead of one network, two networks:
- **Online network** — picks the best action: `a* = argmax Q_online(s')`
- **Target network** — evaluates it: `Q_target(s', a*)`

This removes the overestimation bias of vanilla DQN.

### Dueling Architecture
The network splits into two heads:
```
                    ┌─ Value head ─────► V(s)       (scalar)
Input → Backbone ──┤
                    └─ Advantage head ─► A(s, a)    (per-action)

Q(s,a) = V(s) + A(s,a) − mean_a(A(s,a))
```
The value head learns *how good the state is*.
The advantage head learns *which action is better*.
This separation dramatically speeds up learning.

### Prioritized Experience Replay (PER)
- Transitions with **high TD-error** are sampled more often
- The rover learns more from its *mistakes* and *surprises*
- Importance-sampling weights prevent bias
- Priority exponent α=0.6, annealed β: 0.4→1.0

### Curriculum Learning
```
Episodes 0–399   → EASY   (4 obstacles)
Episodes 400–899 → MEDIUM (6 obstacles)
Episodes 900+    → HARD   (9 obstacles)
```

### Reward Function
```
+20.0   goal reached     (terminal)
 −8.0   collision        (terminal)
 +0.3   moved closer to goal (per forward step)
 −0.05  moved away from goal
 +0.1   facing goal (|θ_goal| < 0.2 rad)
 −0.1   about to hit wall (front sensor < 0.2, forward action)
 −0.05  time penalty per step
```

---

## Project structure

```
rl_rover_v2/
├── run_all.py              ← single entry point
├── requirements.txt
├── README.md
├── src/
│   ├── env.py              ← RoverEnv (8 sensors, 5 actions, 12-D state)
│   ├── agent.py            ← DDQNAgent (Double + Dueling + PER)
│   ├── train.py            ← Training loop with curriculum
│   ├── simulate.py         ← Pygame live window + video recorder
│   └── plot_results.py     ← 6-panel dashboard + individual charts
└── outputs/                ← auto-created
    ├── model.npz
    ├── model_meta.json
    ├── training_log.csv
    ├── summary.json
    ├── summary_dashboard.png
    ├── reward_curve.png
    ├── goal_rate.png
    ├── epsilon_decay.png
    └── loss_curve.png
```

---

## Common issues

**Pygame not found:**
```bash
pip install pygame
```

**No display (SSH/headless):**
```bash
python src/simulate.py --video    # saves MP4 instead
```

**Model not reaching goal after training:**
- Train longer: edit `N_EPISODES = 3000` in `train.py`
- Check `outputs/summary_dashboard.png` — goal rate should be >60% by ep 1500
- Try `--difficulty easy` to validate the model works at all

---

## Authors

Sriramaneni Suhas · Sharuk · B. Jayanth · K. Jayanth · Ch. Rishi · S. Dinesh  
Department of Computer Science & Engineering · VIT-AP University
