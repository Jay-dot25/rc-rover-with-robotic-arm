"""
simulate.py — Real-time Pygame Simulation (v2)
===============================================
Run AFTER training:
    python simulate.py                     # live window (default)
    python simulate.py --video             # save MP4 to outputs/
    python simulate.py --episodes 10       # run 10 episodes
    python simulate.py --random            # show random agent
    python simulate.py --difficulty hard   # choose difficulty

Controls (live mode):
    SPACE      pause / resume
    R          reset episode
    S          toggle sensor rays
    T          toggle trail
    Q / ESC    quit
    +/-        adjust speed (0.5x – 4x)

What you see:
    • Dark grid background
    • Rover (animated triangle with direction arrow)
    • Goal (pulsing green circle)
    • 8 sensor rays (color = proximity: green→yellow→red)
    • HUD panel: episode stats, Q-values, reward bar
    • Trail of visited positions
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
import numpy as np
import argparse
import time

from env   import (RoverEnv, WORLD_W, WORLD_H, ROVER_R, GOAL_R,
                   STATE_DIM, N_ACTIONS, ACTION_NAMES, N_SENSORS)
from agent import DDQNAgent

# ── Try importing Pygame ──────────────────────────────────────────────────────
try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "model")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "outputs")
SEED       = 99

# ── Colour palette (dark theme) ───────────────────────────────────────────────
C_BG        = (13,  17,  23)
C_GRID      = (22,  27,  34)
C_PANEL     = (22,  27,  34)
C_OBSTACLE  = (48,  54,  61)
C_OBS_EDGE  = (99, 110, 123)
C_GOAL      = (63, 185, 80)
C_GOAL_RING = (45, 140, 60)
C_ROVER     = (88, 166, 255)
C_ROVER_FWD = (255, 255, 255)
C_TRAIL     = (40,  70, 120)
C_SENSOR_OK = (80, 220, 100)
C_SENSOR_MID= (255, 200,  50)
C_SENSOR_BAD= (248,  81,  73)
C_TEXT      = (230, 237, 243)
C_TEXT_DIM  = (139, 148, 158)
C_REWARD_POS= (63, 185, 80)
C_REWARD_NEG= (248,  81,  73)
C_WHITE     = (255, 255, 255)
C_YELLOW    = (255, 220,  80)
C_PURPLE    = (180, 120, 255)

HUD_W  = 280   # right panel width
FPS    = 60


# ── Fallback: matplotlib video recorder ───────────────────────────────────────

def record_video_matplotlib(agent, n_episodes, out_path, seed, difficulty):
    """Fall back if Pygame not available: record MP4 with matplotlib+opencv."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    env = RoverEnv(seed=seed, difficulty=difficulty)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, 30, (WORLD_W, WORLD_H))

    fig, ax = plt.subplots(figsize=(WORLD_W/100, WORLD_H/100), dpi=100)
    fig.patch.set_facecolor("#0d1117")

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        trail = [(env.rover["x"], env.rover["y"])]
        ep_reward = 0.0
        step = 0

        while True:
            action = (agent.select_action(state, greedy=True)
                      if agent else np.random.randint(N_ACTIONS))
            state, reward, done, info = env.step(action)
            trail.append((env.rover["x"], env.rover["y"]))
            ep_reward += reward
            step += 1

            # Draw frame
            ax.clear()
            ax.set_facecolor("#161b22")
            ax.set_xlim(0, WORLD_W); ax.set_ylim(WORLD_H, 0)
            ax.set_aspect("equal"); ax.axis("off")

            for (ox, oy, ow, oh) in env.obstacles:
                ax.add_patch(plt.Rectangle((ox,oy), ow, oh,
                    facecolor="#30363d", edgecolor="#8b949e", lw=1))

            # Trail
            if len(trail) > 1:
                tx, ty = zip(*trail)
                ax.plot(tx, ty, color="#1f4080", lw=1.2, alpha=0.6)

            # Goal (pulsing)
            pulse = 1.0 + 0.15 * math.sin(step * 0.2)
            ax.add_patch(plt.Circle((env.goal["x"], env.goal["y"]),
                GOAL_R * pulse, color="#3fb950", alpha=0.9))

            # Sensors
            for (px, py) in env.get_sensor_endpoints():
                ax.plot([env.rover["x"], px], [env.rover["y"], py],
                        color="#58a6ff", lw=0.6, alpha=0.3)

            # Rover triangle
            a = env.rover["angle"]
            pts = np.array([
                [env.rover["x"] + math.cos(a)*ROVER_R*1.5,
                 env.rover["y"] + math.sin(a)*ROVER_R*1.5],
                [env.rover["x"] + math.cos(a+2.4)*ROVER_R,
                 env.rover["y"] + math.sin(a+2.4)*ROVER_R],
                [env.rover["x"] + math.cos(a-2.4)*ROVER_R,
                 env.rover["y"] + math.sin(a-2.4)*ROVER_R],
            ])
            ax.add_patch(plt.Polygon(pts, facecolor="#58a6ff", edgecolor="white", lw=1))

            # HUD text
            color = "#3fb950" if info.get("goal") else ("#f85149" if info.get("collision") else "#e6edf3")
            ax.set_title(f"Ep {ep}  Step {step:3d}  Reward {ep_reward:+.1f}  "
                         f"Action: {ACTION_NAMES[action]}",
                         color=color, fontsize=9, fontfamily="monospace",
                         loc="left", pad=4)

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(WORLD_H, WORLD_W, 3)
            img_bgr = img[:, :, ::-1]
            vw.write(img_bgr)

            if done:
                break

        status = "GOAL!" if info.get("goal") else ("CRASH" if info.get("collision") else "timeout")
        print(f"  ep {ep}/{n_episodes}  {status}  reward={ep_reward:.1f}  steps={step}")

    vw.release()
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Pygame simulation ─────────────────────────────────────────────────────────

class PygameSimulator:
    def __init__(self, agent, difficulty="medium", seed=SEED,
                 record=False, record_path=None):
        pygame.init()
        self.total_w = WORLD_W + HUD_W
        self.total_h = WORLD_H
        self.screen  = pygame.display.set_mode((self.total_w, self.total_h))
        pygame.display.set_caption("RL Rover v2 — Double Dueling DQN")
        self.clock   = pygame.time.Clock()

        self.agent      = agent
        self.env        = RoverEnv(seed=seed, difficulty=difficulty)
        self.difficulty = difficulty
        self.seed       = seed

        self.paused      = False
        self.show_sensors= True
        self.show_trail  = True
        self.speed_mult  = 1.0
        self.episode     = 0
        self.total_goals = 0
        self.total_eps   = 0
        self.trail       = []
        self.reward_history = []
        self.goal_history   = []
        self.ep_reward   = 0.0
        self.step_count  = 0
        self.pulse_t     = 0

        # Fonts
        self.font_big  = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_mid  = pygame.font.SysFont("monospace", 12)
        self.font_sm   = pygame.font.SysFont("monospace", 10)

        # Recording
        self.record     = record
        self.video_writer = None
        if record and record_path:
            try:
                import cv2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(
                    record_path, fourcc, FPS, (self.total_w, self.total_h))
                self.cv2 = cv2
            except ImportError:
                print("  opencv-python not found; recording disabled.")
                self.record = False

        # Start first episode
        self.state    = self.env.reset()
        self.done     = False
        self.episode  = 1
        self.info     = {}

    def run(self, n_episodes=None):
        running = True
        eps_done = 0

        while running:
            # ── Events ────────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self._reset_episode()
                    elif event.key == pygame.K_s:
                        self.show_sensors = not self.show_sensors
                    elif event.key == pygame.K_t:
                        self.show_trail = not self.show_trail
                    elif event.key == pygame.K_EQUALS:
                        self.speed_mult = min(4.0, self.speed_mult * 1.5)
                    elif event.key == pygame.K_MINUS:
                        self.speed_mult = max(0.25, self.speed_mult / 1.5)

            if not self.paused and not self.done:
                # ── Step ──────────────────────────────────────────────────────
                action = (self.agent.select_action(self.state, greedy=True)
                          if self.agent else np.random.randint(N_ACTIONS))
                next_state, reward, done, info = self.env.step(action)
                self.trail.append((self.env.rover["x"], self.env.rover["y"]))
                self.ep_reward += reward
                self.step_count += 1
                self.state = next_state
                self.done  = done
                self.info  = info
                self.pulse_t += 1
                self.last_action = action
                self.last_reward = reward

            if self.done:
                self.total_eps += 1
                if self.info.get("goal"):
                    self.total_goals += 1
                self.reward_history.append(self.ep_reward)
                self.goal_history.append(int(self.info.get("goal", False)))
                eps_done += 1

                if n_episodes and eps_done >= n_episodes:
                    running = False
                    break

                # Auto-reset after short pause
                time.sleep(0.4 / self.speed_mult)
                self._reset_episode()

            # ── Draw ──────────────────────────────────────────────────────────
            self._draw()
            self.clock.tick(int(FPS * self.speed_mult))

            # Record
            if self.record and self.video_writer:
                px = pygame.surfarray.array3d(self.screen)
                px = px.transpose([1, 0, 2])
                self.video_writer.write(self.cv2.cvtColor(px, self.cv2.COLOR_RGB2BGR))

        if self.video_writer:
            self.video_writer.release()
        pygame.quit()

    def _reset_episode(self):
        self.state      = self.env.reset()
        self.done       = False
        self.trail      = [(self.env.rover["x"], self.env.rover["y"])]
        self.ep_reward  = 0.0
        self.step_count = 0
        self.episode   += 1
        self.last_action= 0
        self.last_reward= 0.0

    def _draw(self):
        s = self.screen

        # Background
        s.fill(C_BG)

        # Grid
        for gx in range(0, WORLD_W, 50):
            pygame.draw.line(s, C_GRID, (gx, 0), (gx, WORLD_H))
        for gy in range(0, WORLD_H, 50):
            pygame.draw.line(s, C_GRID, (0, gy), (WORLD_W, gy))

        # Obstacles
        for (ox, oy, ow, oh) in self.env.obstacles:
            pygame.draw.rect(s, C_OBSTACLE,   (ox, oy, ow, oh))
            pygame.draw.rect(s, C_OBS_EDGE,   (ox, oy, ow, oh), 2)

        # Trail
        if self.show_trail and len(self.trail) > 2:
            pts = [(int(x), int(y)) for x, y in self.trail[-400:]]
            pygame.draw.lines(s, C_TRAIL, False, pts, 2)

        # Goal (pulsing)
        pulse_r = int(GOAL_R * (1.0 + 0.18 * math.sin(self.pulse_t * 0.12)))
        gx, gy = int(self.env.goal["x"]), int(self.env.goal["y"])
        pygame.draw.circle(s, C_GOAL_RING, (gx, gy), pulse_r + 5, 3)
        pygame.draw.circle(s, C_GOAL,      (gx, gy), pulse_r)
        self._text(s, "GOAL", gx, gy - pulse_r - 10, self.font_sm, C_GOAL)

        # Sensor rays
        if self.show_sensors:
            endpoints = self.env.get_sensor_endpoints()
            sensor_vals = self.state[:N_SENSORS]
            for i, ((px, py), sv) in enumerate(zip(endpoints, sensor_vals)):
                # Color by proximity
                if sv > 0.6:
                    sc = C_SENSOR_OK
                elif sv > 0.3:
                    sc = C_SENSOR_MID
                else:
                    sc = C_SENSOR_BAD
                pygame.draw.line(s, (*sc, 120), 
                                 (int(self.env.rover["x"]), int(self.env.rover["y"])),
                                 (int(px), int(py)), 1)
                pygame.draw.circle(s, sc, (int(px), int(py)), 3)

        # Rover (triangle)
        rx, ry = self.env.rover["x"], self.env.rover["y"]
        ang    = self.env.rover["angle"]
        pts = [
            (rx + math.cos(ang) * ROVER_R * 1.6,
             ry + math.sin(ang) * ROVER_R * 1.6),
            (rx + math.cos(ang + 2.5) * ROVER_R,
             ry + math.sin(ang + 2.5) * ROVER_R),
            (rx + math.cos(ang - 2.5) * ROVER_R,
             ry + math.sin(ang - 2.5) * ROVER_R),
        ]
        pts_int = [(int(x), int(y)) for x, y in pts]
        pygame.draw.polygon(s, C_ROVER,     pts_int)
        pygame.draw.polygon(s, C_ROVER_FWD, pts_int, 2)

        # Forward direction dot
        fwd_x = rx + math.cos(ang) * ROVER_R * 2.2
        fwd_y = ry + math.sin(ang) * ROVER_R * 2.2
        pygame.draw.circle(s, C_WHITE, (int(fwd_x), int(fwd_y)), 3)

        # ── HUD panel ─────────────────────────────────────────────────────────
        self._draw_hud(s)

    def _draw_hud(self, s):
        px = WORLD_W + 10
        py = 10
        lh = 18   # line height

        def line(text, color=C_TEXT, bold=False):
            nonlocal py
            f = self.font_big if bold else self.font_mid
            surf = f.render(text, True, color)
            s.blit(surf, (px, py))
            py += lh

        def gap(n=1):
            nonlocal py
            py += lh * n

        # Panel background
        pygame.draw.rect(s, C_PANEL, (WORLD_W, 0, HUD_W, WORLD_H))
        pygame.draw.line(s, C_OBS_EDGE, (WORLD_W, 0), (WORLD_W, WORLD_H), 2)

        line("── RL ROVER v2 ──", C_ROVER, bold=True)
        gap(0.3)
        line(f"Episode   : {self.episode}", C_TEXT_DIM)
        line(f"Step      : {self.step_count}", C_TEXT_DIM)

        goal_r = (self.total_goals / max(self.total_eps, 1)) * 100
        col_r  = goal_r  # placeholder
        gr_col = C_GOAL if goal_r > 50 else (C_SENSOR_MID if goal_r > 25 else C_SENSOR_BAD)
        line(f"Goal rate : {goal_r:.1f}%", gr_col)
        line(f"Goals/eps : {self.total_goals}/{self.total_eps}", C_TEXT_DIM)

        gap(0.5)
        line("── EPISODE ──", C_TEXT_DIM, bold=True)
        rew_col = C_REWARD_POS if self.ep_reward >= 0 else C_REWARD_NEG
        line(f"Reward    : {self.ep_reward:+.2f}", rew_col)

        # Reward bar
        bar_max = 20
        bar_val = max(-bar_max, min(bar_max, self.ep_reward))
        bar_w   = int(abs(bar_val) / bar_max * 120)
        bx = px
        by = py
        py += 14
        pygame.draw.rect(s, (40, 40, 40), (bx, by, 240, 10), border_radius=4)
        if bar_val >= 0:
            pygame.draw.rect(s, C_REWARD_POS, (bx + 120, by, bar_w, 10), border_radius=4)
        else:
            pygame.draw.rect(s, C_REWARD_NEG, (bx + 120 - bar_w, by, bar_w, 10), border_radius=4)
        pygame.draw.line(s, C_WHITE, (bx + 120, by), (bx + 120, by + 10))
        gap(0.3)

        # Last action
        aname = ACTION_NAMES[getattr(self, "last_action", 0)]
        line(f"Action    : {aname}", C_YELLOW)
        line(f"Last Δr   : {getattr(self, 'last_reward', 0):+.3f}", C_TEXT_DIM)

        gap(0.5)
        line("── Q-VALUES ──", C_TEXT_DIM, bold=True)
        if self.agent:
            q = self.agent.get_q_values(self.state)
            q_min, q_max_ = q.min(), q.max()
            q_range = max(q_max_ - q_min, 1e-3)
            best = int(np.argmax(q))
            from env import ACTION_NAMES as AN
            for i, (qv, an) in enumerate(zip(q, AN)):
                col = C_YELLOW if i == best else C_TEXT_DIM
                bar_w_q = int((qv - q_min) / q_range * 140)
                qy = py
                pygame.draw.rect(s, (40, 40, 40), (px, qy, 140, 12), border_radius=3)
                pygame.draw.rect(s, (*col[:3], 180), (px, qy, bar_w_q, 12), border_radius=3)
                label = f"{'>' if i==best else ' '}{an:<10} {qv:+.2f}"
                surf = self.font_sm.render(label, True, col)
                s.blit(surf, (px + 2, qy + 1))
                py += 16

        gap(0.5)
        line("── SENSORS ──", C_TEXT_DIM, bold=True)
        sv = self.state[:N_SENSORS]
        for i, v in enumerate(sv):
            col = C_SENSOR_OK if v > 0.6 else (C_SENSOR_MID if v > 0.3 else C_SENSOR_BAD)
            bar_sv = int(v * 130)
            sy = py
            pygame.draw.rect(s, (40, 40, 40), (px, sy, 130, 10), border_radius=3)
            pygame.draw.rect(s, col, (px, sy, bar_sv, 10), border_radius=3)
            surf = self.font_sm.render(f"S{i} {v:.2f}", True, col)
            s.blit(surf, (px + 135, sy))
            py += 14

        gap(0.5)
        line("── CONTROLS ──", C_TEXT_DIM, bold=True)
        for text in ["SPACE: pause", "R: reset", "S: sensors",
                     "T: trail", "+/-: speed", "Q: quit"]:
            line(text, C_TEXT_DIM)

        # Paused overlay
        if self.paused:
            overlay = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            s.blit(overlay, (0, 0))
            txt = self.font_big.render("  PAUSED  (SPACE to resume)  ", True, C_WHITE)
            r = txt.get_rect(center=(WORLD_W//2, WORLD_H//2))
            pygame.draw.rect(s, C_PANEL, r.inflate(20, 12), border_radius=8)
            s.blit(txt, r)

        pygame.display.flip()

    def _text(self, s, text, x, y, font, color):
        surf = font.render(text, True, color)
        r = surf.get_rect(center=(x, y))
        s.blit(surf, r)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      action="store_true",  help="save MP4 instead of live window")
    parser.add_argument("--random",     action="store_true",  help="use random agent (no model)")
    parser.add_argument("--episodes",   type=int, default=0,  help="stop after N episodes (0=infinite)")
    parser.add_argument("--difficulty", default="medium",     choices=["easy","medium","hard"])
    parser.add_argument("--seed",       type=int, default=SEED)
    args = parser.parse_args()

    # Load agent
    agent = None
    if not args.random:
        from env import STATE_DIM, N_ACTIONS
        agent = DDQNAgent(STATE_DIM, N_ACTIONS)
        try:
            agent.load(MODEL_PATH)
            agent.epsilon = 0.0  # pure greedy
        except Exception as e:
            print(f"  [!] Could not load model: {e}")
            print("  [!] Falling back to random agent.")
            agent = None

    n_ep = args.episodes if args.episodes > 0 else None

    if args.video or not PYGAME_OK:
        if not PYGAME_OK:
            print("  [!] Pygame not installed. Falling back to video recording.")
            print("      Install with: pip install pygame")
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "demo_trained_v2.mp4" if agent else "demo_random_v2.mp4")
        record_video_matplotlib(agent, n_ep or 5, out_path, args.seed, args.difficulty)
    else:
        vid_path = None
        if args.video:
            vid_path = os.path.join(OUT_DIR, "demo_v2.mp4")
        sim = PygameSimulator(agent, args.difficulty, args.seed,
                              record=args.video, record_path=vid_path)
        sim.run(n_episodes=n_ep)


if __name__ == "__main__":
    main()
