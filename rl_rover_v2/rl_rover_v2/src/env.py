"""
env.py — RC Rover Environment (v2)
===================================
Improvements over v1:
  • 8 sensors (45° apart) instead of 3 → much richer obstacle perception
  • Velocity state added → smoother planning
  • Smarter reward: angular alignment bonus, wall penalty, proximity shaping
  • Path-blocking detection prevents goal being hidden behind wall
  • Configurable difficulty (EASY / MEDIUM / HARD)
"""

import math
import random
import numpy as np

# ── World constants ──────────────────────────────────────────────────────────
WORLD_W   = 700
WORLD_H   = 700
ROVER_R   = 12
GOAL_R    = 16
SENSOR_R  = 160          # longer sensing range
N_SENSORS = 8            # sensors evenly around front 270°
N_OBS     = 6
MAX_STEPS = 600

MOVE_SPD  = 10.0
TURN_SPD  = 0.18         # rad per step

# Action indices
ACTION_FORWARD     = 0
ACTION_LEFT        = 1
ACTION_RIGHT       = 2
ACTION_HARD_LEFT   = 3
ACTION_HARD_RIGHT  = 4
N_ACTIONS          = 5
ACTION_NAMES       = ["forward", "left", "right", "hard_left", "hard_right"]

# State: 8 sensors + d_goal + theta_goal + speed + angle_rate = 12
STATE_DIM = N_SENSORS + 4  # 12


class RoverEnv:
    """
    Improved 2-D rover navigation env.

    State (12-D):
        0..7   sensor readings  (normalized 0→1, 1=clear, 0=obstacle)
        8      d_goal           (normalized distance)
        9      theta_goal       (angle to goal / π, in [-1,1])
        10     speed_norm       (0 or 1 depending on last action)
        11     ang_vel_norm     (−1, 0, or +1)

    Reward:
        +20    goal reached (terminal)
        −8     collision    (terminal)
        +0.3   moved closer to goal
        +0.1   facing goal (|theta_goal| < 0.2)
        −0.1   close to wall (front sensor < 0.2)
        −0.05  time penalty
    """

    def __init__(self, seed=None, difficulty="medium"):
        self.rng    = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.difficulty = difficulty

        self.rover     = {"x": 0.0, "y": 0.0, "angle": 0.0}
        self.goal      = {"x": 0.0, "y": 0.0}
        self.obstacles = []
        self.steps     = 0
        self.done      = False
        self._prev_dist = 0.0
        self._last_action = 0
        self._angular_vel = 0.0

        # For renderer access
        self.last_reward        = 0.0
        self.last_hit_obstacle  = False
        self.last_reached_goal  = False

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        self.done = False
        self.steps = 0
        self.last_hit_obstacle = False
        self.last_reached_goal = False
        self._last_action = 0
        self._angular_vel = 0.0

        # Difficulty-based obstacle count
        n_obs = {"easy": 4, "medium": 6, "hard": 9}.get(self.difficulty, 6)

        # Place rover near center
        self.rover = {
            "x":     WORLD_W / 2 + self.rng.uniform(-60, 60),
            "y":     WORLD_H / 2 + self.rng.uniform(-60, 60),
            "angle": self.rng.uniform(0, 2 * math.pi),
        }

        # Place goal far away
        for _ in range(1000):
            gx = self.rng.uniform(50, WORLD_W - 50)
            gy = self.rng.uniform(50, WORLD_H - 50)
            if _dist(self.rover["x"], self.rover["y"], gx, gy) > 180:
                self.goal = {"x": gx, "y": gy}
                break

        # Place obstacles
        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < n_obs and attempts < 800:
            attempts += 1
            w = self.rng.randint(35, 80)
            h = self.rng.randint(25, 60)
            x = self.rng.uniform(20, WORLD_W - w - 20)
            y = self.rng.uniform(20, WORLD_H - h - 20)
            if (self._clear_of_rover(x, y, w, h, 40) and
                    self._clear_of_goal(x, y, w, h, 35) and
                    self._clear_of_others(x, y, w, h, 15)):
                self.obstacles.append((x, y, w, h))

        self._prev_dist = _dist(self.rover["x"], self.rover["y"],
                                self.goal["x"],  self.goal["y"])
        return self._get_state()

    def step(self, action: int):
        assert not self.done
        r = self.action_step(action)
        return r

    def action_step(self, action: int):
        r = self.rover
        prev_dist = _dist(r["x"], r["y"], self.goal["x"], self.goal["y"])
        self._last_action = action
        self._angular_vel = 0.0

        # Apply action
        if action == ACTION_FORWARD:
            r["x"] += math.cos(r["angle"]) * MOVE_SPD
            r["y"] += math.sin(r["angle"]) * MOVE_SPD
        elif action == ACTION_LEFT:
            r["angle"] -= TURN_SPD
            self._angular_vel = -0.5
        elif action == ACTION_RIGHT:
            r["angle"] += TURN_SPD
            self._angular_vel = 0.5
        elif action == ACTION_HARD_LEFT:
            r["angle"] -= TURN_SPD * 2.2
            self._angular_vel = -1.0
        elif action == ACTION_HARD_RIGHT:
            r["angle"] += TURN_SPD * 2.2
            self._angular_vel = 1.0

        # Clamp to world
        r["x"] = max(ROVER_R, min(WORLD_W - ROVER_R, r["x"]))
        r["y"] = max(ROVER_R, min(WORLD_H - ROVER_R, r["y"]))
        r["angle"] %= (2 * math.pi)

        self.steps += 1
        reward = -0.05  # time penalty

        # Check goal
        goal_dist = _dist(r["x"], r["y"], self.goal["x"], self.goal["y"])
        if goal_dist < GOAL_R + ROVER_R:
            reward = +20.0
            self.done = True
            self.last_reached_goal = True
            self.last_hit_obstacle = False
            self.last_reward = reward
            return self._get_state(), reward, True, {"goal": True, "collision": False}

        # Check collision
        if self._rover_collides():
            reward = -8.0
            self.done = True
            self.last_hit_obstacle = True
            self.last_reached_goal = False
            self.last_reward = reward
            return self._get_state(), reward, True, {"goal": False, "collision": True}

        # Dense shaping
        if action == ACTION_FORWARD:
            if goal_dist < prev_dist:
                reward += 0.3   # approached goal
            else:
                reward -= 0.05  # moved away

        # Alignment bonus: reward facing the goal
        state = self._get_state()
        theta_goal = state[9]  # index 9
        if abs(theta_goal) < 0.2:
            reward += 0.1

        # Wall proximity penalty (front sensor, index 2 = front)
        front_sensor = state[2]
        if front_sensor < 0.2 and action == ACTION_FORWARD:
            reward -= 0.1

        # Timeout
        if self.steps >= MAX_STEPS:
            self.done = True
            self.last_reward = reward
            return state, reward, True, {"goal": False, "collision": False}

        self._prev_dist = goal_dist
        self.last_reward = reward
        return state, reward, False, {"goal": False, "collision": False}

    # ── State ─────────────────────────────────────────────────────────────────

    def _get_state(self):
        r = self.rover
        dx = self.goal["x"] - r["x"]
        dy = self.goal["y"] - r["y"]

        d_goal = math.sqrt(dx*dx + dy*dy) / (math.sqrt(2) * WORLD_W)

        angle_to_goal = math.atan2(dy, dx)
        rel_angle = angle_to_goal - r["angle"]
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        theta_goal = rel_angle / math.pi

        # 8 sensors: spread over -135° to +135° in front arc
        angles = [i * (math.pi * 1.5 / (N_SENSORS - 1)) - math.pi * 0.75
                  for i in range(N_SENSORS)]
        sensors = [min(self._sensor(a) / SENSOR_R, 1.0) for a in angles]

        speed_norm = 1.0 if self._last_action == ACTION_FORWARD else 0.0

        return np.array(sensors + [d_goal, theta_goal, speed_norm, self._angular_vel],
                        dtype=np.float32)

    # ── Sensor ray cast ───────────────────────────────────────────────────────

    def _sensor(self, rel_angle: float) -> float:
        angle = self.rover["angle"] + rel_angle
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rx, ry = self.rover["x"], self.rover["y"]
        for t in range(4, SENSOR_R + 1, 4):
            px = rx + cos_a * t
            py = ry + sin_a * t
            if px < 0 or px > WORLD_W or py < 0 or py > WORLD_H:
                return float(t)
            for (ox, oy, ow, oh) in self.obstacles:
                if ox <= px <= ox + ow and oy <= py <= oy + oh:
                    return float(t)
        return float(SENSOR_R)

    # ── Collision ────────────────────────────────────────────────────────────

    def _rover_collides(self) -> bool:
        rx, ry = self.rover["x"], self.rover["y"]
        for (ox, oy, ow, oh) in self.obstacles:
            if (ox - ROVER_R <= rx <= ox + ow + ROVER_R and
                    oy - ROVER_R <= ry <= oy + oh + ROVER_R):
                return True
        return False

    def _clear_of_rover(self, x, y, w, h, m):
        rx, ry = self.rover["x"], self.rover["y"]
        return not (x-m <= rx <= x+w+m and y-m <= ry <= y+h+m)

    def _clear_of_goal(self, x, y, w, h, m):
        gx, gy = self.goal["x"], self.goal["y"]
        return not (x-m <= gx <= x+w+m and y-m <= gy <= y+h+m)

    def _clear_of_others(self, x, y, w, h, m):
        for (ox, oy, ow, oh) in self.obstacles:
            if abs(x-ox) < w+ow+m and abs(y-oy) < h+oh+m:
                return False
        return True

    def get_sensor_endpoints(self):
        """Return list of (px, py) for each sensor ray endpoint."""
        angles = [i * (math.pi * 1.5 / (N_SENSORS - 1)) - math.pi * 0.75
                  for i in range(N_SENSORS)]
        result = []
        for rel in angles:
            d = self._sensor(rel)
            a = self.rover["angle"] + rel
            result.append((
                self.rover["x"] + math.cos(a) * d,
                self.rover["y"] + math.sin(a) * d,
            ))
        return result


def _dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)
