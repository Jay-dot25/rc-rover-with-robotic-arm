"""
agent.py — Double DQN with Dueling Architecture + Prioritized Replay (v2)
==========================================================================
Major upgrades over v1:
  • Dueling DQN:  Q(s,a) = V(s) + A(s,a) − mean(A)
                  Separate value & advantage streams → better learning
  • Double DQN:   Use online net for action selection, target net for Q value
                  → removes overestimation bias
  • Prioritized Experience Replay (PER):
                  Sample transitions by TD error magnitude → learn faster
  • Larger network: 12 → 256 → 256 → 128 → 5
  • Gradient clipping: prevents exploding gradients
"""

import numpy as np
import random
from collections import deque
import json, os

# ── Hyper-parameters ──────────────────────────────────────────────────────────
LEARNING_RATE  = 0.0005       # lower LR for stability
GAMMA          = 0.99         # higher discount
EPSILON_START  = 1.0
EPSILON_END    = 0.02
EPSILON_DECAY  = 0.9975       # slower decay → more exploration
BUFFER_SIZE    = 50_000       # much larger buffer
BATCH_SIZE     = 128          # bigger batches
TARGET_UPDATE  = 15           # sync target net
GRAD_CLIP      = 1.0          # gradient clipping
LEARN_EVERY    = 2            # train every N steps (not every step)

# PER parameters
PER_ALPHA      = 0.6          # priority exponent
PER_BETA_START = 0.4          # importance sampling start
PER_BETA_END   = 1.0
PER_EPS        = 1e-6         # min priority


# ── Activation helpers ────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)


# ── Adam optimiser ────────────────────────────────────────────────────────────

class AdamState:
    def __init__(self, shape, lr=LEARNING_RATE, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0

    def update(self, grad):
        # Clip gradient
        grad = np.clip(grad, -GRAD_CLIP, GRAD_CLIP)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat  = self.m / (1 - self.beta1 ** self.t)
        v_hat  = self.v / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Dueling Neural Network ────────────────────────────────────────────────────

class DuelingNet:
    """
    Dueling DQN architecture:
      Shared backbone → two heads:
        Value head:     scalar V(s)
        Advantage head: vector A(s, a) for each action
      Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

    Architecture:
      Input(12) → Dense(256,ReLU) → Dense(256,ReLU) → Dense(128,ReLU)
        ├── Value head:     Dense(64,ReLU) → Dense(1)
        └── Advantage head: Dense(64,ReLU) → Dense(n_actions)
    """

    def __init__(self, input_dim=12, n_actions=5, lr=LEARNING_RATE):
        self.input_dim = input_dim
        self.n_actions = n_actions

        def fan_in(n): return np.sqrt(2.0 / n)

        # Shared backbone
        self.W1 = np.random.randn(input_dim, 256).astype(np.float32) * fan_in(input_dim)
        self.b1 = np.zeros((1, 256), dtype=np.float32)
        self.W2 = np.random.randn(256, 256).astype(np.float32) * fan_in(256)
        self.b2 = np.zeros((1, 256), dtype=np.float32)
        self.W3 = np.random.randn(256, 128).astype(np.float32) * fan_in(256)
        self.b3 = np.zeros((1, 128), dtype=np.float32)

        # Value stream
        self.WV1 = np.random.randn(128, 64).astype(np.float32) * fan_in(128)
        self.bV1 = np.zeros((1, 64), dtype=np.float32)
        self.WV2 = np.random.randn(64, 1).astype(np.float32) * fan_in(64)
        self.bV2 = np.zeros((1, 1), dtype=np.float32)

        # Advantage stream
        self.WA1 = np.random.randn(128, 64).astype(np.float32) * fan_in(128)
        self.bA1 = np.zeros((1, 64), dtype=np.float32)
        self.WA2 = np.random.randn(64, n_actions).astype(np.float32) * fan_in(64)
        self.bA2 = np.zeros((1, n_actions), dtype=np.float32)

        # Adam states
        self.opt = {k: AdamState(getattr(self, k).shape, lr) for k in
                    ["W1","b1","W2","b2","W3","b3",
                     "WV1","bV1","WV2","bV2",
                     "WA1","bA1","WA2","bA2"]}

    def forward(self, x):
        """x: (B, input_dim) → Q: (B, n_actions)"""
        self._x   = x
        self._z1  = x @ self.W1 + self.b1;   self._a1 = relu(self._z1)
        self._z2  = self._a1 @ self.W2 + self.b2; self._a2 = relu(self._z2)
        self._z3  = self._a2 @ self.W3 + self.b3; self._a3 = relu(self._z3)

        # Value stream
        self._zV1 = self._a3 @ self.WV1 + self.bV1; self._aV1 = relu(self._zV1)
        V = self._aV1 @ self.WV2 + self.bV2          # (B, 1)

        # Advantage stream
        self._zA1 = self._a3 @ self.WA1 + self.bA1; self._aA1 = relu(self._zA1)
        A = self._aA1 @ self.WA2 + self.bA2          # (B, n_actions)

        # Combine: Q = V + A - mean(A)
        Q = V + A - A.mean(axis=1, keepdims=True)
        self._V, self._A, self._Q = V, A, Q
        return Q

    def backward(self, dQ, weights=None):
        """Backprop with optional importance-sampling weights."""
        B = self._x.shape[0]
        if weights is not None:
            dQ = dQ * weights.reshape(-1, 1)

        # dQ → dV, dA
        dV = dQ.sum(axis=1, keepdims=True)
        dA = dQ - dQ.mean(axis=1, keepdims=True)

        # Advantage head
        dWA2 = (self._aA1.T @ dA) / B
        dbA2 = dA.mean(axis=0, keepdims=True)
        daA1 = dA @ self.WA2.T
        dzA1 = daA1 * relu_grad(self._zA1)
        dWA1 = (self._a3.T @ dzA1) / B
        dbA1 = dzA1.mean(axis=0, keepdims=True)

        # Value head
        dWV2 = (self._aV1.T @ dV) / B
        dbV2 = dV.mean(axis=0, keepdims=True)
        daV1 = dV @ self.WV2.T
        dzV1 = daV1 * relu_grad(self._zV1)
        dWV1 = (self._a3.T @ dzV1) / B
        dbV1 = dzV1.mean(axis=0, keepdims=True)

        # Shared backbone
        da3  = (dzA1 @ self.WA1.T) + (dzV1 @ self.WV1.T)
        dz3  = da3 * relu_grad(self._z3)
        dW3  = (self._a2.T @ dz3) / B
        db3  = dz3.mean(axis=0, keepdims=True)
        da2  = dz3 @ self.W3.T
        dz2  = da2 * relu_grad(self._z2)
        dW2  = (self._a1.T @ dz2) / B
        db2  = dz2.mean(axis=0, keepdims=True)
        da1  = dz2 @ self.W2.T
        dz1  = da1 * relu_grad(self._z1)
        dW1  = (self._x.T @ dz1) / B
        db1  = dz1.mean(axis=0, keepdims=True)

        for name, grad in [("W1",dW1),("b1",db1),("W2",dW2),("b2",db2),
                            ("W3",dW3),("b3",db3),("WV1",dWV1),("bV1",dbV1),
                            ("WV2",dWV2),("bV2",dbV2),("WA1",dWA1),("bA1",dbA1),
                            ("WA2",dWA2),("bA2",dbA2)]:
            setattr(self, name, getattr(self, name) - self.opt[name].update(grad))

    def copy_weights_from(self, other):
        for k in ["W1","b1","W2","b2","W3","b3",
                  "WV1","bV1","WV2","bV2","WA1","bA1","WA2","bA2"]:
            setattr(self, k, getattr(other, k).copy())

    def predict(self, state_vec):
        x = state_vec.reshape(1, -1).astype(np.float32)
        return self.forward(x)[0]

    def save(self, path):
        arrays = {k: getattr(self, k) for k in
                  ["W1","b1","W2","b2","W3","b3",
                   "WV1","bV1","WV2","bV2","WA1","bA1","WA2","bA2"]}
        np.savez(path, **arrays)

    def load(self, path):
        f = path if path.endswith(".npz") else path + ".npz"
        d = np.load(f)
        for k in d:
            setattr(self, k, d[k])


# ── Prioritized Experience Replay ─────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Sum-tree based PER buffer.
    Transitions with higher TD-error are sampled more often.
    """

    def __init__(self, capacity=BUFFER_SIZE, alpha=PER_ALPHA):
        self.capacity = capacity
        self.alpha    = alpha
        self.buf      = []
        self.pos      = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        transition = (state.astype(np.float32), int(action),
                      float(reward), next_state.astype(np.float32), bool(done))
        if len(self.buf) < self.capacity:
            self.buf.append(transition)
        else:
            self.buf[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        n = len(self.buf)
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buf[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = (abs(err) + PER_EPS) ** self.alpha
            self.priorities[idx] = p
            if p > self.max_priority:
                self.max_priority = p

    def __len__(self):
        return len(self.buf)


# ── Double Dueling DQN Agent ──────────────────────────────────────────────────

class DDQNAgent:
    """
    Double DQN + Dueling Architecture + Prioritized Experience Replay.

    Double DQN trick:
      a* = argmax_a  Q_online(s', a)       ← online net picks action
      target = r + γ * Q_target(s', a*)   ← target net evaluates it
      Eliminates the overestimation bias of vanilla DQN.
    """

    def __init__(self, state_dim=12, n_actions=5, lr=LEARNING_RATE):
        self.n_actions = n_actions
        self.epsilon   = EPSILON_START
        self.episode   = 0
        self.step_count = 0
        self.beta      = PER_BETA_START

        self.online_net = DuelingNet(state_dim, n_actions, lr)
        self.target_net = DuelingNet(state_dim, n_actions, lr)
        self.target_net.copy_weights_from(self.online_net)

        self.buffer = PrioritizedReplayBuffer(BUFFER_SIZE)
        self.losses = []

    def select_action(self, state, greedy=False):
        eps = 0.0 if greedy else self.epsilon
        if np.random.random() < eps:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.online_net.predict(state)))

    def get_q_values(self, state):
        return self.online_net.predict(state)

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return None

        self.step_count += 1
        if self.step_count % LEARN_EVERY != 0:
            return None

        # Anneal beta
        self.beta = min(PER_BETA_END,
                        self.beta + (PER_BETA_END - PER_BETA_START) / 50000)

        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(BATCH_SIZE, self.beta)

        # Double DQN: online net selects action, target net evaluates
        q_online_next = self.online_net.forward(next_states)   # (B, A)
        best_actions  = np.argmax(q_online_next, axis=1)       # (B,)
        q_target_next = self.target_net.forward(next_states)   # (B, A)
        q_next_val    = q_target_next[np.arange(BATCH_SIZE), best_actions]  # (B,)

        targets = rewards + GAMMA * q_next_val * (1.0 - dones)  # (B,)

        # Current Q values
        q_all    = self.online_net.forward(states)
        q_chosen = q_all[np.arange(BATCH_SIZE), actions]

        td_errors = q_chosen - targets
        loss_val  = float(np.mean(weights * td_errors ** 2))

        # Update priorities
        self.buffer.update_priorities(indices, td_errors)

        # Backprop
        grad_out = np.zeros_like(q_all)
        grad_out[np.arange(BATCH_SIZE), actions] = 2.0 * td_errors / BATCH_SIZE
        self.online_net.backward(grad_out, weights)

        self.losses.append(loss_val)
        return loss_val

    def end_episode(self):
        self.episode += 1
        self.epsilon  = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        if self.episode % TARGET_UPDATE == 0:
            self.target_net.copy_weights_from(self.online_net)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.online_net.save(path)
        meta = {"episode": self.episode, "epsilon": float(self.epsilon),
                "beta": float(self.beta)}
        with open(path + "_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Agent] Saved → {path}")

    def load(self, path):
        self.online_net.load(path)
        self.target_net.copy_weights_from(self.online_net)
        meta_path = path + "_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.episode = meta.get("episode", 0)
            self.epsilon = meta.get("epsilon", EPSILON_END)
            self.beta    = meta.get("beta", PER_BETA_END)
        print(f"[Agent] Loaded ← {path}  (ep={self.episode}, ε={self.epsilon:.3f})")
