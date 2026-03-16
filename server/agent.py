"""
SB3Agent - A robust wrapper for Stable Baselines3 DQN.
Supports manual training loops while keeping SB3's internal state consistent.
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

class SB3Agent:
    """
    Agent wrapper for Stable Baselines3 DQN.
    Matches the original DQNAgent interface for compatibility with app.py.
    """

    def __init__(
        self,
        state_dim=12,
        action_dim=9,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_capacity=50000,
        device="auto"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create a dummy environment for SB3 initialization
        class DummyEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(state_dim,), dtype=np.float32)
                self.action_space = gym.spaces.Discrete(action_dim)
            def reset(self, seed=None): return np.zeros(state_dim, dtype=np.float32), {}
            def step(self, action): return np.zeros(state_dim, dtype=np.float32), 0, False, False, {}

        self.dummy_env = DummyEnv()
        
        # Initialize the SB3 DQN model
        # We set exploration parameters here, but we'll also manage them manually for the log
        self.model = DQN(
            "MlpPolicy",
            self.dummy_env,
            learning_rate=lr,
            buffer_size=buffer_capacity,
            learning_starts=batch_size,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=100,
            exploration_initial_eps=epsilon_start,
            exploration_final_eps=epsilon_end,
            exploration_fraction=0.5,
            policy_kwargs=dict(net_arch=[128, 128]),
            verbose=0,
            device=device
        )
        
        # CRITICAL: Initialize internal SB3 attributes for manual .train() calls
        self.model.set_logger(configure(None, ["stdout"]))
        self.model._current_progress_remaining = 1.0 # 1.0 to 0.0
        # This is needed for the learning rate schedule
        
        # Manual epsilon for compatibility with app.py log
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0

    def select_action(self, state, training=True):
        """Select an action using the model."""
        # Ensure state is the right shape for SB3 (usually (12,) or (1, 12))
        obs = np.array(state, dtype=np.float32)
        if training:
            action, _ = self.model.predict(obs, deterministic=False)
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        
        # Sync the compatibility epsilon with SB3's internal one
        self.epsilon = self.model.exploration_rate
        return int(action)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in SB3 replay buffer."""
        # SB3 buffer expects [batch_size, obs_shape]
        obs = np.array(state, dtype=np.float32)
        next_obs = np.array(next_state, dtype=np.float32)
        
        self.model.replay_buffer.add(
            obs, 
            next_obs, 
            np.array([action]), 
            np.array([reward]), 
            np.array([done]), 
            [{}]
        )
        
        # Increment internal timestep counters
        self.total_steps += 1
        self.model.num_timesteps += 1
        
        # Update exploration rate schedule
        # SB3 uses progress_remaining (1.0 to 0.0) for its schedules
        self.model.exploration_rate = self.model.exploration_schedule(self.model._current_progress_remaining)

    def learn(self):
        """Trigger one training step."""
        if self.model.replay_buffer.size() >= self.model.batch_size:
            # We must ensure progress_remaining is set so LR schedule works
            # Here we just keep it at 1.0 or decay it slowly based on a hypothetical max steps
            # For now, 1.0 is safe to prevent crashes.
            self.model._current_progress_remaining = max(0.01, 1.0 - (self.total_steps / 1000000))
            
            self.model.train(gradient_steps=1, batch_size=self.model.batch_size)
            return 0.0 # Loss is not easily returned by SB3 .train()
        return 0.0

    def decay_epsilon(self):
        """
        In SB3 DQN, epsilon decay is usually handled per-step.
        We provide this for compatibility with the original app.py loop.
        """
        # If we want to override SB3's internal linear decay with our exponential one:
        # self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # self.model.exploration_rate = self.epsilon
        pass

    def save(self, filepath):
        """Save model."""
        # SB3 adds .zip automatically if not present in some methods, 
        # but .save() usually takes the exact path.
        self.model.save(filepath)

    def load(self, filepath):
        """Load model."""
        # Try with and without .zip
        path = filepath if filepath.endswith(".zip") else filepath + ".zip"
        if not os.path.exists(path):
            if not os.path.exists(filepath):
                return False
            path = filepath
        
        # Load the model and ensure the environment/logger are re-linked
        self.model = DQN.load(path, env=self.dummy_env, device=self.model.device)
        self.model.set_logger(configure(None, ["stdout"]))
        self.model._current_progress_remaining = 1.0
        return True

# Alias for backward compatibility
DQNAgent = SB3Agent
