"""
ChaseEnv - A Gym-like environment for the 3D Ghost Chase game.

Features:
- 20x20 bounded arena
- Two agents: Hunter (ghost) and Prey (human)
- 120° limited field of view for both agents (expandable via power-up)
- 9 discrete actions (8 directions + stay)
- Capture when distance < 1.0
- FOV power-up item: expands vision to 360° for 10 seconds
"""

import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

# Action definitions: 8 directions + stay
ACTIONS = {
    0: np.array([0.0, 0.0]),    # Stay
    1: np.array([0.0, 1.0]),    # North
    2: np.array([1.0, 1.0]),    # NE
    3: np.array([1.0, 0.0]),    # East
    4: np.array([1.0, -1.0]),   # SE
    5: np.array([0.0, -1.0]),   # South
    6: np.array([-1.0, -1.0]),  # SW
    7: np.array([-1.0, 0.0]),   # West
    8: np.array([-1.0, 1.0]),   # NW
}

# Normalize diagonal movements
for k, v in ACTIONS.items():
    norm = np.linalg.norm(v)
    if norm > 0:
        ACTIONS[k] = v / norm

NUM_ACTIONS = len(ACTIONS)
ARENA_SIZE = 20.0
HALF_ARENA = ARENA_SIZE / 2.0
CAPTURE_DIST = 1.0
MOVE_SPEED = 0.3
FOV_ANGLE = math.radians(120)  # 120 degrees field of view

# Power-up constants
POWERUP_PICKUP_DIST = 1.0      # Distance to pick up the item
POWERUP_BUFF_DURATION = 10.0   # seconds of 360° FOV
POWERUP_RESPAWN_COOLDOWN = 20.0  # seconds before next item spawns
STEPS_PER_SECOND = 10.0        # Assumed simulation steps per second for timer conversion


class ChaseEnv:
    """
    Game environment for Hunter vs Prey chase.

    State observation (per agent, 12 dimensions):
        0:  Distance to opponent (normalized, max if out of FOV)
        1:  Angle to opponent relative to facing direction (normalized)
        2:  Whether opponent is visible (0 or 1)
        3:  Own position normalized x
        4:  Own position normalized z
        5:  Distance to nearest boundary (normalized)
        6:  Facing direction cos
        7:  Facing direction sin
        8:  Distance to FOV power-up (normalized, 1.0 if not present)
        9:  Angle to FOV power-up relative to facing (normalized, 0 if not present)
       10:  FOV power-up present on map (0 or 1)
       11:  Own FOV buff active (0 or 1)
    """

    def __init__(self):
        self.hunter_pos = np.zeros(2)
        self.hunter_angle = 0.0
        self.prey_pos = np.zeros(2)
        self.prey_angle = 0.0
        self.steps = 0
        self.max_steps = 1000

        # FOV power-up state
        self.powerup_pos = np.zeros(2)
        self.powerup_active = False
        self.powerup_cooldown = 0.0

        # Buff state (per agent)
        self.hunter_fov_buff = 0.0
        self.prey_fov_buff = 0.0

        # Obstacles: [x, z, width, depth]
        self.obstacles = [
            {'x': -5.0, 'z': -5.0, 'w': 2.0, 'd': 2.0},
            {'x': 5.0, 'z': 5.0, 'w': 3.0, 'd': 3.0},
            {'x': -6.0, 'z': 4.0, 'w': 1.5, 'd': 4.0},
            {'x': 4.0, 'z': -6.0, 'w': 4.0, 'd': 1.5},
        ]

        self.reset()

    def reset(self):
        """Reset to random positions ensuring minimum distance and not inside obstacles."""
        while True:
            self.hunter_pos = np.random.uniform(-HALF_ARENA + 1, HALF_ARENA - 1, 2)
            self.prey_pos = np.random.uniform(-HALF_ARENA + 1, HALF_ARENA - 1, 2)
            
            # Check obstacles for hunter
            h_in_obs = any(self._is_inside_obstacle(self.hunter_pos, obs) for obs in self.obstacles)
            # Check obstacles for prey
            p_in_obs = any(self._is_inside_obstacle(self.prey_pos, obs) for obs in self.obstacles)
            
            if not h_in_obs and not p_in_obs and np.linalg.norm(self.hunter_pos - self.prey_pos) > 5.0:
                break

        self.hunter_angle = np.random.uniform(-math.pi, math.pi)
        self.prey_angle = np.random.uniform(-math.pi, math.pi)
        self.steps = 0

        # Spawn items
        self._spawn_powerup()
        self.powerup_cooldown = 0.0
        self.hunter_fov_buff = 0.0
        self.prey_fov_buff = 0.0

        return self._get_obs_hunter(), self._get_obs_prey()

    def _is_inside_obstacle(self, pos, obs, padding=0.4):
        """Check if a position is inside an obstacle with optional padding."""
        half_w = obs['w'] / 2 + padding
        half_d = obs['d'] / 2 + padding
        return (obs['x'] - half_w <= pos[0] <= obs['x'] + half_w and
                obs['z'] - half_d <= pos[1] <= obs['z'] + half_d)

    def _spawn_powerup(self):
        """Spawn the FOV power-up at a random position away from both agents and obstacles."""
        for _ in range(100):
            pos = np.random.uniform(-HALF_ARENA + 2, HALF_ARENA - 2, 2)
            
            # Check obstacles
            if any(self._is_inside_obstacle(pos, obs, padding=0.2) for obs in self.obstacles):
                continue
                
            dist_h = np.linalg.norm(pos - self.hunter_pos)
            dist_p = np.linalg.norm(pos - self.prey_pos)
            if dist_h > 3.0 and dist_p > 3.0:
                self.powerup_pos = pos
                self.powerup_active = True
                return
        self.powerup_pos = np.random.uniform(-HALF_ARENA + 2, HALF_ARENA - 2, 2)
        self.powerup_active = True

    def _get_effective_fov(self, agent):
        """Get the effective FOV angle based on buff status."""
        if agent == 'hunter' and self.hunter_fov_buff > 0:
            return math.radians(360)
        elif agent == 'prey' and self.prey_fov_buff > 0:
            return math.radians(360)
        return FOV_ANGLE

    def _is_in_fov(self, observer_pos, observer_angle, target_pos, fov_override=None):
        """Check if target is within the FOV of the observer."""
        direction = target_pos - observer_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return True, 0.0, dist

        angle_to_target = math.atan2(direction[1], direction[0])
        rel_angle = angle_to_target - observer_angle
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

        fov = fov_override if fov_override is not None else FOV_ANGLE
        is_visible = abs(rel_angle) <= fov / 2.0
        return is_visible, rel_angle, dist

    def _get_item_obs(self, agent_pos, agent_angle, item_pos, item_active):
        """Get observation components related to an item."""
        max_dist = math.sqrt(2) * ARENA_SIZE

        if not item_active:
            return 1.0, 0.0, 0.0  # norm_dist, norm_angle, present

        direction = item_pos - agent_pos
        dist = np.linalg.norm(direction)
        norm_dist = dist / max_dist

        angle_to_item = math.atan2(direction[1], direction[0])
        rel_angle = angle_to_item - agent_angle
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        norm_angle = rel_angle / math.pi

        return norm_dist, norm_angle, 1.0

    def _get_obs_hunter(self):
        """Get observation for the hunter (12 dimensions)."""
        fov = self._get_effective_fov('hunter')
        visible, rel_angle, dist = self._is_in_fov(
            self.hunter_pos, self.hunter_angle, self.prey_pos, fov_override=fov
        )

        max_dist = math.sqrt(2) * ARENA_SIZE
        norm_dist = dist / max_dist if visible else 1.0
        norm_angle = rel_angle / math.pi if visible else 0.0
        is_visible = 1.0 if visible else 0.0

        norm_x = self.hunter_pos[0] / HALF_ARENA
        norm_z = self.hunter_pos[1] / HALF_ARENA

        dist_to_boundary = min(
            HALF_ARENA - abs(self.hunter_pos[0]),
            HALF_ARENA - abs(self.hunter_pos[1])
        ) / HALF_ARENA

        face_cos = math.cos(self.hunter_angle)
        face_sin = math.sin(self.hunter_angle)

        # FOV power-up observation
        pu_dist, pu_angle, pu_present = self._get_item_obs(
            self.hunter_pos, self.hunter_angle, self.powerup_pos, self.powerup_active
        )
        has_buff = 1.0 if self.hunter_fov_buff > 0 else 0.0

        return np.array([
            norm_dist, norm_angle, is_visible,
            norm_x, norm_z, dist_to_boundary,
            face_cos, face_sin,
            pu_dist, pu_angle, pu_present, has_buff,
        ], dtype=np.float32)

    def _get_obs_prey(self):
        """Get observation for the prey (12 dimensions)."""
        fov = self._get_effective_fov('prey')
        visible, rel_angle, dist = self._is_in_fov(
            self.prey_pos, self.prey_angle, self.hunter_pos, fov_override=fov
        )

        max_dist = math.sqrt(2) * ARENA_SIZE
        norm_dist = dist / max_dist if visible else 1.0
        norm_angle = rel_angle / math.pi if visible else 0.0
        is_visible = 1.0 if visible else 0.0

        norm_x = self.prey_pos[0] / HALF_ARENA
        norm_z = self.prey_pos[1] / HALF_ARENA

        dist_to_boundary = min(
            HALF_ARENA - abs(self.prey_pos[0]),
            HALF_ARENA - abs(self.prey_pos[1])
        ) / HALF_ARENA

        face_cos = math.cos(self.prey_angle)
        face_sin = math.sin(self.prey_angle)

        # FOV power-up observation
        pu_dist, pu_angle, pu_present = self._get_item_obs(
            self.prey_pos, self.prey_angle, self.powerup_pos, self.powerup_active
        )
        has_buff = 1.0 if self.prey_fov_buff > 0 else 0.0

        return np.array([
            norm_dist, norm_angle, is_visible,
            norm_x, norm_z, dist_to_boundary,
            face_cos, face_sin,
            pu_dist, pu_angle, pu_present, has_buff,
        ], dtype=np.float32)

    def step(self, hunter_action, prey_action):
        """
        Execute one step.

        Returns:
            hunter_obs, prey_obs, hunter_reward, prey_reward, terminated, truncated, info
        """
        self.steps += 1

        # --- Move Hunter ---
        move_dir_h = ACTIONS[hunter_action].copy()
        if np.linalg.norm(move_dir_h) > 0:
            new_h_pos = self.hunter_pos + move_dir_h * MOVE_SPEED
            # Collision check
            if not any(self._is_inside_obstacle(new_h_pos, obs) for obs in self.obstacles):
                self.hunter_angle = math.atan2(move_dir_h[1], move_dir_h[0])
                self.hunter_pos = new_h_pos

        # --- Move Prey ---
        move_dir_p = ACTIONS[prey_action].copy()
        if np.linalg.norm(move_dir_p) > 0:
            new_p_pos = self.prey_pos + move_dir_p * MOVE_SPEED
            # Collision check
            if not any(self._is_inside_obstacle(new_p_pos, obs) for obs in self.obstacles):
                self.prey_angle = math.atan2(move_dir_p[1], move_dir_p[0])
                self.prey_pos = new_p_pos

        # --- Clamp to arena ---
        self.hunter_pos = np.clip(self.hunter_pos, -HALF_ARENA, HALF_ARENA)
        self.prey_pos = np.clip(self.prey_pos, -HALF_ARENA, HALF_ARENA)

        # --- FOV Power-up pickup ---
        powerup_picked_by = None
        if self.powerup_active:
            dist_h_pu = np.linalg.norm(self.hunter_pos - self.powerup_pos)
            dist_p_pu = np.linalg.norm(self.prey_pos - self.powerup_pos)

            if dist_h_pu < POWERUP_PICKUP_DIST:
                self.hunter_fov_buff = POWERUP_BUFF_DURATION * STEPS_PER_SECOND
                self.powerup_active = False
                self.powerup_cooldown = POWERUP_RESPAWN_COOLDOWN * STEPS_PER_SECOND
                powerup_picked_by = 'hunter'
            elif dist_p_pu < POWERUP_PICKUP_DIST:
                self.prey_fov_buff = POWERUP_BUFF_DURATION * STEPS_PER_SECOND
                self.powerup_active = False
                self.powerup_cooldown = POWERUP_RESPAWN_COOLDOWN * STEPS_PER_SECOND
                powerup_picked_by = 'prey'

        # --- Buff countdown ---
        if self.hunter_fov_buff > 0:
            self.hunter_fov_buff -= 1
        if self.prey_fov_buff > 0:
            self.prey_fov_buff -= 1

        # --- Item respawn ---
        if not self.powerup_active and self.powerup_cooldown > 0:
            self.powerup_cooldown -= 1
            if self.powerup_cooldown <= 0:
                self._spawn_powerup()

        # --- Determine current roles ---
        dist = np.linalg.norm(self.hunter_pos - self.prey_pos)
        captured = bool(dist < CAPTURE_DIST)

        # --- Rewards ---
        h_fov = self._get_effective_fov('hunter')
        p_fov = self._get_effective_fov('prey')
        h_vis, _, _ = self._is_in_fov(self.hunter_pos, self.hunter_angle, self.prey_pos, fov_override=h_fov)
        p_vis, _, _ = self._is_in_fov(self.prey_pos, self.prey_angle, self.hunter_pos, fov_override=p_fov)

        # Normal roles: hunter chases, prey flees
        if h_vis:
            hunter_reward = -dist * 0.01
        else:
            hunter_reward = -0.05

        # Prey: Penalized for seeing the ghost (exposed), rewarded for hiding (safe)
        if p_vis:
            prey_reward = -0.5  # Penality for seeing the danger
        else:
            prey_reward = 0.2   # Reward for staying out of sight (hiding)

        # Still keep distance reward for Prey to encourage running far
        prey_reward += dist * 0.01

        if captured:
            hunter_reward += 10.0
            prey_reward -= 10.0

        # FOV Power-up pickup reward
        if powerup_picked_by == 'hunter':
            hunter_reward += 2.0
        elif powerup_picked_by == 'prey':
            prey_reward += 2.0

        # Observation reward (Prey reward moved to hiding logic above)
        if h_vis:
            hunter_reward += 0.5

        # Boundary penalty
        for pos, prefix in [(self.hunter_pos, 'hunter'), (self.prey_pos, 'prey')]:
            wall_dist = min(
                HALF_ARENA - abs(pos[0]),
                HALF_ARENA - abs(pos[1])
            )
            if wall_dist < 1.0:
                penalty = (1.0 - wall_dist) * 0.5
                if prefix == 'hunter':
                    hunter_reward -= penalty
                else:
                    prey_reward -= penalty

        # Timeout
        terminated = bool(captured)
        truncated = bool(self.steps >= self.max_steps)

        info = {
            'captured': captured,
            'steps': int(self.steps),
            'distance': float(dist),
            'powerup_picked_by': powerup_picked_by,
        }

        return (
            self._get_obs_hunter(),
            self._get_obs_prey(),
            float(hunter_reward),
            float(prey_reward),
            terminated,
            truncated,
            info
        )

    def get_state(self):
        """Get full state for visualization."""
        return {
            'hunter_x': float(self.hunter_pos[0]),
            'hunter_z': float(self.hunter_pos[1]),
            'hunter_angle': float(self.hunter_angle),
            'prey_x': float(self.prey_pos[0]),
            'prey_z': float(self.prey_pos[1]),
            'prey_angle': float(self.prey_angle),
            # FOV Power-up state
            'powerup_x': float(self.powerup_pos[0]),
            'powerup_z': float(self.powerup_pos[1]),
            'powerup_active': self.powerup_active,
            'hunter_fov_buff': self.hunter_fov_buff > 0,
            'prey_fov_buff': self.prey_fov_buff > 0,
            'obstacles': self.obstacles,
        }


class HunterEnv(gym.Env):
    """Gym-compatible wrapper for the Hunter agent."""
    def __init__(self, base_env, prey_policy_fn=None):
        super().__init__()
        self.base_env = base_env
        self.prey_policy_fn = prey_policy_fn
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_h, _ = self.base_env.reset()
        return obs_h, {}

    def step(self, action):
        # We need an action for the prey. If no policy provided, stay still.
        prey_obs = self.base_env._get_obs_prey()
        prey_action = self.prey_policy_fn(prey_obs) if self.prey_policy_fn else 0
        
        obs_h, _, reward_h, _, terminated, truncated, info = self.base_env.step(action, prey_action)
        return obs_h, reward_h, terminated, truncated, info


class PreyEnv(gym.Env):
    """Gym-compatible wrapper for the Prey agent."""
    def __init__(self, base_env, hunter_policy_fn=None):
        super().__init__()
        self.base_env = base_env
        self.hunter_policy_fn = hunter_policy_fn
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        _, obs_p = self.base_env.reset()
        return obs_p, {}

    def step(self, action):
        # We need an action for the hunter.
        hunter_obs = self.base_env._get_obs_hunter()
        hunter_action = self.hunter_policy_fn(hunter_obs) if self.hunter_policy_fn else 0
        
        _, obs_p, _, reward_p, terminated, truncated, info = self.base_env.step(hunter_action, action)
        return obs_p, reward_p, terminated, truncated, info
