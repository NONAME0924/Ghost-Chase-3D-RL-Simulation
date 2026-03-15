"""
ChaseEnv - A Gym-like environment for the 3D Ghost Chase game.

Features:
- 20x20 bounded arena
- Two agents: Hunter (ghost) and Prey (human)
- 120° limited field of view for both agents (expandable via power-up)
- 9 discrete actions (8 directions + stay)
- Capture when distance < 1.0
- FOV power-up item: expands vision to 360° for 10 seconds
- Swap power-up item: swaps hunter and prey positions
"""

import numpy as np
import math

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
SWAP_RESPAWN_COOLDOWN = 30.0   # seconds before next swap item spawns
STEPS_PER_SECOND = 10.0        # Assumed simulation steps per second for timer conversion


class ChaseEnv:
    """
    Game environment for Hunter vs Prey chase.

    State observation (per agent, 15 dimensions):
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
       12:  Distance to swap item (normalized, 1.0 if not present)
       13:  Angle to swap item relative to facing (normalized, 0 if not present)
       14:  Swap item present on map (0 or 1)
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

        # Swap item state
        self.swap_pos = np.zeros(2)
        self.swap_active = False
        self.swap_cooldown = 0.0

        # Buff state (per agent)
        self.hunter_fov_buff = 0.0
        self.prey_fov_buff = 0.0

        # Role swap state
        self.roles_swapped = False

        self.reset()

    def reset(self):
        """Reset to random positions ensuring minimum distance."""
        while True:
            self.hunter_pos = np.random.uniform(-HALF_ARENA + 1, HALF_ARENA - 1, 2)
            self.prey_pos = np.random.uniform(-HALF_ARENA + 1, HALF_ARENA - 1, 2)
            if np.linalg.norm(self.hunter_pos - self.prey_pos) > 5.0:
                break

        self.hunter_angle = np.random.uniform(-math.pi, math.pi)
        self.prey_angle = np.random.uniform(-math.pi, math.pi)
        self.steps = 0

        # Spawn items
        self._spawn_powerup()
        self._spawn_swap()
        self.powerup_cooldown = 0.0
        self.swap_cooldown = 0.0
        self.hunter_fov_buff = 0.0
        self.prey_fov_buff = 0.0

        return self._get_obs_hunter(), self._get_obs_prey()

    def _spawn_powerup(self):
        """Spawn the FOV power-up at a random position away from both agents."""
        for _ in range(50):
            pos = np.random.uniform(-HALF_ARENA + 2, HALF_ARENA - 2, 2)
            dist_h = np.linalg.norm(pos - self.hunter_pos)
            dist_p = np.linalg.norm(pos - self.prey_pos)
            if dist_h > 3.0 and dist_p > 3.0:
                self.powerup_pos = pos
                self.powerup_active = True
                return
        self.powerup_pos = np.random.uniform(-HALF_ARENA + 2, HALF_ARENA - 2, 2)
        self.powerup_active = True

    def _spawn_swap(self):
        """Spawn the swap item at a random position away from both agents and other items."""
        for _ in range(50):
            pos = np.random.uniform(-HALF_ARENA + 2, HALF_ARENA - 2, 2)
            dist_h = np.linalg.norm(pos - self.hunter_pos)
            dist_p = np.linalg.norm(pos - self.prey_pos)
            dist_pu = np.linalg.norm(pos - self.powerup_pos) if self.powerup_active else 999
            if dist_h > 3.0 and dist_p > 3.0 and dist_pu > 3.0:
                self.swap_pos = pos
                self.swap_active = True
                return
        self.swap_pos = np.random.uniform(-HALF_ARENA + 2, HALF_ARENA - 2, 2)
        self.swap_active = True

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
        """Get observation for the hunter (15 dimensions)."""
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

        # Swap item observation
        sw_dist, sw_angle, sw_present = self._get_item_obs(
            self.hunter_pos, self.hunter_angle, self.swap_pos, self.swap_active
        )

        return np.array([
            norm_dist, norm_angle, is_visible,
            norm_x, norm_z, dist_to_boundary,
            face_cos, face_sin,
            pu_dist, pu_angle, pu_present, has_buff,
            sw_dist, sw_angle, sw_present
        ], dtype=np.float32)

    def _get_obs_prey(self):
        """Get observation for the prey (15 dimensions)."""
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

        # Swap item observation
        sw_dist, sw_angle, sw_present = self._get_item_obs(
            self.prey_pos, self.prey_angle, self.swap_pos, self.swap_active
        )

        return np.array([
            norm_dist, norm_angle, is_visible,
            norm_x, norm_z, dist_to_boundary,
            face_cos, face_sin,
            pu_dist, pu_angle, pu_present, has_buff,
            sw_dist, sw_angle, sw_present
        ], dtype=np.float32)

    def step(self, hunter_action, prey_action):
        """
        Execute one step.

        Returns:
            hunter_obs, prey_obs, hunter_reward, prey_reward, done, info
        """
        self.steps += 1

        # --- Move Hunter ---
        move_dir_h = ACTIONS[hunter_action].copy()
        if np.linalg.norm(move_dir_h) > 0:
            self.hunter_angle = math.atan2(move_dir_h[1], move_dir_h[0])
            self.hunter_pos += move_dir_h * MOVE_SPEED

        # --- Move Prey ---
        move_dir_p = ACTIONS[prey_action].copy()
        if np.linalg.norm(move_dir_p) > 0:
            self.prey_angle = math.atan2(move_dir_p[1], move_dir_p[0])
            self.prey_pos += move_dir_p * MOVE_SPEED

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

        # --- Swap item pickup (ROLE SWAP) ---
        swap_picked_by = None
        if self.swap_active:
            dist_h_sw = np.linalg.norm(self.hunter_pos - self.swap_pos)
            dist_p_sw = np.linalg.norm(self.prey_pos - self.swap_pos)

            if dist_h_sw < POWERUP_PICKUP_DIST or dist_p_sw < POWERUP_PICKUP_DIST:
                # Toggle role swap! Ghost becomes human, human becomes ghost
                self.roles_swapped = not self.roles_swapped

                if dist_h_sw < dist_p_sw:
                    swap_picked_by = 'hunter'
                else:
                    swap_picked_by = 'prey'

                self.swap_active = False
                self.swap_cooldown = SWAP_RESPAWN_COOLDOWN * STEPS_PER_SECOND

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

        if not self.swap_active and self.swap_cooldown > 0:
            self.swap_cooldown -= 1
            if self.swap_cooldown <= 0:
                self._spawn_swap()

        # --- Determine current roles ---
        # When roles_swapped: hunter_pos is the RUNNER, prey_pos is the CHASER
        # When normal: hunter_pos is the CHASER, prey_pos is the RUNNER
        dist = np.linalg.norm(self.hunter_pos - self.prey_pos)

        if self.roles_swapped:
            # Roles are flipped: prey_pos chases hunter_pos
            # "Capture" = prey catches hunter
            captured = bool(dist < CAPTURE_DIST)
        else:
            # Normal: hunter_pos chases prey_pos
            captured = bool(dist < CAPTURE_DIST)

        # --- Rewards ---
        h_fov = self._get_effective_fov('hunter')
        p_fov = self._get_effective_fov('prey')
        h_vis, _, _ = self._is_in_fov(self.hunter_pos, self.hunter_angle, self.prey_pos, fov_override=h_fov)
        p_vis, _, _ = self._is_in_fov(self.prey_pos, self.prey_angle, self.hunter_pos, fov_override=p_fov)

        if not self.roles_swapped:
            # Normal roles: hunter chases, prey flees
            if h_vis:
                hunter_reward = -dist * 0.01
            else:
                hunter_reward = -0.05

            if p_vis:
                prey_reward = dist * 0.01
            else:
                prey_reward = -0.02

            if captured:
                hunter_reward += 10.0
                prey_reward -= 10.0
        else:
            # SWAPPED roles: hunter must flee, prey must chase
            if h_vis:
                hunter_reward = dist * 0.01   # Hunter now wants DISTANCE
            else:
                hunter_reward = -0.02

            if p_vis:
                prey_reward = -dist * 0.01    # Prey now wants CLOSENESS
            else:
                prey_reward = -0.05

            if captured:
                hunter_reward -= 10.0  # Hunter (now runner) gets caught = bad
                prey_reward += 10.0    # Prey (now chaser) catches = good

        # FOV Power-up pickup reward
        if powerup_picked_by == 'hunter':
            hunter_reward += 2.0
        elif powerup_picked_by == 'prey':
            prey_reward += 2.0

        # Swap item pickup reward (asymmetric)
        # Hunter loses advantage → penalty; Prey gains advantage → reward
        if swap_picked_by is not None:
            if swap_picked_by == 'hunter':
                hunter_reward -= 2.0   # Hunter loses chaser role
            else:
                prey_reward += 3.0     # Prey gains chaser role

        # Observation reward
        if h_vis:
            hunter_reward += 0.5
        if p_vis:
            prey_reward += 0.5

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
        done = bool(captured or self.steps >= self.max_steps)

        info = {
            'captured': captured,
            'steps': int(self.steps),
            'distance': float(dist),
            'powerup_picked_by': powerup_picked_by,
            'swap_picked_by': swap_picked_by,
            'roles_swapped': self.roles_swapped,
        }

        return (
            self._get_obs_hunter(),
            self._get_obs_prey(),
            float(hunter_reward),
            float(prey_reward),
            done,
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
            # Swap item state
            'swap_x': float(self.swap_pos[0]),
            'swap_z': float(self.swap_pos[1]),
            'swap_active': self.swap_active,
            # Role swap state
            'roles_swapped': self.roles_swapped,
        }

