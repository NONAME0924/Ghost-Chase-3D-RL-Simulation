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

# Actions: 0-8: Movement, 9: Jump
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
    9: np.array([0.0, 0.0]),    # Jump (Stay horizontally)
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
HUNTER_FOV_ANGLE = math.radians(180) # Semi-circle
PREY_FOV_ANGLE = math.radians(120)   # Original
HUNTER_FOV_RANGE = 10.0              # Reduced from 15.0
PREY_FOV_RANGE = 6.0                 # Reduced from 8.0

# Physics for Jumping
GRAVITY = 0.03
JUMP_IMPULSE = 0.38 # Increased from 0.3, lower than original 0.5

# Power-up constants
POWERUP_PICKUP_DIST = 1.0      # Distance to pick up the item
POWERUP_BUFF_DURATION = 10.0   # seconds of 360° FOV
POWERUP_RESPAWN_COOLDOWN = 20.0  # seconds before next item spawns
STEPS_PER_SECOND = 10.0        # Assumed simulation steps per second for timer conversion


class ChaseEnv:
    """
    Game environment for Hunter vs Prey chase.

    State observation (per agent, 14 dimensions):
        0:  Distance to opponent (normalized, max if out of FOV)
        1:  Angle to opponent relative to facing direction (normalized)
        2:  Whether opponent is visible (0 or 1)
        3:  Own position normalized x
        4:  Own position normalized z
        5:  Own position normalized y
        6:  Distance to nearest boundary (normalized)
        7:  Facing direction cos
        8:  Facing direction sin
        9:  Distance to nearest point (normalized, 1.0 if not present)
       10:  Angle to nearest point relative to facing (normalized, 0 if not present)
       11:  Any point present (0 or 1)
       12:  Collection progress (0.0 to 1.0)
       13:  Opponent y (normalized, 0.0 if not visible)
    """

    def __init__(self):
        self.hunter_pos = np.zeros(2)
        self.hunter_y = 0.0
        self.hunter_vel_y = 0.0
        self.hunter_angle = 0.0

        self.prey_pos = np.zeros(2)
        self.prey_y = 0.0
        self.prey_vel_y = 0.0
        self.prey_angle = 0.0

        self.steps = 0
        self.max_steps = 3000

        # Point collection system (replace FOV power-up)
        self.points_pos = [np.zeros(2) for _ in range(3)]
        self.points_active = [False, False, False]
        self.points_collected = 0

        # L-shaped obstacles (Each L is two rectangles)
        self.obstacles = [
            # L-shape 1 (Central-ish)
            {'x': -4.0, 'z': -4.0, 'w': 3.0, 'd': 1.0},
            {'x': -5.0, 'z': -3.0, 'w': 1.0, 'd': 3.0},
            
            # L-shape 2 (Symmetrical-ish)
            {'x': 4.0, 'z': 4.0, 'w': 3.0, 'd': 1.0},
            {'x': 5.0, 'z': 3.0, 'w': 1.0, 'd': 3.0},

            # L-shape 3 (New)
            {'x': -4.0, 'z': 4.0, 'w': 3.0, 'd': 1.0},
            {'x': -3.0, 'z': 5.0, 'w': 1.0, 'd': 3.0},

            # L-shape 4 (New)
            {'x': 4.0, 'z': -4.0, 'w': 3.0, 'd': 1.0},
            {'x': 3.0, 'z': -5.0, 'w': 1.0, 'd': 3.0},

            # Some straight blocks for variety
            {'x': -7.0, 'z': 0.0, 'w': 3.0, 'd': 1.0},
            {'x': 7.0, 'z': 0.0, 'w': 3.0, 'd': 1.0},
            {'x': 0.0, 'z': -7.0, 'w': 1.0, 'd': 3.0},
            {'x': 0.0, 'z': 7.0, 'w': 1.0, 'd': 3.0},

            # Scattered small blocks (More cover)
            {'x': -1.5, 'z': -1.5, 'w': 1.2, 'd': 1.2},
            {'x': 1.5, 'z': 1.5, 'w': 1.2, 'd': 1.2},
            {'x': -6.0, 'z': 5.0, 'w': 2.0, 'd': 1.5},
            {'x': 6.0, 'z': -5.0, 'w': 1.5, 'd': 2.0},
        ]

        self.reset()

    def reset(self, episode=0):
        """Reset to random positions ensuring minimum distance and not inside obstacles."""
        spawn_attempts = 0
        while True:
            self.hunter_pos = np.random.uniform(-HALF_ARENA + 1, HALF_ARENA - 1, 2)
            self.prey_pos = np.random.uniform(-HALF_ARENA + 1, HALF_ARENA - 1, 2)
            
            # Check obstacles (at reset, they start on ground y=0)
            h_in_obs = any(self._is_inside_obstacle(self.hunter_pos, 0.0, obs) for obs in self.obstacles)
            p_in_obs = any(self._is_inside_obstacle(self.prey_pos, 0.0, obs) for obs in self.obstacles)
            
            if not h_in_obs and not p_in_obs and np.linalg.norm(self.hunter_pos - self.prey_pos) > 8.0:
                break
            
            # Safety yield to prevent locking the server if spawn is hard to find
            import time
            if self.steps == 0: # Only at start or when called externally
                pass 
            
            # Prevent infinite loop if constraints are impossible
            spawn_attempts += 1
            if spawn_attempts > 1000:
                print("  Warning: Spawn took too many attempts, relaxing constraints.")
                break

        self.hunter_angle = np.random.uniform(-math.pi, math.pi)
        self.hunter_y = 0.0
        self.hunter_vel_y = 0.0

        self.prey_angle = np.random.uniform(-math.pi, math.pi)
        self.prey_y = 0.0
        self.prey_vel_y = 0.0

        self.steps = 0
        self.points_collected = 0
        self.steps_since_last_point = 0

        # Spawn 3 points (clustered for early episodes)
        if episode > 0 and episode <= 10:
            cluster_center = self._spawn_point_pos()
            for i in range(3):
                # Put them within 2 units of each other
                offset = np.random.uniform(-1.5, 1.5, 2)
                self.points_pos[i] = np.clip(cluster_center + offset, -HALF_ARENA + 2, HALF_ARENA - 2)
                self.points_active[i] = True
        else:
            for i in range(3):
                self.points_pos[i] = self._spawn_point_pos()
                self.points_active[i] = True

        return self._get_obs_hunter(), self._get_obs_prey()

    def _is_inside_obstacle(self, pos, y_pos, obs, padding=0.5, agent_height=0.8):
        """Check if a position is inside an obstacle with optional padding, considering height."""
        obs_height = 1.2  # As defined in scene.js
        # If agent's bottom is above the obstacle top, no collision
        if (y_pos) > obs_height:
            return False

        half_w = obs['w'] / 2 + padding
        half_d = obs['d'] / 2 + padding
        return (obs['x'] - half_w <= pos[0] <= obs['x'] + half_w and
                obs['z'] - half_d <= pos[1] <= obs['z'] + half_d)

    def _spawn_point_pos(self):
        """Find a valid random position for a point."""
        for _ in range(100):
            pos = np.random.uniform(-HALF_ARENA + 2, HALF_ARENA - 2, 2)
            if any(self._is_inside_obstacle(pos, 0.0, obs, padding=0.3) for obs in self.obstacles):
                continue
            # Also stay away from other points if possible
            return pos
        return np.random.uniform(-HALF_ARENA + 2, HALF_ARENA - 2, 2)

    def _get_obs_hunter(self):
        """Get observation for the hunter (14 dimensions)."""
        visible, rel_angle, dist = self._is_in_fov(
            self.hunter_pos, self.hunter_angle, self.prey_pos,
            HUNTER_FOV_ANGLE, HUNTER_FOV_RANGE
        )

        max_val = math.sqrt(2) * ARENA_SIZE
        norm_dist = dist / max_val if visible else 1.0
        norm_angle = rel_angle / math.pi if visible else 0.0
        is_visible = 1.0 if visible else 0.0

        norm_x = self.hunter_pos[0] / HALF_ARENA
        norm_z = self.hunter_pos[1] / HALF_ARENA
        norm_y = self.hunter_y / 5.0 # Max expected jump height is roughly few units

        dist_to_boundary = min(
            HALF_ARENA - abs(self.hunter_pos[0]),
            HALF_ARENA - abs(self.hunter_pos[1])
        ) / HALF_ARENA

        face_cos = math.cos(self.hunter_angle)
        face_sin = math.sin(self.hunter_angle)

        # Observation of nearest active point
        p_dist, p_angle, p_any = self._get_nearest_point_obs(self.hunter_pos, self.hunter_angle)
        progress = self.points_collected / 3.0
        
        opp_y = self.prey_y / 5.0 if visible else 0.0

        return np.array([
            norm_dist, norm_angle, is_visible,
            norm_x, norm_z, norm_y,
            dist_to_boundary,
            face_cos, face_sin,
            p_dist, p_angle, p_any, progress,
            opp_y
        ], dtype=np.float32)

    def _get_obs_prey(self):
        """Get observation for the prey (14 dimensions)."""
        visible, rel_angle, dist = self._is_in_fov(
            self.prey_pos, self.prey_angle, self.hunter_pos,
            PREY_FOV_ANGLE, PREY_FOV_RANGE
        )

        max_val = math.sqrt(2) * ARENA_SIZE
        norm_dist = dist / max_val if visible else 1.0
        norm_angle = rel_angle / math.pi if visible else 0.0
        is_visible = 1.0 if visible else 0.0

        norm_x = self.prey_pos[0] / HALF_ARENA
        norm_z = self.prey_pos[1] / HALF_ARENA
        norm_y = self.prey_y / 5.0

        dist_to_boundary = min(
            HALF_ARENA - abs(self.prey_pos[0]),
            HALF_ARENA - abs(self.prey_pos[1])
        ) / HALF_ARENA

        face_cos = math.cos(self.prey_angle)
        face_sin = math.sin(self.prey_angle)

        # Observation of nearest active point
        p_dist, p_angle, p_any = self._get_nearest_point_obs(self.prey_pos, self.prey_angle)
        progress = self.points_collected / 3.0
        
        opp_y = self.hunter_y / 5.0 if visible else 0.0

        return np.array([
            norm_dist, norm_angle, is_visible,
            norm_x, norm_z, norm_y,
            dist_to_boundary,
            face_cos, face_sin,
            p_dist, p_angle, p_any, progress,
            opp_y
        ], dtype=np.float32)

    def _get_nearest_point_obs(self, agent_pos, agent_angle):
        """Find the nearest active point and return normalized distance and angle."""
        max_dist = math.sqrt(2) * ARENA_SIZE
        nearest_dist = float('inf')
        nearest_idx = -1

        for i in range(3):
            if self.points_active[i]:
                d = np.linalg.norm(self.points_pos[i] - agent_pos)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = i

        if nearest_idx == -1:
            return 1.0, 0.0, 0.0

        direction = self.points_pos[nearest_idx] - agent_pos
        angle_to_point = math.atan2(direction[1], direction[0])
        rel_angle = angle_to_point - agent_angle
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

        return nearest_dist / max_dist, rel_angle / math.pi, 1.0

    def _is_in_fov(self, observer_pos, observer_angle, target_pos, fov_angle, fov_range):
        """Check if target is within the FOV of the observer, considering walls and range."""
        direction = target_pos - observer_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return True, 0.0, dist
            
        if dist > fov_range:
            return False, 0.0, dist

        angle_to_target = math.atan2(direction[1], direction[0])
        rel_angle = angle_to_target - observer_angle
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

        is_in_angle = abs(rel_angle) <= fov_angle / 2.0
        
        if not is_in_angle:
            return False, rel_angle, dist
            
        # Line of sight check: is any wall blocking the path?
        if self._is_line_blocked(observer_pos, target_pos):
            return False, rel_angle, dist

        return True, rel_angle, dist

    def _is_line_blocked(self, p1, p2):
        """Check if the line segment from p1 to p2 intersects any obstacle."""
        for obs in self.obstacles:
            half_w = obs['w'] / 2.0
            half_d = obs['d'] / 2.0
            rect_min = np.array([obs['x'] - half_w, obs['z'] - half_d])
            rect_max = np.array([obs['x'] + half_w, obs['z'] + half_d])
            
            if self._line_intersects_rect(p1, p2, rect_min, rect_max):
                return True
        return False

    def _line_intersects_rect(self, p1, p2, r_min, r_max):
        """Liang-Barsky line segment intersection with AABB."""
        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]
        
        p = [-dx, dx, -dz, dz]
        q = [p1[0] - r_min[0], r_max[0] - p1[0], p1[1] - r_min[1], r_max[1] - p1[1]]
        
        u1 = 0.0
        u2 = 1.0
        
        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return False
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    if t > u2: return False
                    if t > u1: u1 = t
                else:
                    if t < u1: return False
                    if t < u2: u2 = t
                    
        return u1 <= u2

    def step(self, hunter_action, prey_action):
        """Execute one step with point collection mechanics."""
        self.steps += 1
        info = {'captured': False, 'points_win': False}

        # Calculate initial distances for "Progress" rewards
        diff_pos_old = self.hunter_pos - self.prey_pos
        diff_y_old = self.hunter_y - self.prey_y
        dist_3d_old = math.sqrt(np.sum(diff_pos_old**2) + diff_y_old**2)
        
        d_p_nearest_old, _, p_any = self._get_nearest_point_obs(self.prey_pos, self.prey_angle)

        # --- Move Hunter ---
        old_h_pos = self.hunter_pos.copy()
        move_dir_h = ACTIONS[hunter_action].copy()
        is_moving_h = np.linalg.norm(move_dir_h) > 0
        if is_moving_h:
            new_h_pos = self.hunter_pos + move_dir_h * MOVE_SPEED
            if not any(self._is_inside_obstacle(new_h_pos, self.hunter_y, obs) for obs in self.obstacles):
                self.hunter_angle = math.atan2(move_dir_h[1], move_dir_h[0])
                self.hunter_pos = new_h_pos

        # Hunter Jump Physics
        self.hunter_vel_y -= GRAVITY
        self.hunter_y += self.hunter_vel_y
        if self.hunter_y <= 0:
            self.hunter_y = 0
            self.hunter_vel_y = 0
        
        if hunter_action == 9 and self.hunter_y == 0:
            self.hunter_vel_y = JUMP_IMPULSE

        # --- Move Prey ---
        old_p_pos = self.prey_pos.copy()
        move_dir_p = ACTIONS[prey_action].copy()
        is_moving_p = np.linalg.norm(move_dir_p) > 0
        if is_moving_p:
            new_p_pos = self.prey_pos + move_dir_p * MOVE_SPEED
            if not any(self._is_inside_obstacle(new_p_pos, self.prey_y, obs) for obs in self.obstacles):
                self.prey_angle = math.atan2(move_dir_p[1], move_dir_p[0])
                self.prey_pos = new_p_pos

        # Prey Jump Physics
        self.prey_vel_y -= GRAVITY
        self.prey_y += self.prey_vel_y
        if self.prey_y <= 0:
            self.prey_y = 0
            self.prey_vel_y = 0
        
        if prey_action == 9 and self.prey_y == 0:
            self.prey_vel_y = JUMP_IMPULSE

        # --- Clamp to arena ---
        self.hunter_pos = np.clip(self.hunter_pos, -HALF_ARENA, HALF_ARENA)
        self.prey_pos = np.clip(self.prey_pos, -HALF_ARENA, HALF_ARENA)

        # --- Point collection ---
        point_collected_this_step = False
        self.steps_since_last_point += 1
        
        for i in range(3):
            if self.points_active[i]:
                dist_p_point = np.linalg.norm(self.prey_pos - self.points_pos[i])
                if dist_p_point < 1.0:
                    self.points_active[i] = False
                    self.points_collected += 1
                    self.steps_since_last_point = 0
                    point_collected_this_step = True

        # --- Win Conditions ---
        diff_pos = self.hunter_pos - self.prey_pos
        diff_y = self.hunter_y - self.prey_y
        dist_3d = math.sqrt(np.sum(diff_pos**2) + diff_y**2)
        
        captured = bool(dist_3d < CAPTURE_DIST)
        points_win = bool(self.points_collected >= 3)

        # --- Rewards ---
        h_vis, _, _ = self._is_in_fov(self.hunter_pos, self.hunter_angle, self.prey_pos, 
                                      HUNTER_FOV_ANGLE, HUNTER_FOV_RANGE)
        p_vis, _, _ = self._is_in_fov(self.prey_pos, self.prey_angle, self.hunter_pos,
                                      PREY_FOV_ANGLE, PREY_FOV_RANGE)

        # Hunter Reward: Much more positive and growth-oriented
        hunter_reward = 0.0 
        
        # 1. Base distance cost (scaled down to avoid heavy negatives)
        hunter_reward -= dist_3d * 0.01 
        
        # 2. Progress Reward: Reward for closing the gap
        if dist_3d < dist_3d_old:
            hunter_reward += (dist_3d_old - dist_3d) * 10.0
            
        # 3. Proximity Bonus: "Magnetic" pull when close
        if dist_3d < 5.0:
            hunter_reward += (5.0 - dist_3d) * 0.4
            
        # 4. Visibility Bonus: Staying on target is highly rewarded
        if h_vis: 
            hunter_reward += 2.0 
            
        # 5. Directional Bonus: Small reward for facing the prey
        _, rel_h, _ = self._is_in_fov(self.hunter_pos, self.hunter_angle, self.prey_pos, math.pi*2, 100.0)
        if abs(rel_h) < math.radians(20):
            hunter_reward += 0.5
        
        # Prey Reward: Balanced (Reverted from Overhauled)
        prey_reward = dist_3d * 0.005
        if not p_vis: prey_reward += 0.1
        else: prey_reward -= 0.1  
        
        # Point Gravity Reward: Encourage moving towards points
        if p_any > 0:
            d_p_nearest_new, _, _ = self._get_nearest_point_obs(self.prey_pos, self.prey_angle)
            if d_p_nearest_new < d_p_nearest_old:
                prey_reward += 0.1  
            elif d_p_nearest_new > d_p_nearest_old:
                prey_reward -= 0.05 
            
            # Visibility reward for points
            for i in range(3):
                if self.points_active[i]:
                    pt_vis, _, _ = self._is_in_fov(self.prey_pos, self.prey_angle, self.points_pos[i],
                                                  PREY_FOV_ANGLE, PREY_FOV_RANGE)
                    if pt_vis:
                        prey_reward += 0.005  
        
        if point_collected_this_step:
            # First point bonus (+100), others (+50)
            if self.points_collected == 1:
                prey_reward += 100.0
            else:
                prey_reward += 50.0
            # Extra prompt-requested point collection bonus
            prey_reward += 10.0

        # Softened inactivity penalty: Enough to discourage camping without being crushing
        if self.steps_since_last_point > 400:
            prey_reward -= 0.15

        # Corner Dead-Zone Penalty (2x2 units in each corner)
        # $|x| > 8$ and $|z| > 8$ for a 20x20 arena (HALF_ARENA = 10)
        for pos, role in [(self.hunter_pos, 'h'), (self.prey_pos, 'p')]:
            if abs(pos[0]) > 8.0 and abs(pos[1]) > 8.0:
                if role == 'h': hunter_reward -= 2.0
                else: prey_reward -= 5.0 # Heavy penalty for Prey

        # Collision Penalty: If agent tried to move but failed (hit wall/obstacle)
        if is_moving_h and np.linalg.norm(self.hunter_pos - old_h_pos) < 0.01:
            hunter_reward -= 1.0
        if is_moving_p and np.linalg.norm(self.prey_pos - old_p_pos) < 0.01:
            prey_reward -= 2.0

        if captured:
            hunter_reward += 150.0 
            prey_reward -= 50.0 # Reverted from -150
            info['captured'] = True
        elif points_win:
            hunter_reward -= 50.0
            prey_reward += 300.0 # Reverted from +500
            info['points_win'] = True

        # Boundary penalty (Increased weight to -2.0)
        for pos, role in [(self.hunter_pos, 'h'), (self.prey_pos, 'p')]:
            w_dist = min(HALF_ARENA - abs(pos[0]), HALF_ARENA - abs(pos[1]))
            if w_dist < 1.0:
                penalty = (1.0 - w_dist) * 2.0 # Increased from 0.5
                if role == 'h': hunter_reward -= penalty
                else: prey_reward -= penalty

        terminated = bool(captured or points_win)
        truncated = bool(self.steps >= self.max_steps)

        info.update({
            'steps': int(self.steps),
            'distance': float(dist_3d),
            'points_collected': self.points_collected
        })

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
            'hunter_y': float(self.hunter_y),
            'hunter_z': float(self.hunter_pos[1]),
            'hunter_angle': float(self.hunter_angle),
            'prey_x': float(self.prey_pos[0]),
            'prey_y': float(self.prey_y),
            'prey_z': float(self.prey_pos[1]),
            'prey_angle': float(self.prey_angle),
            'points_pos': [[float(p[0]), float(p[1])] for p in self.points_pos],
            'points_active': self.points_active,
            'points_collected': self.points_collected,
            'steps': int(self.steps),
            'obstacles': self.obstacles,
        }
