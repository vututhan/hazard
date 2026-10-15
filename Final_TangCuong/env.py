import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class RandomHazardEnv(gym.Env):
    def __init__(self, grid_size=6, hazard_type="chasing", num_hazards=1, has_smoke_item=True):
        super().__init__()
        self.grid_size = grid_size
        self.hazard_type = hazard_type  
        self.num_hazards = num_hazards
        self.has_smoke_item = has_smoke_item
        
        self.action_space = spaces.Discrete(5)
        
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.smoke_pos = (grid_size // 2, grid_size // 2) if has_smoke_item else None
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start_pos)
        
        if self.num_hazards == 1:
            self.hazards_pos = [[self.grid_size - 1, 0]]
        else:
            self.hazards_pos = [[self.grid_size - 1, 0], [0, self.grid_size - 1]]
            
        self.has_smoke = 1 if self.has_smoke_item else 0
        self.smoke_timer = 0
        
        return self._get_state(), {}

    def _get_state(self):
        
        state = [self.agent_pos[0], self.agent_pos[1]]
        for h in self.hazards_pos:
            state.extend([h[0], h[1]])
        state.append(self.has_smoke)
        state.append(self.smoke_timer)
        return tuple(state)

    def _get_valid_moves(self, pos):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid = []
        for dr, dc in directions:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                valid.append((nr, nc))
        return valid

    def _move_hazard(self, h_pos):
        valid_moves = self._get_valid_moves(h_pos)
        
        if self.smoke_timer > 0 or self.hazard_type == "random":
            probs = [1.0 / len(valid_moves)] * len(valid_moves)
        elif self.hazard_type == "chasing":
            dists = [abs(m[0] - self.agent_pos[0]) + abs(m[1] - self.agent_pos[1]) for m in valid_moves]
            min_dist = min(dists)
            
            weights = []
            for d in dists:
                if d == min_dist:
                    weights.append(0.6)
                else:
                    weights.append(0.4 / max(1, (len(valid_moves) - 1)))
            
            total_w = sum(weights)
            probs = [w / total_w for w in weights]

        idx = self.np_random.choice(len(valid_moves), p=probs)
        return list(valid_moves[idx])

    def step(self, action):
        # 1. Di chuyển Agent
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        dr, dc = moves[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            self.agent_pos = [nr, nc]

        # 2. Xử lý ăn Smoke bomb
        if self.smoke_pos and tuple(self.agent_pos) == self.smoke_pos and self.has_smoke == 1:
            self.has_smoke = 0
            self.smoke_timer = 3  # Kích hoạt hiệu lực 3 bước
        elif self.smoke_timer > 0:
            self.smoke_timer -= 1

        # 3. Di chuyển các Hazard
        for i in range(len(self.hazards_pos)):
            self.hazards_pos[i] = self._move_hazard(self.hazards_pos[i])

        # 4. Reward & Terminal condition
        reward = -1  # Step penalty
        terminated = False

        # Kiểm tra va chạm
        for h in self.hazards_pos:
            if self.agent_pos == h:
                reward = -100
                terminated = True
                return self._get_state(), reward, terminated, False, {}

        # Đến Goal
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 100
            terminated = True

        return self._get_state(), reward, terminated, False, {}
